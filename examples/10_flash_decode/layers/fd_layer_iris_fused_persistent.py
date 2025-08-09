import torch
import triton
import math
import iris

# Import the two kernels that will be launched sequentially
from kernels.decode_kernels import (
    gqa_persistent_split_k_and_push,
    gqa_global_reduce_wait_tile
)

class SpGQAFlashDecodeAttentionIrisFusedPersistent(torch.nn.Module):
    def __init__(self, iris_instance, rank, node, num_ranks, num_nodes, num_q_heads, num_kv_heads, q_head_dim, v_head_dim, page_size=1,
                 scale=1, soft_cap=0, max_allowed_batch=1, num_sms=132):
        super().__init__()
        self.iris_instance = iris_instance
        self.rank = rank
        self.num_ranks = num_ranks
        self.node = node
        self.num_nodes = num_nodes
        self.num_sms = num_sms

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.q_head_dim = q_head_dim
        self.v_head_dim = v_head_dim
        self.page_size = page_size
        self.soft_cap = soft_cap
        self.scale = scale
        self.kv_split = 32
        self.max_allowed_batch = max_allowed_batch
        
        self.BLOCK_DV = triton.next_power_of_2(self.v_head_dim)

        self.gathered_buffer = self.iris_instance.empty(
            (self.num_ranks, self.max_allowed_batch, self.num_q_heads, self.v_head_dim + 1),
            dtype=torch.float16
        )
        self.signal_flags = self.iris_instance.zeros(
            (self.num_ranks, self.num_ranks, self.max_allowed_batch, self.num_q_heads), dtype=torch.int32
        )
        self.workspace = self.iris_instance.empty((self.num_sms,), dtype=torch.int32)

    def clear_flags(self):
        """Resets synchronization flags for the next iteration."""
        self.signal_flags.zero_()
        self.workspace.zero_()
        self.iris_instance.barrier()

    def forward(self, q, k_cache, v_cache, global_kv_lens, block_table):
        batch = q.shape[0]
        assert batch <= self.max_allowed_batch

        # Intermediate buffer for the split-k results
        output_split = torch.empty(
            [batch, self.num_q_heads, self.kv_split, self.v_head_dim + 1], dtype=torch.float32, device=q.device
        )
        
        # --- Kernel Launch 1: Producer ---
        # Persistent kernel that performs split-k, local combine, and pushes to all ranks.
        producer_grid = (self.num_sms,)
        gqa_persistent_split_k_and_push[producer_grid](
            q, k_cache, v_cache, output_split, self.gathered_buffer,
            self.scale, block_table, global_kv_lens, self.signal_flags, self.workspace,
            self.iris_instance.get_heap_bases(),
            batch,
            q.stride(0), q.stride(1),
            k_cache.stride(-3), k_cache.stride(-2),
            v_cache.stride(-3), v_cache.stride(-2),
            output_split.stride(0), output_split.stride(1), output_split.stride(2),
            self.gathered_buffer.stride(0), self.gathered_buffer.stride(1), self.gathered_buffer.stride(2),
            block_table.stride(0),
            self.signal_flags.stride(0), self.signal_flags.stride(1), self.signal_flags.stride(2), self.signal_flags.stride(3),
            self.num_q_heads // self.num_kv_heads, self.num_q_heads,
            2**int(math.log2(self.q_head_dim)), self.q_head_dim - 2**int(math.log2(self.q_head_dim)),
            self.BLOCK_DV, 64, 16, self.kv_split, self.page_size, self.soft_cap,
            self.q_head_dim, self.v_head_dim,
            self.rank, self.num_ranks
        )

        # --- Kernel Launch 2: Consumer ---
        # Standard kernel that waits for signals and performs the final global reduction.
        final_output = torch.empty([batch, self.num_q_heads, self.v_head_dim], dtype=q.dtype, device=q.device)
        consumer_grid = (batch, self.num_q_heads)
        gqa_global_reduce_wait_tile[consumer_grid](
            self.gathered_buffer, final_output, global_kv_lens, self.signal_flags,
            self.signal_flags.stride(0), self.signal_flags.stride(1), self.signal_flags.stride(2), self.signal_flags.stride(3),
            batch, self.num_q_heads,
            self.gathered_buffer.stride(1), self.gathered_buffer.stride(2), self.gathered_buffer.stride(0),
            final_output.stride(0), final_output.stride(1),
            self.rank, self.num_ranks,
            self.BLOCK_DV, self.v_head_dim,
        )
        
        return final_output