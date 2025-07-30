import torch
import torch.distributed as dist
from decode_kernels import (gqa_fwd_batch_decode_intra_rank, kernel_inter_rank_gqa_fwd_batch_decode_combine_kv)

class SpGQAFlashDecodeAttentionRCCL2(torch.nn.Module):
    def __init__(self, rank: int, num_ranks: int, num_q_heads: int, num_kv_heads: int, q_head_dim: int, v_head_dim: int, 
                 process_group, page_size: int = 1, scale: float = 1.0, soft_cap: float = 0.0, max_allowed_batch: int = 1):
        super().__init__()
        self.rank = rank
        self.num_ranks = num_ranks
        self.process_group = process_group

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.q_head_dim = q_head_dim
        self.v_head_dim = v_head_dim
        self.page_size = page_size
        self.soft_cap = soft_cap
        self.scale = scale
        
        self.kv_split = 32
        self.max_allowed_batch = max_allowed_batch
        
        # --- OPTIMIZATION: Create a separate stream for the final combine kernel ---
        self.combine_stream = torch.cuda.Stream()

    def forward(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, global_kv_lens: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        batch_size = q.shape[0]

        # Assertions remain the same...
        assert global_kv_lens.shape[0] == self.num_ranks
        assert global_kv_lens.shape[1] == batch_size
        assert batch_size <= self.max_allowed_batch

        # Intermediate and final tensors
        output_combine = torch.empty(
            [batch_size, self.num_q_heads, self.v_head_dim + 1], dtype=q.dtype, device=q.device
        )
        final_output = torch.empty(
            [batch_size, self.num_q_heads, self.v_head_dim], dtype=q.dtype, device=q.device
        )
        
        # 1. Launch local computation on the default stream
        gqa_fwd_batch_decode_intra_rank(
            q, k_cache, v_cache,
            workspace=None,
            q_lens=[1] * batch_size,
            kv_lens=global_kv_lens[self.rank],
            block_table=block_table,
            scale=self.scale,
            soft_cap=self.soft_cap,
            output_combine=output_combine,
            kv_split=self.kv_split
        )

        # 2. --- OPTIMIZATION: Use a non-blocking all_gather ---
        all_ranks_output_combine = [
            torch.empty_like(output_combine) for _ in range(self.num_ranks)
        ]
        work_handle = dist.all_gather(
            all_ranks_output_combine,
            output_combine,
            group=self.process_group,
            async_op=True  # This makes the call non-blocking
        )
        
        # 3. Launch the final combination kernel on its own stream
        with torch.cuda.stream(self.combine_stream):
            # This stream must wait until the communication is complete
            work_handle.wait()
            
            # Now that data is ready, we can combine it
            # We need to stack the list of tensors into a single tensor for the kernel
            all_ranks_output_tensor = torch.stack(all_ranks_output_combine)

            kernel_inter_rank_gqa_fwd_batch_decode_combine_kv[(batch_size, self.num_q_heads, 1)](
                all_ranks_output_tensor,
                final_output,
                global_kv_lens,
                batch_size,
                self.num_q_heads,
                all_ranks_output_tensor.stride(1),  # stride_mid_ob
                all_ranks_output_tensor.stride(2),  # stride_mid_oh
                all_ranks_output_tensor.stride(0),  # stride_mid_os
                final_output.stride(0),              # stride_obs
                final_output.stride(1),              # stride_oh
                self.num_ranks,                      # NUM_KV_SPLITS
                512,                                 # BLOCK_DV 
                self.v_head_dim,                     # Lv
            )

        # 4. Synchronize the default stream with the combine stream before returning
        torch.cuda.current_stream().wait_stream(self.combine_stream)
        
        return final_output