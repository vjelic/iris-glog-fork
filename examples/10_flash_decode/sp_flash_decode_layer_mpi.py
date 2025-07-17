import torch
import os
import numpy as np
import iris
from decode_kernels import (gqa_fwd_batch_decode_intra_rank, kernel_inter_rank_gqa_fwd_batch_decode_combine_kv)


class SpGQAFlashDecodeAttentionMPI(torch.nn.Module):

    def __init__(self, iris_instance, rank, node, num_ranks, num_nodes, num_q_heads, num_kv_heads, q_head_dim, v_head_dim, page_size=1,
                 scale=1, soft_cap=0, max_allowed_batch=1, thrink_buffer_threshold=500, stages=20):
        super().__init__()
        self.rank = rank
        self.num_ranks = num_ranks
        self.node = node
        self.num_nodes = num_nodes

        self.workspace = None  # gqa_fwd doesn't need workspace
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.q_head_dim = q_head_dim
        self.v_head_dim = v_head_dim
        self.page_size = page_size
        self.soft_cap = soft_cap
        self.scale = scale
        self.kv_split = 32
        self.max_allowed_batch = max_allowed_batch
        self.stages = stages

        self.iris_instance = iris_instance

        self.count_less_than_half = 0
        self.shrink_buffer_threshold = thrink_buffer_threshold

    
        max_output_combine_elements = max_allowed_batch * num_q_heads * (v_head_dim + 1)
        self.iris_allgather_buffer_elements = max_output_combine_elements * self.num_ranks
        self.iris_allgather_buffer = self.iris_instance.allocate(
            self.iris_allgather_buffer_elements, dtype=torch.float16
        )

    def forward(self, q, k_cache, v_cache, global_kv_lens, block_table):
        """
        q: each rank has the same q
        k_cache: each rank's shard of k_cache
        v_cache: each rank's shard of v_cache
        global_kv_lens: all the rank's kv shard's length
        block_table: each rank's kv shard's kv_table
        """
        batch = q.shape[0]
        assert global_kv_lens.shape[0] == self.num_ranks
        assert global_kv_lens.shape[1] == batch
        assert batch <= self.max_allowed_batch, f"Only support {self.max_allowed_batch} queries decode now"
        output_split = torch.empty([batch, self.num_q_heads, self.kv_split, self.v_head_dim + 1], dtype=q.dtype,
                                   device=q.device)
        output_combine = torch.empty([batch, self.num_q_heads, self.v_head_dim + 1], dtype=q.dtype, device=q.device)
        final_output = torch.empty([batch, self.num_q_heads, self.v_head_dim], dtype=q.dtype, device=q.device)

        
        gqa_fwd_batch_decode_intra_rank(q, k_cache, v_cache, self.workspace, [1] * q.shape[0],
                                        global_kv_lens[self.rank], block_table, self.scale, soft_cap=self.soft_cap,
                                        output_split=output_split, output_combine=output_combine,
                                        kv_split=self.kv_split)
        ################
        # allgather part
        
        output_combine_numpy_flattened = output_combine.cpu().numpy().astype(np.float32).flatten()
        all_gathered_numpy_flattened = iris._mpi_helpers.mpi_allgather(output_combine_numpy_flattened)


        all_ranks_output_combine = torch.from_numpy(all_gathered_numpy_flattened).to(q.device).view(
            self.num_ranks, batch, self.num_q_heads, self.v_head_dim + 1
        ).to(q.dtype)

        ################
        # final combine
        kernel_inter_rank_gqa_fwd_batch_decode_combine_kv[(batch, self.num_q_heads, 1)](
            all_ranks_output_combine, final_output, global_kv_lens, batch, self.num_q_heads,
            all_ranks_output_combine.stride(1),  # batch
            all_ranks_output_combine.stride(2),  # head
            all_ranks_output_combine.stride(0),  # num_ranks
            final_output.stride(0),  # batch
            final_output.stride(1),  # head
            self.num_ranks,  # split_kv
            512,  # BLOCK_DV
            self.v_head_dim,  # Lv
        )

        return final_output