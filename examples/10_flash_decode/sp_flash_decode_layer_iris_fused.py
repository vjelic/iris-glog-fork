import torch
import os
import triton
import numpy as np
import iris
from decode_kernels import (gqa_fwd_batch_decode_intra_rank, kernel_fused_wait_and_combine)
from all_gather_layer_fused import IrisAllGatherLayerFused

class SpGQAFlashDecodeAttentionIrisAGFused(torch.nn.Module):

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

        self.BLOCK_DV = triton.next_power_of_2(self.v_head_dim)
        self.iris_instance = iris_instance

        self.count_less_than_half = 0
        self.shrink_buffer_threshold = thrink_buffer_threshold

    
        max_inter_nbytes = max_allowed_batch * self.num_q_heads * (self.v_head_dim + 1) * 2
        max_ag_buffer_size = max_inter_nbytes * self.num_ranks
        
        self.iris_ag_layer = IrisAllGatherLayerFused(
            self.iris_instance,
            max_buffer_size=max_ag_buffer_size,
            dtype=torch.float16,
        )

    def forward(self, q, k_cache, v_cache, global_kv_lens, block_table):
        batch = q.shape[0]
        assert global_kv_lens.shape[0] == self.num_ranks
        assert global_kv_lens.shape[1] == batch
        assert batch <= self.max_allowed_batch

        output_split = torch.empty([batch, self.num_q_heads, self.kv_split, self.v_head_dim + 1], dtype=q.dtype, device=q.device)
        output_combine = torch.empty([batch, self.num_q_heads, self.v_head_dim + 1], dtype=q.dtype, device=q.device)
        final_output = torch.empty([batch, self.num_q_heads, self.v_head_dim], dtype=q.dtype, device=q.device)

        gqa_fwd_batch_decode_intra_rank(
            q, k_cache, v_cache, self.workspace, [1] * q.shape[0],
            global_kv_lens[self.rank], block_table, self.scale, soft_cap=self.soft_cap,
            output_split=output_split, output_combine=output_combine,
            kv_split=self.kv_split
        )
        
        self.iris_ag_layer.push_data(output_combine.contiguous())

        ag_buffer = self.iris_ag_layer.gathered_buffer
        
        all_ranks_output_combine = ag_buffer.view(q.dtype).view(
            self.num_ranks, batch, self.num_q_heads, self.v_head_dim + 1
        )

        kernel_fused_wait_and_combine[(batch, self.num_q_heads, 1)](
            all_ranks_output_combine,
            final_output,
            global_kv_lens,
            self.iris_ag_layer.signal_flags,
            batch,
            self.num_q_heads,
            all_ranks_output_combine.stride(1),    
            all_ranks_output_combine.stride(2),   
            all_ranks_output_combine.stride(0),   
            final_output.stride(0),                
            final_output.stride(1),                
            self.rank,                             
            self.num_ranks,                        
            self.BLOCK_DV,                         
            self.v_head_dim,                       
        )

        return final_output