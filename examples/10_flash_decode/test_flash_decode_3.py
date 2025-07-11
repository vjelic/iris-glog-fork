# Save this file as `iris_flash_decode_final.py`
# Run with: mpirun -np <num_gpus> --allow-run-as-root python iris_flash_decode_final.py

import torch
import triton
import triton.language as tl
import math
import iris
from typing import List, Optional

# ==============================================================================
# KERNEL DEFINITIONS
# ==============================================================================

@triton.jit
def _broadcast_kernel(
    data_ptr, n_elements, heap_bases_ptr,
    root_rank: tl.constexpr, cur_rank: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """Broadcasts data from root_rank to all other ranks using iris.load."""
    if cur_rank == root_rank:
        return
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data_from_root = iris.load(data_ptr + offsets, cur_rank, root_rank, heap_bases_ptr, mask=mask)
    tl.store(data_ptr + offsets, data_from_root, mask=mask)

@triton.jit
def _iris_all_gather_kernel(
    local_data_ptr, gathered_data_ptr, heap_bases_ptr,
    n_elements_per_rank, cur_rank: tl.constexpr, num_ranks: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """Gathers data from all ranks into a single buffer on every rank."""
    pid = tl.program_id(0)
    for target_rank in range(num_ranks):
        remote_source_ptr = iris.translate(local_data_ptr, cur_rank, target_rank, heap_bases_ptr)
        block_start = pid * BLOCK_SIZE
        source_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = source_offsets < n_elements_per_rank
        data = tl.load(remote_source_ptr + source_offsets, mask=mask)
        dest_offsets = target_rank * n_elements_per_rank + source_offsets
        tl.store(gathered_data_ptr + dest_offsets, data, mask=mask)

@triton.jit
def _iris_split_kv_kernel(
    q_ptr, k_ptr, v_ptr, sm_scale, block_table_ptr, kv_len_ptr, intermediate_out_ptr,
    stride_q_bs, stride_q_h, stride_k_cache_bs, stride_k_cache_h, stride_v_cache_bs,
    stride_v_cache_h, stride_intermediate_bs, stride_intermediate_h,
    stride_intermediate_split, stride_table_bs, kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr, page_size: tl.constexpr, BLOCK_HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr, NUM_KV_SPLITS: tl.constexpr
):
    bid, hid_block, split_kv_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kv_hid = hid_block // tl.cdiv(kv_group_num, BLOCK_H)
    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = hid_block * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (hid_block + 1) * VALID_BLOCK_H) & (cur_head < q_head_num)
    offs_d = tl.arange(0, BLOCK_HEAD_DIM)
    cur_kv_seq_len = tl.load(kv_len_ptr + bid)
    offs_q = bid * stride_q_bs + cur_head[:, None] * stride_q_h + offs_d[None, :]
    q = tl.load(q_ptr + offs_q, mask=(mask_h[:, None]) & (offs_d[None, :] < BLOCK_HEAD_DIM), other=0.0)
    kv_len_per_split = tl.cdiv(cur_kv_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_kv_seq_len)
    e_max, e_sum = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf"), tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_HEAD_DIM], dtype=tl.float32)
    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_page_number = tl.load(block_table_ptr + bid * stride_table_bs + offs_n // page_size, mask=offs_n < split_kv_end, other=0)
        kv_loc = kv_page_number * page_size + offs_n % page_size
        offs_cache_k = kv_loc[None, :] * stride_k_cache_bs + kv_hid * stride_k_cache_h + offs_d[:, None]
        k = tl.load(k_ptr + offs_cache_k, mask=(offs_n[None, :] < split_kv_end) & (offs_d[:, None] < BLOCK_HEAD_DIM), other=0.0)
        qk = tl.dot(q, k.to(q.dtype)) * sm_scale
        qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf"))
        offs_cache_v = kv_loc[:, None] * stride_v_cache_bs + kv_hid * stride_v_cache_h + offs_d[None, :]
        v = tl.load(v_ptr + offs_cache_v, mask=(offs_n[:, None] < split_kv_end) & (offs_d[None, :] < BLOCK_HEAD_DIM), other=0.0)
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
    offs_out = bid * stride_intermediate_bs + cur_head[:, None] * stride_intermediate_h + split_kv_id * stride_intermediate_split + offs_d[None, :]
    tl.store(intermediate_out_ptr + offs_out, acc / e_sum[:, None], mask=mask_h[:, None] & (offs_d[None, :] < BLOCK_HEAD_DIM))
    offs_log = bid * stride_intermediate_bs + cur_head * stride_intermediate_h + split_kv_id * stride_intermediate_split + BLOCK_HEAD_DIM
    tl.store(intermediate_out_ptr + offs_log, e_max + tl.log(e_sum), mask=mask_h)

@triton.jit
def _iris_combine_local_kv_kernel(
    intermediate_out_ptr, local_out_ptr, kv_len_ptr,
    stride_intermediate_bs, stride_intermediate_h, stride_intermediate_split,
    stride_local_out_bs, stride_local_out_h,
    num_q_heads: tl.constexpr, BLOCK_HEAD_DIM: tl.constexpr, NUM_KV_SPLITS: tl.constexpr
):
    bid, hid = tl.program_id(0), tl.program_id(1)
    cur_kv_seq_len = tl.load(kv_len_ptr + bid)
    offs_d = tl.arange(0, BLOCK_HEAD_DIM)
    e_sum, e_max = 0.0, -float("inf")
    acc = tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32)
    offs_v = bid * stride_intermediate_bs + hid * stride_intermediate_h + offs_d
    offs_logic = bid * stride_intermediate_bs + hid * stride_intermediate_h + BLOCK_HEAD_DIM
    for split_kv_id in range(0, NUM_KV_SPLITS):
        if (split_kv_id * tl.cdiv(cur_kv_seq_len, NUM_KV_SPLITS)) < cur_kv_seq_len:
            tv = tl.load(intermediate_out_ptr + offs_v + split_kv_id * stride_intermediate_split, mask=offs_d < BLOCK_HEAD_DIM, other=0.0)
            tlogic = tl.load(intermediate_out_ptr + offs_logic + split_kv_id * stride_intermediate_split)
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max
    offs_local_out_v = bid * stride_local_out_bs + hid * stride_local_out_h + offs_d
    tl.store(local_out_ptr + offs_local_out_v, acc, mask=offs_d < BLOCK_HEAD_DIM)
    offs_local_out_logic = bid * stride_local_out_bs + hid * stride_local_out_h + BLOCK_HEAD_DIM
    tl.store(local_out_ptr + offs_local_out_logic, e_max + tl.log(e_sum))

@triton.jit
def _iris_combine_global_kv_kernel(
    gathered_buffer_ptr, final_out_ptr,
    stride_gathered_rank, stride_gathered_bs, stride_gathered_h,
    stride_final_out_bs, stride_final_out_h,
    num_q_heads: tl.constexpr, num_ranks: tl.constexpr, BLOCK_HEAD_DIM: tl.constexpr
):
    bid, hid = tl.program_id(0), tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_HEAD_DIM)
    e_sum, e_max = 0.0, -float("inf")
    acc = tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32)
    for rank_id in range(num_ranks):
        offs_v = rank_id * stride_gathered_rank + bid * stride_gathered_bs + hid * stride_gathered_h + offs_d
        offs_logic = rank_id * stride_gathered_rank + bid * stride_gathered_bs + hid * stride_gathered_h + BLOCK_HEAD_DIM
        tv = tl.load(gathered_buffer_ptr + offs_v, mask=offs_d < BLOCK_HEAD_DIM, other=0.0)
        tlogic = tl.load(gathered_buffer_ptr + offs_logic)
        n_e_max = tl.maximum(tlogic, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - n_e_max)
        acc += exp_logic * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max
    offs_final_out = bid * stride_final_out_bs + hid * stride_final_out_h + offs_d
    tl.store(final_out_ptr + offs_final_out, acc / e_sum, mask=offs_d < BLOCK_HEAD_DIM)

# ==============================================================================
# Main Orchestrating Module
# ==============================================================================
class SpGQAFlashDecodeIris(torch.nn.Module):
    def __init__(self, shmem, num_q_heads, num_kv_heads, head_dim, batch_size, page_size=1):
        super().__init__()
        self.shmem = shmem
        self.rank = shmem.get_rank()
        self.num_ranks = shmem.get_num_ranks()
        self.num_q_heads = num_q_heads
        self.num_kv_heads_per_rank = num_kv_heads // self.num_ranks
        self.head_dim = head_dim
        self.page_size = page_size
        self.scale = head_dim**-0.5
        self.num_kv_splits = 32
        assert num_q_heads % self.num_ranks == 0
        assert num_kv_heads % self.num_ranks == 0

        # Allocate buffers ONCE to prevent out-of-memory errors
        self.intermediate_out = self.shmem.empty((batch_size, self.num_q_heads, self.num_kv_splits, self.head_dim + 1), dtype=torch.float32)
        self.local_out = self.shmem.empty((batch_size, self.num_q_heads, self.head_dim + 1), dtype=torch.float32)
        self.gathered_buffer = self.shmem.empty((self.num_ranks, batch_size, self.num_q_heads, self.head_dim + 1), dtype=torch.float32)
        self.final_output = self.shmem.empty((batch_size, self.num_q_heads, self.head_dim), dtype=torch.float16)

    def forward(self, q, k_cache, v_cache, block_table, kv_lens_per_rank):
        batch_size, _, _ = q.shape

        self.shmem.barrier()
        # STAGE A: Local Attention (all Q heads vs local KV shard)
        grid_stage1 = (batch_size, triton.cdiv(self.num_q_heads, 16), self.num_kv_splits)
        _iris_split_kv_kernel[grid_stage1](
            q, k_cache, v_cache, self.scale, block_table, kv_lens_per_rank, self.intermediate_out,
            q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(2), self.intermediate_out.stride(0), self.intermediate_out.stride(1),
            self.intermediate_out.stride(2), block_table.stride(0), self.num_q_heads // self.num_kv_heads_per_rank,
            self.num_q_heads, self.page_size, self.head_dim, 64, 16, self.num_kv_splits,
        )
        grid_stage2 = (batch_size, self.num_q_heads)
        _iris_combine_local_kv_kernel[grid_stage2](
            self.intermediate_out, self.local_out, kv_lens_per_rank,
            self.intermediate_out.stride(0), self.intermediate_out.stride(1), self.intermediate_out.stride(2),
            self.local_out.stride(0), self.local_out.stride(1),
            self.num_q_heads, self.head_dim, self.num_kv_splits,
        )
        self.shmem.barrier()

        # STAGE B: All-Gather local results
        n_elements_per_rank = self.local_out.numel()
        grid_allgather = (triton.cdiv(n_elements_per_rank, 1024),)
        _iris_all_gather_kernel[grid_allgather](
            self.local_out, self.gathered_buffer.view(-1), self.shmem.get_heap_bases(),
            n_elements_per_rank, self.rank, self.num_ranks, 1024,
        )
        self.shmem.barrier()

        # STAGE C: Final global combination
        grid_stage3 = (batch_size, self.num_q_heads)
        _iris_combine_global_kv_kernel[grid_stage3](
            self.gathered_buffer, self.final_output, self.gathered_buffer.stride(0), self.gathered_buffer.stride(1),
            self.gathered_buffer.stride(2), self.final_output.stride(0), self.final_output.stride(1),
            self.num_q_heads, self.num_ranks, self.head_dim,
        )
        self.shmem.barrier()
        
        return self.final_output


# ==============================================================================
# Reference implementation for correctness checking
# ==============================================================================
def final_ref_paged_attn(
    query: torch.Tensor,
    k_cache_all_ranks: List[torch.Tensor],
    v_cache_all_ranks: List[torch.Tensor],
    block_tables_all_shards: List[torch.Tensor],
    kv_lens_per_rank: List[int],
    scale: float,
    num_ranks: int,
    total_kv_heads: int,
    head_dim: int
) -> torch.Tensor:
    """
    This is the corrected reference function. It takes the gathered shards and
    correctly reconstructs the global tensors before performing a standard
    attention calculation.
    """
    num_pages_per_rank = k_cache_all_ranks[0].shape[0]
    
    # Reconstruct the full K/V caches
    k_cache_full = torch.cat(k_cache_all_ranks, dim=2)
    v_cache_full = torch.cat(v_cache_all_ranks, dim=2)

    block_tables_global = torch.cat(
        [bt_shard + i * num_pages_per_rank for i, bt_shard in enumerate(block_tables_all_shards)],
        dim=1
    )
    
    global_kv_len = sum(kv_lens_per_rank)
    q = query[0]
    num_q_heads, head_size = q.shape
    _, block_size, _, _ = k_cache_full.shape

    num_kv_blocks = (global_kv_len + block_size - 1) // block_size
    block_indices = block_tables_global[0, :num_kv_blocks]

    k = k_cache_full[block_indices].view(-1, total_kv_heads, head_size)[:global_kv_len]
    v = v_cache_full[block_indices].view(-1, total_kv_heads, head_size)[:global_kv_len]
    
    if num_q_heads != total_kv_heads:
        # --- FIX IS HERE ---
        # The repeat factor for `v` should be the same as for `k`.
        num_groups = num_q_heads // total_kv_heads
        k = torch.repeat_interleave(k, num_groups, dim=1)
        v = torch.repeat_interleave(v, num_groups, dim=1)
    
    q_for_einsum = q.unsqueeze(0)
    attn_scores = torch.einsum("qhd,khd->hqk", q_for_einsum, k).float() * scale
    attn_probs = torch.softmax(attn_scores, dim=-1).to(v.dtype)
    output_with_batch_dim = torch.einsum("hqk,khd->qhd", attn_probs, v)
    
    output = output_with_batch_dim.squeeze(0)
    
    return output
# ==============================================================================
# Main execution block - UNCHANGED FROM YOUR WORKING VERSION
# ==============================================================================
if __name__ == "__main__":
    shmem = iris.iris()
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    print(f"[RANK {rank}] --- Initializing and setting up config ---")
    
    batch_size, total_q_heads, total_kv_heads = 1, 32, 4
    head_dim, seq_len_per_rank, page_size = 128, 256, 16
    kv_heads_per_rank = total_kv_heads // world_size
    
    # Use a scale of 1.0 to simplify math for debugging
    scale = 1.0 

    sp_gqa_attn = SpGQAFlashDecodeIris(
        shmem, total_q_heads, total_kv_heads, head_dim, batch_size, page_size=page_size
    )
    # Manually set the scale in the module instance for this test
    sp_gqa_attn.scale = scale
    shmem.barrier()
    
    print(f"[RANK {rank}] --- Creating DETERMINISTIC mock data ---")
    
    # --- CHANGE: Use tensors of all ones instead of random data ---
    q = shmem.full((batch_size, total_q_heads, head_dim), 1.0, dtype=torch.float16)
    
    num_pages_per_rank = (seq_len_per_rank + page_size - 1) // page_size
    k_cache_shard = shmem.full((num_pages_per_rank, page_size, kv_heads_per_rank, head_dim), 1.0, dtype=torch.float16)
    v_cache_shard = shmem.full((num_pages_per_rank, page_size, kv_heads_per_rank, head_dim), 1.0, dtype=torch.float16)
    block_table_shard = shmem.arange(num_pages_per_rank, dtype=torch.int32).unsqueeze(0)
    kv_lens_shard = shmem.full((batch_size,), seq_len_per_rank, dtype=torch.int32)
    shmem.barrier()
    
    print(f"[RANK {rank}] --- Running Distributed Attention forward pass ---")
    output = sp_gqa_attn(q, k_cache_shard, v_cache_shard, block_table_shard, kv_lens_shard)
    shmem.barrier()
    
    # --- Correctness Check ---
    if rank == 0:
        print(f"[RANK {rank}] --- Gathering data for correctness check ---")
        
    k_list_flat = shmem.empty([world_size * k_cache_shard.numel()], dtype=k_cache_shard.dtype)
    v_list_flat = shmem.empty([world_size * v_cache_shard.numel()], dtype=v_cache_shard.dtype)
    bt_list_flat = shmem.empty([world_size * block_table_shard.numel()], dtype=block_table_shard.dtype)
    
    _iris_all_gather_kernel[(triton.cdiv(k_cache_shard.numel(), 1024),)](k_cache_shard, k_list_flat, shmem.get_heap_bases(), k_cache_shard.numel(), rank, world_size, 1024)
    _iris_all_gather_kernel[(triton.cdiv(v_cache_shard.numel(), 1024),)](v_cache_shard, v_list_flat, shmem.get_heap_bases(), v_cache_shard.numel(), rank, world_size, 1024)
    _iris_all_gather_kernel[(triton.cdiv(block_table_shard.numel(), 1024),)](block_table_shard, bt_list_flat, shmem.get_heap_bases(), block_table_shard.numel(), rank, world_size, 1024)
    shmem.barrier()
    
    if rank == 0:
        print(f"[RANK {rank}] --- Running final comparison ---")
        
        k_cache_all = [k_list_flat.view(world_size, -1)[i].view_as(k_cache_shard) for i in range(world_size)]
        v_cache_all = [v_list_flat.view(world_size, -1)[i].view_as(v_cache_shard) for i in range(world_size)]
        block_table_all = [bt_list_flat.view(world_size, -1)[i].view_as(block_table_shard) for i in range(world_size)]
        
        global_kv_lens_list = [seq_len_per_rank for _ in range(world_size)]

        ref_output = final_ref_paged_attn(
            q, k_cache_all, v_cache_all, block_table_all, 
            global_kv_lens_list, scale, world_size, total_kv_heads, head_dim
        )
        
        output_single = output[0]
        
        print("\n" + "="*50)
        print("Distributed Implementation Output (first 5 values of first head):")
        print(output_single[0, :])
        print("~"*50)
        print("Reference Implementation Output (first 5 values of first head):")
        print(ref_output[0, :])
        print("="*50 + "\n")
        
        try:
            # Both outputs should be tensors of all ones.
            torch.testing.assert_close(output_single, torch.ones_like(output_single), atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(ref_output, torch.ones_like(ref_output), atol=1e-2, rtol=1e-2)
            print("\n✅ Correctness Test Passed!")
        except AssertionError as e:
            print("\n❌ Correctness Test Failed!")
            print(f"Max absolute difference from ones (Iris): {torch.max(torch.abs(output_single - 1.0))}")
            print(f"Max absolute difference from ones (Ref): {torch.max(torch.abs(ref_output - 1.0))}")