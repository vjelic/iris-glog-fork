# Save this file as `debug_stage_A.py`
# Run with: mpirun -np 1 python debug_stage_A.py

import torch
import triton
import triton.language as tl
import math
import iris
import argparse

# ==============================================================================
# KERNEL DEFINITIONS FOR STAGE A
# ==============================================================================

@triton.jit
def _iris_split_kv_kernel_stable(
    q_ptr, k_ptr, v_ptr, sm_scale, block_table_ptr, kv_len_ptr, intermediate_out_ptr,
    stride_q_bs, stride_q_h, stride_k_cache_bs, stride_k_cache_h, stride_v_cache_bs,
    stride_v_cache_h, stride_intermediate_bs, stride_intermediate_h,
    stride_intermediate_split, stride_table_bs, kv_group_num: tl.constexpr,
    page_size: tl.constexpr, BLOCK_HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr, NUM_KV_SPLITS: tl.constexpr
):
    """
    A simple and stable Stage 1 kernel that processes one head at a time.
    """
    bid, hid, split_kv_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cur_kv_head = hid // kv_group_num
    
    offs_d = tl.arange(0, BLOCK_HEAD_DIM)
    mask_d = offs_d < BLOCK_HEAD_DIM
    cur_kv_seq_len = tl.load(kv_len_ptr + bid)

    offs_q = bid * stride_q_bs + hid * stride_q_h + offs_d
    q = tl.load(q_ptr + offs_q, mask=mask_d, other=0.0)
    
    kv_len_per_split = tl.cdiv(cur_kv_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_kv_seq_len)
    
    e_max, e_sum = -float("inf"), 0.0
    acc = tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32)

    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < split_kv_end
        
        kv_page_number = tl.load(block_table_ptr + bid * stride_table_bs + offs_n // page_size, mask=mask_n, other=0)
        kv_loc = kv_page_number * page_size + offs_n % page_size
        
        offs_cache_k = kv_loc[:, None] * stride_k_cache_bs + cur_kv_head * stride_k_cache_h + offs_d[None, :]
        k = tl.load(k_ptr + offs_cache_k, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        qk = tl.sum(q[None, :] * k, 1) * sm_scale
        qk = tl.where(mask_n, qk, float("-inf"))
        
        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        
        offs_cache_v = kv_loc[:, None] * stride_v_cache_bs + cur_kv_head * stride_v_cache_h + offs_d[None, :]
        v = tl.load(v_ptr + offs_cache_v, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        acc *= re_scale
        acc += tl.sum(p[:, None] * v, 0)
        e_sum = e_sum * re_scale + tl.sum(p, 0)
        e_max = n_e_max
        
    # Store the un-normalized accumulator and the log-sum-exp statistic
    offs_out = bid * stride_intermediate_bs + hid * stride_intermediate_h + split_kv_id * stride_intermediate_split + offs_d
    tl.store(intermediate_out_ptr + offs_out, acc, mask=mask_d)
    offs_log = bid * stride_intermediate_bs + hid * stride_intermediate_h + split_kv_id * stride_intermediate_split + BLOCK_HEAD_DIM
    tl.store(intermediate_out_ptr + offs_log, e_max + tl.log(e_sum))

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

    offs_v_base = bid * stride_intermediate_bs + hid * stride_intermediate_h + offs_d
    offs_logic_base = bid * stride_intermediate_bs + hid * stride_intermediate_h + BLOCK_HEAD_DIM

    for split_kv_id in range(0, NUM_KV_SPLITS):
        if (split_kv_id * tl.cdiv(cur_kv_seq_len, NUM_KV_SPLITS)) < cur_kv_seq_len:
            offs_v = offs_v_base + split_kv_id * stride_intermediate_split
            offs_logic = offs_logic_base + split_kv_id * stride_intermediate_split
            
            tv = tl.load(intermediate_out_ptr + offs_v, mask=offs_d < BLOCK_HEAD_DIM, other=0.0)
            tlogic = tl.load(intermediate_out_ptr + offs_logic)
            
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    # Store the locally combined but un-normalized accumulator and its statistic
    offs_local_out_v = bid * stride_local_out_bs + hid * stride_local_out_h + offs_d
    tl.store(local_out_ptr + offs_local_out_v, acc, mask=offs_d < BLOCK_HEAD_DIM)
    offs_local_out_logic = bid * stride_local_out_bs + hid * stride_local_out_h + BLOCK_HEAD_DIM
    tl.store(local_out_ptr + offs_local_out_logic, e_max + tl.log(e_sum))

# ==============================================================================
# Test Function for Stage A
# ==============================================================================
def test_stage_A(shmem):
    rank = shmem.get_rank()
    # This test runs on a single GPU
    if rank != 0: return
    print(f"\n--- [RANK {rank}] RUNNING TEST FOR STAGE A (Local Attention) ---")

    # Config
    batch_size, total_q_heads, total_kv_heads = 1, 32, 4
    head_dim, seq_len, page_size, scale = 128, 256, 16, 1.0
    num_kv_splits = 32

    # Data (All ones for deterministic output)
    q = torch.ones((batch_size, total_q_heads, head_dim), dtype=torch.float16, device="cuda")
    num_pages = (seq_len + page_size - 1) // page_size
    k_cache = torch.ones((num_pages, page_size, total_kv_heads, head_dim), dtype=torch.float16, device="cuda")
    v_cache = torch.ones((num_pages, page_size, total_kv_heads, head_dim), dtype=torch.float16, device="cuda")
    block_table = torch.arange(num_pages, dtype=torch.int32, device="cuda").unsqueeze(0)
    kv_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

    # Buffers
    intermediate_out = torch.empty((batch_size, total_q_heads, num_kv_splits, head_dim + 1), dtype=torch.float32, device="cuda")
    local_out = torch.empty((batch_size, total_q_heads, head_dim + 1), dtype=torch.float32, device="cuda")

    # Run Stage A kernels
    grid1 = (batch_size, total_q_heads, num_kv_splits)
    _iris_split_kv_kernel_stable[grid1](
        q, k_cache, v_cache, scale, block_table, kv_lens, intermediate_out,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(2), intermediate_out.stride(0), intermediate_out.stride(1),
        intermediate_out.stride(2), block_table.stride(0), total_q_heads // total_kv_heads,
        page_size, head_dim, 64, num_kv_splits
    )
    
    # In this test, the "local_out" is the final result of the local computation
    final_local_out = torch.empty((batch_size, total_q_heads, head_dim), dtype=q.dtype, device="cuda")
    
    grid2 = (batch_size, total_q_heads)
    _iris_combine_local_kv_kernel[grid2](
        intermediate_out, final_local_out, kv_lens, # We repurpose the combine kernel to get the final normalized output
        intermediate_out.stride(0), intermediate_out.stride(1), intermediate_out.stride(2),
        final_local_out.stride(0), final_local_out.stride(1),
        total_q_heads, head_dim, num_kv_splits
    )

    # For all-ones input, the output should be a vector of ones
    print("Stage A Final Output Vector (first 5 of first head):", final_local_out[0, 0, :5])
    
    try:
        torch.testing.assert_close(final_local_out, torch.ones_like(final_local_out), atol=1e-2, rtol=0)
        print("✅ Stage A Test Passed!")
    except AssertionError:
        print("❌ Stage A Test Failed!")
        print(f"Max absolute difference from ones: {torch.max(torch.abs(final_local_out - 1.0))}")

# ==============================================================================
# Main execution block
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add an argument to select the test case, defaulting to Stage A
    parser.add_argument("--case", type=str, default="test_stage_A", choices=["test_stage_A"])
    args = parser.parse_args()

    shmem = iris.iris()

    if args.case == "test_stage_A":
        test_stage_A(shmem)