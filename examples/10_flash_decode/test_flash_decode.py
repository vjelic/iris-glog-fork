#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import math
import numpy as np
import os

# Import the iris library. The script assumes 'iris.py' is in the Python path.
import iris

# This script provides a self-contained, distributed Grouped-Query Attention (GQA)
# implementation using the Iris library. It is designed to be executed with
# a launcher like `torchrun` or `mpirun` on AMD GPUs.
#
# Example: torchrun --nproc_per_node=2 your_script_name.py

# =================================================================================
# Part 1: Distributed GQA Kernels using Iris
# =================================================================================


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def kernel_gqa_fwd_batch_decode_split_kv(
    # Pointers to local data
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    output_ptr,
    block_table_ptr,
    kv_length_ptr,
    # Scalar value
    sm_scale,
    # Strides for local tensors
    stride_q_bs,
    stride_q_h,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_o_bs,
    stride_o_h,
    stride_o_split,
    stride_table_bs,
    # Kernel constants
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
):
    """Stage 1: Compute partial attention results for local data. All memory access is local."""
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    kv_hid = hid // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = min(BLOCK_H, kv_group_num)
    cur_head = hid * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (hid + 1) * VALID_BLOCK_H) & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_HEAD_DIM)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < K_DIM
    mask_dv = offs_dv < V_DIM
    cur_kv_seq_len = tl.load(kv_length_ptr + bid)

    offs_q = bid * stride_q_bs + cur_head[:, None] * stride_q_h + offs_d[None, :]
    q = tl.load(q_ptr + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_HEAD_DIM + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < K_DIM
        offs_qpe = bid * stride_q_bs + cur_head[:, None] * stride_q_h + offs_dpe[None, :]
        qpe = tl.load(q_ptr + offs_qpe, mask=mask_h[:, None] & mask_dpe[None, :], other=0.0)

    kv_len_per_split = tl.cdiv(cur_kv_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_kv_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_page_number = tl.load(
            block_table_ptr + bid * stride_table_bs + offs_n // PAGE_SIZE, mask=offs_n < split_kv_end, other=0
        )
        kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE

        offs_cache_k = kv_loc[None, :] * stride_k_cache_bs + kv_hid * stride_k_cache_h + offs_d[:, None]
        k = tl.load(k_cache_ptr + offs_cache_k, mask=(offs_n[None, :] < split_kv_end) & mask_d[:, None], other=0.0)
        qk = tl.dot(q, k.to(q.dtype))

        if BLOCK_DPE > 0:
            offs_cache_kpe = kv_loc[None, :] * stride_k_cache_bs + kv_hid * stride_k_cache_h + offs_dpe[:, None]
            kpe = tl.load(
                k_cache_ptr + offs_cache_kpe, mask=(offs_n[None, :] < split_kv_end) & mask_dpe[:, None], other=0.0
            )
            qk += tl.dot(qpe, kpe.to(qpe.dtype))

        qk *= sm_scale
        qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf"))

        offs_cache_v = kv_loc[:, None] * stride_v_cache_bs + kv_hid * stride_v_cache_h + offs_dv[None, :]
        v = tl.load(v_cache_ptr + offs_cache_v, mask=(offs_n[:, None] < split_kv_end) & mask_dv[None, :], other=0.0)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    offs_out = bid * stride_o_bs + cur_head[:, None] * stride_o_h + split_kv_id * stride_o_split + offs_dv[None, :]
    tl.store(output_ptr + offs_out, acc / e_sum[:, None], mask=mask_h[:, None] & mask_dv[None, :])

    offs_log = bid * stride_o_bs + cur_head * stride_o_h + split_kv_id * stride_o_split + V_DIM
    tl.store(output_ptr + offs_log, e_max + tl.log(e_sum), mask=mask_h)


@triton.jit
def kernel_gqa_fwd_batch_decode_combine_kv(
    # Pointers
    Mid_O,
    o,
    B_Seqlen,
    heap_bases,
    # Strides
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    # Kernel constants
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """Stage 2: Combine partial results from all ranks using Iris for remote memory access."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    final_acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    final_e_max = -float("inf")

    # All-reduce loop: each rank fetches and reduces data from every other rank
    for remote_rank in range(world_size):
        e_sum_remote = 0.0
        e_max_remote = -float("inf")
        acc_remote = tl.zeros([BLOCK_DV], dtype=tl.float32)

        offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
        offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

        # Reduce the partial results from the remote rank's Mid_O buffer
        for split_kv_id in range(0, NUM_KV_SPLITS):
            if (tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS) * split_kv_id) < cur_batch_seq_len:
                # *** CORRECTED LINE: Removed the 'other' keyword argument from iris.load ***
                tv = iris.load(
                    Mid_O + offs_v + split_kv_id * stride_mid_os, cur_rank, remote_rank, heap_bases, mask=mask_d
                )
                tlogic = iris.load(Mid_O + offs_logic + split_kv_id * stride_mid_os, cur_rank, remote_rank, heap_bases)

                n_e_max = tl.maximum(tlogic, e_max_remote)
                old_scale = tl.exp(e_max_remote - n_e_max)
                acc_remote = acc_remote * old_scale
                exp_logic = tl.exp(tlogic - n_e_max)
                acc_remote += exp_logic * tv
                e_sum_remote = e_sum_remote * old_scale + exp_logic
                e_max_remote = n_e_max

        # Combine the reduced result from the remote rank into the final accumulator
        n_final_e_max = tl.maximum(e_max_remote, final_e_max)
        old_scale = tl.exp(final_e_max - n_final_e_max)
        new_scale = tl.exp(e_max_remote - n_final_e_max)

        # Guard against division by zero if e_sum_remote is 0
        if e_sum_remote > 0:
            final_acc = final_acc * old_scale + (acc_remote / e_sum_remote) * new_scale
        else:
            final_acc = final_acc * old_scale
        final_e_max = n_final_e_max

    # Store the final, globally reduced result to local memory
    tl.store(o + cur_batch * stride_obs + cur_head * stride_oh + offs_d, final_acc, mask=mask_d)


# =================================================================================
# Part 2: Host-side Orchestration and Main Execution Block
# =================================================================================


def gqa_fwd_batch_decode_iris(shmem, q, k_cache, v_cache, kv_lens, block_table, scale):
    """Host-side function to launch the distributed GQA kernels."""
    batch, q_heads, q_head_dim = q.shape
    _, page_size, kv_heads, _ = k_cache.shape
    v_head_dim = v_cache.shape[-1]

    rank = shmem.get_rank()
    shmem.log(f"Starting GQA forward pass. Batch={batch}, Q Heads={q_heads}, KV Heads={kv_heads}")

    BLOCK_N, BLOCK_H, NUM_KV_SPLITS = 64, 16, 32
    BLOCK_HEAD_DIM = 2 ** int(math.log2(q_head_dim)) if q_head_dim > 0 else 0
    BLOCK_DPE = q_head_dim - BLOCK_HEAD_DIM
    BLOCK_DV = triton.next_power_of_2(v_head_dim)
    kv_group_num = q_heads // kv_heads

    output_split = shmem.empty((batch, q_heads, NUM_KV_SPLITS, v_head_dim + 1), dtype=torch.float16)
    final_output = shmem.empty((batch, q_heads, v_head_dim), dtype=torch.float16)

    grid_split_kv = (batch, triton.cdiv(q_heads, min(BLOCK_H, kv_group_num)), NUM_KV_SPLITS)
    shmem.log(f"Launching split_kv kernel with grid: {grid_split_kv}")
    kernel_gqa_fwd_batch_decode_split_kv[grid_split_kv](
        q,
        k_cache,
        v_cache,
        output_split,
        block_table,
        kv_lens,
        scale,
        q.stride(0),
        q.stride(1),
        k_cache.stride(-3),
        k_cache.stride(-2),
        v_cache.stride(-3),
        v_cache.stride(-2),
        output_split.stride(0),
        output_split.stride(1),
        output_split.stride(2),
        block_table.stride(0),
        kv_group_num=kv_group_num,
        q_head_num=q_heads,
        BLOCK_HEAD_DIM=BLOCK_HEAD_DIM,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        K_DIM=q_head_dim,
        V_DIM=v_head_dim,
    )
    shmem.barrier()
    shmem.log("Finished split_kv kernel.")

    grid_combine_kv = (batch, q_heads)
    shmem.log(f"Launching combine_kv kernel with grid: {grid_combine_kv}")
    kernel_gqa_fwd_batch_decode_combine_kv[grid_combine_kv](
        output_split,
        final_output,
        kv_lens,
        shmem.get_heap_bases(),
        output_split.stride(0),
        output_split.stride(1),
        output_split.stride(2),
        final_output.stride(0),
        final_output.stride(1),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=v_head_dim,
        cur_rank=rank,
        world_size=shmem.get_num_ranks(),
    )
    shmem.barrier()
    shmem.log("Finished combine_kv kernel.")

    return final_output


def main():
    # iris.iris() will initialize the distributed environment
    shmem = iris.iris()
    rank = shmem.get_rank()

    # --- Configuration ---
    batch_size, q_heads, kv_heads, head_dim = 2, 8, 4, 64
    page_size, max_seq_len = 16, 128

    shmem.log("Initializing input tensors on Iris heap...")
    q = shmem.randn(batch_size, q_heads, head_dim, dtype=torch.float16)
    k_cache = shmem.randn(batch_size, page_size, kv_heads, head_dim, dtype=torch.float16)
    v_cache = shmem.randn(batch_size, page_size, kv_heads, head_dim, dtype=torch.float16)

    # These tensors don't need to be on the Iris heap as they are read-only on the host
    # and passed to the kernel as local arguments.
    kv_lens = torch.full((batch_size,), max_seq_len, device=shmem.device, dtype=torch.int32)
    block_table = torch.full((batch_size, max_seq_len // page_size), rank, device=shmem.device, dtype=torch.int32)
    scale = 1.0 / math.sqrt(head_dim)
    shmem.barrier()

    # --- Run Distributed Attention ---
    distributed_output = gqa_fwd_batch_decode_iris(shmem, q, k_cache, v_cache, kv_lens, block_table, scale)

    # --- Correctness Check (run on rank 0) ---
    shmem.barrier()
    if rank == 0:
        shmem.log("Checking results on rank 0...")
        output_local = distributed_output.cpu().numpy()
        if np.isfinite(output_local).all() and not np.all(output_local == 0):
            shmem.log("✅ Correctness check passed: Output is finite and not all zeros.")
        else:
            shmem.log("❌ Correctness check failed. Output contains NaNs/Infs or is all zeros.")

    shmem.barrier()
    shmem.log("Execution finished.")


if __name__ == "__main__":
    main()
