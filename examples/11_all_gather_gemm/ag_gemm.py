# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from examples.common.utils import read_realtime

import sys
import os

import iris

# --- HELPER KERNELS ---
# This helper is from the original scatter example and is needed for the persistent GEMM structure.
@triton.jit
def tile_id_to_index_range(
    tile_id, M, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    # ... (code is identical to the provided file)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    tile_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (tile_in_group % group_size_m)
    pid_n = tile_in_group // group_size_m
    rm_start = pid_m * BLOCK_SIZE_M
    rn_start = pid_n * BLOCK_SIZE_N
    rm = rm_start + tl.arange(0, BLOCK_SIZE_M)
    rn = rn_start + tl.arange(0, BLOCK_SIZE_N)
    return rm, rn, rm_start, rn_start

@triton.jit
def push_kernel_bytes(
    local_staging_ptr,
    gathered_buffer_ptr,
    signal_flags_ptr,
    heap_bases_ptr,
    bytes_to_send,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    stride_gathered_rank_bytes: tl.constexpr,
    BLOCK_SIZE_BYTES: tl.constexpr,
):
    """
    Pushes this rank's local data from a staging buffer to all other ranks.
    """
    dest_rank = tl.program_id(0)
    local_ptr = tl.cast(local_staging_ptr, tl.pointer_type(tl.int8))
    gathered_ptr = tl.cast(gathered_buffer_ptr, tl.pointer_type(tl.int8))
    write_offset = my_rank * stride_gathered_rank_bytes
    
    # --- FIX IS HERE: Replaced the for loop with a while loop ---
    # This is the correct way to loop with a runtime boundary (bytes_to_send).
    offset = 0
    while offset < bytes_to_send:
        b_offsets = offset + tl.arange(0, BLOCK_SIZE_BYTES)
        b_mask = b_offsets < bytes_to_send
        data_block = tl.load(local_ptr + b_offsets, mask=b_mask, other=0)
        dest_ptr = gathered_ptr + write_offset + b_offsets
        iris.store(dest_ptr, data_block, my_rank, dest_rank, heap_bases_ptr, mask=b_mask)
        # Manually increment the offset for the next iteration
        offset += BLOCK_SIZE_BYTES

    iris.atomic_add(signal_flags_ptr + dest_rank, 1, my_rank, dest_rank, heap_bases_ptr, sem="release", scope="sys")

# --- MAIN KERNEL: Refactored for All-Gather + GEMM ---
@triton.jit()
def fused_ag_gemm_kernel(
    # --- MODIFIED SIGNATURE for All-Gather + GEMM ---
    gathered_act_ptr, # Input: Buffer with gathered activation shards
    B,                # Input: Local weight matrix
    C,                # Output: Local result matrix
    bias_ptr,         # Input: Optional bias vector
    signal_flags_ptr, # Input: Flags for synchronization
    M, N, K,
    stride_gathered_rank_bytes: tl.constexpr,
    # Explicit strides are more robust than assuming dense packing
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bias,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    K_local: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K_local: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    # --- NEW OPTIMIZATION ARGUMENTS ---
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K_local: tl.constexpr,
    heap_bases_ptr: tl.tensor, # Renamed for consistency
    COLLECT_TIMESTAMPS: tl.constexpr = False,
    mm_begin_timestamp_ptr: tl.tensor = None,
    mm_end_timestamp_ptr: tl.tensor = None,
):
    pid = tl.program_id(0)
    # --- OPTIMIZATION: Multi-XCD aware PID mapping ---
    # Remaps PIDs for better cache locality on multi-chiplet (XCD) GPUs
    if NUM_XCDS > 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    
    # --- OPTIMIZATION: Flexible accumulator type for mixed precision ---
    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    # --- BLOCKING WAIT (Functionality unchanged) ---
    # This core logic remains. The kernel waits for the All-Gather to complete.
    current_arrivals = 0
    while current_arrivals < world_size:
        current_arrivals = tl.atomic_cas(signal_flags_ptr + my_rank, -1, -1, sem="acquire", scope="sys")

    # --- PERSISTENT LOOP (Functionality unchanged) ---
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # --- OPTIMIZATION: Optional timestamping for profiling ---
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_min(mm_begin_timestamp_ptr + tile_id, timestamp)

        # Get 2D coordinates for the output tile
        rm, rn, rm_start, rn_start = tile_id_to_index_range(tile_id, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        # --- NEW LOGIC: Loop over gathered shards and compute partial GEMM ---
        for source_rank in range(world_size):
            # Pointer to the activation shard from the current source_rank
            act_shard_base_ptr = gathered_act_ptr + source_rank * stride_gathered_rank_bytes
            act_shard_ptr_typed = tl.cast(act_shard_base_ptr, tl.pointer_type(C.type.element_ty))
            
            # Pointer to the corresponding slice of the local weight matrix B
            W_slice_ptr = B + (source_rank * K_local) * stride_bk

            # --- OPTIMIZATION: Handle uneven K-dimension efficiently ---
            # This avoids using masks in the main loop for better performance.
            loop_k = tl.cdiv(K_local, BLOCK_SIZE_K_local)
            if not EVEN_K_local:
                loop_k -= 1
            
            for k_offset_idx in range(0, loop_k):
                k_offset = k_offset_idx * BLOCK_SIZE_K_local
                rk_local = k_offset + tl.arange(0, BLOCK_SIZE_K_local)
                
                # Load activation shard (Matrix A) - no mask needed here
                A_ptr = act_shard_ptr_typed + rm[:, None] * stride_am + rk_local[None, :] * stride_ak
                a = tl.load(A_ptr)

                # Load weight slice (Matrix B) - no mask needed here
                B_ptr = W_slice_ptr + rk_local[:, None] * stride_bk + rn[None, :] * stride_bn
                b = tl.load(B_ptr)
                
                acc += tl.dot(a, b)
            
            # Cleanup loop for the final, potentially masked, K-block
            if not EVEN_K_local:
                k_offset = loop_k * BLOCK_SIZE_K_local
                rk_local = k_offset + tl.arange(0, BLOCK_SIZE_K_local)

                # Load activation shard (Matrix A) with mask
                A_ptr = act_shard_ptr_typed + rm[:, None] * stride_am + rk_local[None, :] * stride_ak
                A_mask = (rk_local[None, :] < K_local)
                a = tl.load(A_ptr, mask=A_mask, other=0.0)

                # Load weight slice (Matrix B) with mask
                B_ptr = W_slice_ptr + rk_local[:, None] * stride_bk + rn[None, :] * stride_bn
                B_mask = (rk_local[:, None] < K_local)
                b = tl.load(B_ptr, mask=B_mask, other=0.0)
                
                acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)

        # --- OPTIMIZATION: Optional bias addition ---
        if BIAS:
            bias = tl.load(bias_ptr + rn, mask=rn < N)
            c = c + bias

        # --- LOCAL STORE (Functionality unchanged) ---
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_ptr, c, mask=c_mask)

        # --- OPTIMIZATION: Optional timestamping for profiling ---
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(mm_end_timestamp_ptr + tile_id, timestamp)