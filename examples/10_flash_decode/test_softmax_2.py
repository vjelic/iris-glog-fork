#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import iris

# --- Kernel Definitions (Unchanged) ---

# Kernel to compute softmax on a local slice of data.
@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride_x_row, stride_y_row, num_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0)
    row_x_ptr = x_ptr + row_idx * stride_x_row
    row_y_ptr = y_ptr + row_idx * stride_y_row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_cols
    row_data = tl.load(row_x_ptr + col_offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(row_data, axis=0)
    numerator = tl.exp(row_data - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(row_y_ptr + col_offsets, softmax_output, mask=mask)

# Kernel to "put" (push) data to a remote GPU. Used for the final gather.
@triton.jit
def put_kernel(local_source_ptr, remote_dest_ptr, num_elements, current_rank, remote_rank, heap_bases_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    iris.put(local_source_ptr + offsets, remote_dest_ptr + offsets, current_rank, remote_rank, heap_bases_ptr, mask=mask)

# --- New Kernel for the Scatter Phase ---

# Kernel to "get" (pull) data from a remote GPU. Used for the initial scatter.
@triton.jit
def get_kernel(remote_source_ptr, local_dest_ptr, num_elements, current_rank, remote_rank, heap_bases_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    iris.get(remote_source_ptr + offsets, local_dest_ptr + offsets, current_rank, remote_rank, heap_bases_ptr, mask=mask)


def main():
    """
    Main function to orchestrate the distributed softmax.
    """
    shmem = iris.iris()
    current_rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    print(f"[Rank {current_rank}] Initializing distributed softmax for a pre-existing matrix.")

    # --- Configuration ---
    Q = 256  # Total rows in the matrix
    K = 4096 # Total columns
    dtype = torch.float32
    master_rank = 0

    # For simplicity, we assume the number of rows is perfectly divisible by the number of GPUs.
    if Q % world_size != 0:
        if current_rank == master_rank:
            print(f"ERROR: Total rows ({Q}) must be divisible by the world size ({world_size}).")
        return
    
    ROWS_PER_GPU = Q // world_size

    # --- Triton Kernel Configurations ---
    SOFTMAX_BLOCK_SIZE = triton.next_power_of_2(K)
    softmax_grid = (ROWS_PER_GPU,)

    SLICE_NUMEL = ROWS_PER_GPU * K
    COMM_BLOCK_SIZE = 1024
    comm_grid = lambda meta: (triton.cdiv(SLICE_NUMEL, meta["BLOCK_SIZE"]),)

    # --- Step 0: Data Loading and Allocation ---
    # The full input matrix X will exist on the master_rank.
    # All other ranks will have an empty placeholder for it.
    x_global = shmem.empty((Q, K), dtype=dtype)

    if current_rank == master_rank:
        print(f"[Rank {master_rank}] Creating the master input matrix X.")
        # Create a sample matrix. In a real application, you would load your data here.
        master_X = torch.randn((Q, K), dtype=dtype, device=shmem.get_device())
        x_global.copy_(master_X)

    # Each rank allocates a local buffer to hold its slice of the input matrix.
    x_slice = shmem.empty((ROWS_PER_GPU, K), dtype=dtype)
    # Each rank allocates the full output buffer.
    y_global = shmem.zeros((Q, K), dtype=dtype)
    
    shmem.barrier()

    # --- Step 1: Scatter ---
    # Each GPU pulls its assigned slice of rows from the master rank.
    print(f"[Rank {current_rank}] Scattering input: Getting my slice from Rank {master_rank}.")
    
    # Calculate which slice of the global matrix this rank is responsible for.
    my_slice_in_global_x = x_global[current_rank * ROWS_PER_GPU : (current_rank + 1) * ROWS_PER_GPU, :]
    
    get_kernel[comm_grid](
        my_slice_in_global_x, # The remote source on the master GPU
        x_slice,              # The local destination
        SLICE_NUMEL,
        current_rank,
        master_rank,
        shmem.get_heap_bases(),
        BLOCK_SIZE=COMM_BLOCK_SIZE,
    )
    shmem.barrier()

    # --- Step 2: Compute ---
    # Each GPU computes softmax on its local slice.
    print(f"[Rank {current_rank}] Computing local softmax...")
    my_slice_in_global_y = y_global[current_rank * ROWS_PER_GPU : (current_rank + 1) * ROWS_PER_GPU, :]
    
    softmax_kernel[softmax_grid](
        x_slice,
        my_slice_in_global_y,
        x_slice.stride(0),
        y_global.stride(0),
        K,
        BLOCK_SIZE=SOFTMAX_BLOCK_SIZE,
    )
    shmem.barrier()

    # --- Step 3: Gather ---
    # Each GPU sends its computed slice to all other GPUs.
    print(f"[Rank {current_rank}] Gathering results via all-to-all...")
    for remote_rank in range(world_size):
        if remote_rank == current_rank:
            continue
        
        local_source_slice = y_global[current_rank * ROWS_PER_GPU : (current_rank + 1) * ROWS_PER_GPU, :]
        remote_dest_slice = y_global[current_rank * ROWS_PER_GPU : (current_rank + 1) * ROWS_PER_GPU, :]

        put_kernel[comm_grid](
            local_source_slice,
            remote_dest_slice,
            SLICE_NUMEL,
            current_rank,
            remote_rank,
            shmem.get_heap_bases(),
            BLOCK_SIZE=COMM_BLOCK_SIZE,
        )
    shmem.barrier()
    print(f"[Rank {current_rank}] Process finished.")

    # --- Final Validation (on master rank) ---
    if current_rank == master_rank:
        print(f"\n[Rank {master_rank}] Validating final result...")
        # The master rank already has the original full matrix `x_global`.
        expected_y = torch.softmax(x_global, dim=1)
        is_correct = torch.allclose(y_global, expected_y, atol=1e-5, rtol=1e-4)

        if is_correct:
            print("✅ Validation Successful! The distributed softmax result is correct.")
        else:
            print("❌ Validation FAILED! The result is incorrect.")
    
    shmem.barrier()

if __name__ == "__main__":
    main()
