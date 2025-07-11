#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import iris

# This kernel computes a numerically stable softmax for each row of a 2D tensor.
@triton.jit
def softmax_kernel(
    x_ptr,               # Pointer to the input tensor slice.
    y_ptr,               # Pointer to the output tensor slice.
    stride_x_row,        # Stride to move from one row to the next in the input tensor.
    stride_y_row,        # Stride to move from one row to the next in the output tensor.
    num_cols,            # Number of columns in the matrix.
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance computes the softmax for one row of its assigned slice.
    row_idx = tl.program_id(axis=0)

    # Pointers for the current row.
    row_x_ptr = x_ptr + row_idx * stride_x_row
    row_y_ptr = y_ptr + row_idx * stride_y_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_cols
    row_data = tl.load(row_x_ptr + col_offsets, mask=mask, other=-float('inf'))

    # Numerically stable softmax calculation
    row_max = tl.max(row_data, axis=0)
    numerator = tl.exp(row_data - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Store the result back to the output tensor slice.
    tl.store(row_y_ptr + col_offsets, softmax_output, mask=mask)


# A generic kernel to "put" (push) a contiguous block of data
# from a local buffer to a remote buffer.
@triton.jit
def put_kernel(
    local_source_ptr,
    remote_dest_ptr,
    num_elements,
    current_rank,
    remote_rank,
    heap_bases_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    # iris.put is the primitive that performs a local load and a remote store.
    iris.put(local_source_ptr + offsets, remote_dest_ptr + offsets, current_rank, remote_rank, heap_bases_ptr, mask=mask)


def main():
    """
    Main function to run distributed softmax and gather results.
    """
    shmem = iris.iris()
    current_rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    print(f"[Rank {current_rank}] Initializing distributed softmax test with {world_size} GPUs.")

    # --- Configuration ---
    NUM_ROWS_PER_GPU = 64
    NUM_COLS = 4096
    TOTAL_ROWS = NUM_ROWS_PER_GPU * world_size
    dtype = torch.float32

    # --- Triton Kernel Configuration ---
    SOFTMAX_BLOCK_SIZE = triton.next_power_of_2(NUM_COLS)
    softmax_grid = (NUM_ROWS_PER_GPU,)

    PUT_BLOCK_SIZE = 1024
    slice_numel = NUM_ROWS_PER_GPU * NUM_COLS
    put_grid = lambda meta: (triton.cdiv(slice_numel, meta["BLOCK_SIZE"]),)

    # --- Memory Allocation & Data Creation ---
    # Each rank creates its own unique, but predictable, slice of the input data.
    # This avoids issues with random number generation in a distributed setting.
    slice_data = torch.arange(NUM_COLS, device="cuda", dtype=dtype).unsqueeze(0).repeat(NUM_ROWS_PER_GPU, 1)
    slice_data += current_rank * 100 # Add a large offset to make each rank's data unique.
    
    x_slice = shmem.empty((NUM_ROWS_PER_GPU, NUM_COLS), dtype=dtype)
    x_slice.copy_(slice_data)

    # Each rank allocates a buffer for the FULL global result.
    # This buffer will be populated by gathering results from all other GPUs.
    y_global = shmem.zeros((TOTAL_ROWS, NUM_COLS), device="cuda", dtype=dtype)

    shmem.barrier()

    # --- Step 1: Local Computation ---
    # Each GPU computes softmax on its own slice and stores it in the
    # correct location within its local y_global buffer.
    print(f"[Rank {current_rank}] Computing local softmax...")

    # Get a view of the tensor slice this rank is responsible for.
    y_slice_for_local_write = y_global[current_rank * NUM_ROWS_PER_GPU : (current_rank + 1) * NUM_ROWS_PER_GPU, :]

    softmax_kernel[softmax_grid](
        x_slice,
        y_slice_for_local_write,
        x_slice.stride(0),
        y_global.stride(0), # Use the stride of the larger global tensor
        NUM_COLS,
        BLOCK_SIZE=SOFTMAX_BLOCK_SIZE,
    )

    # Wait for all local computations to finish.
    shmem.barrier()

    # --- Step 2: All-Gather Communication ---
    # Each GPU now sends its computed slice to every other GPU.
    print(f"[Rank {current_rank}] Starting all-gather to distribute results...")
    for remote_rank in range(world_size):
        if remote_rank == current_rank:
            continue # Don't need to send data to ourselves.

        # The data we are sending is our computed slice.
        local_source_slice = y_global[current_rank * NUM_ROWS_PER_GPU : (current_rank + 1) * NUM_ROWS_PER_GPU, :]

        # *** FIX IS HERE ***
        # The destination on the remote GPU is the slice corresponding to our (current) rank.
        # We get a pointer to this slice from our OWN y_global tensor.
        # Iris correctly translates this "local" pointer to the "remote" address.
        remote_dest_slice = y_global[current_rank * NUM_ROWS_PER_GPU : (current_rank + 1) * NUM_ROWS_PER_GPU, :]

        put_kernel[put_grid](
            local_source_slice,
            remote_dest_slice,
            slice_numel,
            current_rank,
            remote_rank,
            shmem.get_heap_bases(),
            BLOCK_SIZE=PUT_BLOCK_SIZE,
        )

    # Wait for all communication to complete.
    shmem.barrier()
    print(f"[Rank {current_rank}] All-gather finished.")

    # --- Step 3: Validation ---
    # Now, every GPU should have the full, identical result in y_global.
    print(f"[Rank {current_rank}] Validating the final gathered result...")

    # Reconstruct the full input tensor based on our predictable data generation scheme.
    full_x = torch.empty((TOTAL_ROWS, NUM_COLS), device='cuda', dtype=dtype)
    for i in range(world_size):
        slice_data = torch.arange(NUM_COLS, device="cuda", dtype=dtype).unsqueeze(0).repeat(NUM_ROWS_PER_GPU, 1)
        slice_data += i * 100
        full_x[i * NUM_ROWS_PER_GPU:(i + 1) * NUM_ROWS_PER_GPU, :] = slice_data
    
    expected_y = torch.softmax(full_x, dim=1)
    is_correct = torch.allclose(y_global, expected_y, atol=1e-5, rtol=1e-4)

    if is_correct:
        print(f"\n[Rank {current_rank}] ✅ Validation Successful! The globally gathered result is correct.")
    else:
        print(f"\n[Rank {current_rank}] ❌ Validation FAILED! The final result is incorrect.")

    shmem.barrier()

if __name__ == "__main__":
    main()
