#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import iris


# A generic kernel to "put" (push) a contiguous block of data
# from a local buffer to a remote buffer. This is the core communication
# primitive we will use to build the all-gather.
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
    # Standard Triton boilerplate to parallelize the copy.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # iris.put is the key operation. It reads from the local_source_ptr
    # and writes to the remote_dest_ptr on the specified remote_rank.
    iris.put(
        local_source_ptr + offsets, remote_dest_ptr + offsets, current_rank, remote_rank, heap_bases_ptr, mask=mask
    )


def main():
    """
    Main function to orchestrate the all-gather operation and validate the result.
    """
    # Initialize Iris, which sets up MPI and the shared memory heaps.
    # It also calls set_device() for each rank.
    shmem = iris.iris()
    current_rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    print(f"[Rank {current_rank}] Initializing all-gather test with {world_size} GPUs.")

    # --- Configuration ---
    ELEMENTS_PER_GPU = 2048
    TOTAL_ELEMENTS = ELEMENTS_PER_GPU * world_size
    dtype = torch.int32

    # --- Triton Kernel Configuration ---
    PUT_BLOCK_SIZE = 1024
    put_grid = lambda meta: (triton.cdiv(ELEMENTS_PER_GPU, meta["BLOCK_SIZE"]),)

    # --- Memory Allocation & Data Creation ---
    # Each rank creates its own unique, but predictable, slice of data.
    # The `device` argument is removed, as torch will use the active device set by Iris.
    start_value = current_rank * 100
    local_data_torch = torch.arange(start_value, start_value + ELEMENTS_PER_GPU, dtype=dtype)

    # Copy the torch-created data into an Iris-managed buffer.
    local_slice = shmem.empty((ELEMENTS_PER_GPU,), dtype=dtype)
    local_slice.copy_(local_data_torch)

    # Each rank allocates a buffer for the FULL global result.
    # The device is implicitly handled by the shmem object.
    global_buffer = shmem.zeros((TOTAL_ELEMENTS,), dtype=dtype)

    # --- Step 1: Place local data into the global buffer ---
    # Each GPU copies its own local_slice into the correct position
    # within its own global_buffer.
    my_slice_in_global = global_buffer[current_rank * ELEMENTS_PER_GPU : (current_rank + 1) * ELEMENTS_PER_GPU]
    my_slice_in_global.copy_(local_slice)

    # Wait for all GPUs to finish this initial copy before communicating.
    shmem.barrier()

    # --- Step 2: All-to-All Communication ---
    # Each GPU now sends its slice of data to every other GPU.
    print(f"[Rank {current_rank}] Starting all-to-all data exchange...")
    for dest_rank in range(world_size):
        if dest_rank == current_rank:
            continue  # No need to send data to ourselves.

        # The destination for our data on the remote GPU is the slice
        # corresponding to our own rank. We get a pointer to this slice
        # from our *own* global_buffer. Iris handles the translation.
        remote_dest_slice = global_buffer[current_rank * ELEMENTS_PER_GPU : (current_rank + 1) * ELEMENTS_PER_GPU]

        # Launch the kernel to push our local_slice to the dest_rank.
        put_kernel[put_grid](
            local_slice,
            remote_dest_slice,
            ELEMENTS_PER_GPU,
            current_rank,
            dest_rank,
            shmem.get_heap_bases(),
            BLOCK_SIZE=PUT_BLOCK_SIZE,
        )

    # Wait for all communication to complete.
    shmem.barrier()
    print(f"[Rank {current_rank}] All-gather communication finished.")

    # --- Step 3: Validation ---
    # At this point, the `global_buffer` on every GPU should be identical
    # and contain the concatenated data from all ranks.
    print(f"[Rank {current_rank}] Validating the final gathered result...")

    # Reconstruct the full expected result locally for comparison.
    expected_result = torch.empty((TOTAL_ELEMENTS,), dtype=dtype)
    for i in range(world_size):
        start_val = i * 100
        slice_data = torch.arange(start_val, start_val + ELEMENTS_PER_GPU, dtype=dtype)
        expected_result[i * ELEMENTS_PER_GPU : (i + 1) * ELEMENTS_PER_GPU] = slice_data

    # Move the expected result to the correct GPU for comparison
    expected_result = expected_result.to(shmem.get_device())

    # Use torch.equal for an exact match, since we are using integers.
    is_correct = torch.equal(global_buffer, expected_result)

    if is_correct:
        print(f"\n[Rank {current_rank}] ✅ Validation Successful! The all-gather operation was correct.")
    else:
        print(f"\n[Rank {current_rank}] ❌ Validation FAILED! The final result is incorrect.")
        # Find the first mismatch to help with debugging.
        mismatched_indices = torch.nonzero(global_buffer != expected_result).squeeze()
        if mismatched_indices.numel() > 0:
            first_mismatch_idx = mismatched_indices[0].item()
            print(f"   First mismatch at index {first_mismatch_idx}:")
            print(f"   Expected: {expected_result[first_mismatch_idx].item()}")
            print(f"   Received: {global_buffer[first_mismatch_idx].item()}")

    shmem.barrier()


if __name__ == "__main__":
    main()
