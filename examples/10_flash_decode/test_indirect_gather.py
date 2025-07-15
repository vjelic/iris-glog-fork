#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import iris


# This kernel is run by the 'gatherer' GPU.
# It uses an index map on a remote GPU to gather data from that same remote GPU.
@triton.jit
def indirect_gather_kernel(
    source_data_ptr,  # Remote pointer to the source data array.
    index_map_ptr,  # Remote pointer to the index map.
    local_result_ptr,  # Local pointer to store the gathered data.
    num_elements,
    gatherer_rank,  # The rank of this GPU, the one doing the gathering.
    data_holder_rank,  # The rank of the GPU that holds the data and index map.
    heap_bases_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Step 1: Load the indices from the remote index map.
    # We tell iris.load to get data from the 'data_holder_rank'.
    indices_to_fetch = iris.load(index_map_ptr + offsets, gatherer_rank, data_holder_rank, heap_bases_ptr, mask=mask)

    # Step 2: Use the fetched indices to perform an indirect load from the remote source data.
    # The 'indices_to_fetch' are used as offsets into the source_data_ptr.
    # This is the "indirect gather" step.
    gathered_values = tl.load(source_data_ptr + indices_to_fetch, mask=mask)

    # Step 3: Store the gathered values into our local result buffer.
    tl.store(local_result_ptr + offsets, gathered_values, mask=mask)


# A simple utility kernel to push data from a local buffer to a remote buffer.
@triton.jit
def put_kernel(
    local_source_ptr,
    remote_destination_ptr,
    num_elements,
    current_rank,
    remote_rank,
    heap_bases_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    iris.put(
        local_source_ptr + offsets,
        remote_destination_ptr + offsets,
        current_rank,
        remote_rank,
        heap_bases_ptr,
        mask=mask,
    )


def main():
    """
    Main function to orchestrate the indirect gather and validation.
    """
    # Initialize Iris and get rank information.
    shmem = iris.iris()
    current_rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    if world_size != 2:
        if current_rank == 0:
            print(f"ERROR: This script requires exactly 2 GPUs, but found {world_size}.")
        return

    # --- Configuration ---
    data_holder_rank = 0  # GPU 0 holds the original data and indices.
    gatherer_rank = 1  # GPU 1 performs the gather operation.
    buffer_size = 16384
    dtype = torch.float32
    index_dtype = torch.int32
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(buffer_size, meta["BLOCK_SIZE"]),)

    # --- Memory Allocation (using shmem for cross-GPU access) ---
    # On GPU 0: The source data to be gathered from.
    source_data = shmem.arange(buffer_size, device="cuda", dtype=dtype)

    # *** FIX IS HERE ***
    # The iris.arange wrapper doesn't support start/stop/step arguments.
    # To work around this, we create the tensor with torch.arange first,
    # then copy it into an Iris-managed buffer.
    if current_rank == data_holder_rank:
        torch_index_map = torch.arange(buffer_size - 1, -1, -1, device="cuda", dtype=index_dtype)
        index_map = shmem.empty(torch_index_map.shape, dtype=index_dtype)
        index_map.copy_(torch_index_map)
    else:
        # Other ranks just need an empty placeholder with the right shape and type.
        index_map = shmem.empty((buffer_size,), dtype=index_dtype)

    # On GPU 1: The buffer where the gathered data will be stored.
    gathered_result_buffer = shmem.zeros(buffer_size, device="cuda", dtype=dtype)
    # On GPU 0: A buffer to receive the final result for validation.
    validation_buffer = shmem.zeros(buffer_size, device="cuda", dtype=dtype)

    shmem.barrier()

    # --- Execution ---
    # Step 1: The 'gatherer' GPU runs the kernel to pull data.
    if current_rank == gatherer_rank:
        print(f"[Rank {current_rank}] Launching kernel to gather data from Rank {data_holder_rank}...")
        indirect_gather_kernel[grid](
            source_data,
            index_map,
            gathered_result_buffer,
            buffer_size,
            gatherer_rank,
            data_holder_rank,
            shmem.get_heap_bases(),
            BLOCK_SIZE=BLOCK_SIZE,
        )

    shmem.barrier()

    # Step 2: The 'gatherer' GPU sends its result back to the 'data_holder' for validation.
    if current_rank == gatherer_rank:
        print(f"[Rank {current_rank}] Sending my result back to Rank {data_holder_rank} for validation...")
        put_kernel[grid](
            gathered_result_buffer,  # Local source
            validation_buffer,  # Remote destination
            buffer_size,
            gatherer_rank,
            data_holder_rank,
            shmem.get_heap_bases(),
            BLOCK_SIZE=BLOCK_SIZE,
        )

    shmem.barrier()

    # --- Validation with Torch ---
    # Step 3: The 'data_holder' GPU checks if the result it received is correct.
    if current_rank == data_holder_rank:
        print(f"[Rank {current_rank}] Validating the received result using torch...")

        # Create the expected result locally using pure torch.
        # This is what we expect the gatherer to have computed.
        expected_result = source_data[index_map]

        # Use torch.allclose to compare the kernel's output with the expected output.
        is_correct = torch.allclose(validation_buffer, expected_result)

        if is_correct:
            print("\n✅ Validation Successful! The remote indirect gather was correct.")
            print(f"   Original data starts with:  {source_data[:8].tolist()}")
            print(f"   Index map starts with:      {index_map[:8].tolist()}")
            print(f"   Gathered result starts with: {validation_buffer[:8].tolist()}")
        else:
            print("\n❌ Validation FAILED! The result is incorrect.")
            # Find the first mismatch to help with debugging.
            for i in range(buffer_size):
                if not torch.isclose(validation_buffer[i], expected_result[i]):
                    print(f"   Mismatch at index {i}:")
                    print(f"   Expected value: {expected_result[i]}")
                    print(f"   Received value: {validation_buffer[i]}")
                    break

    shmem.barrier()


if __name__ == "__main__":
    main()
