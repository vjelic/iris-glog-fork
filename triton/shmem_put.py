import torch
import triton
import triton.language as tl
import pyrocSHMEM as pyshmem
import ctypes


@triton.jit
def copy_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    heap_bases,
    cur_rank: tl.constexpr,
    to_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_data = pyshmem.get(
        input_ptr + offsets,
        cur_rank,
        cur_rank,
        heap_bases,
        mask,
    )

    output_data = input_data
    pyshmem.put(
        output_ptr + offsets,
        output_data,
        cur_rank,
        to_rank,
        heap_bases,
        mask=mask,
    )


def main():

    heap_size = 1 << 30
    shmem = pyshmem.pyrocSHMEM(heap_size)

    num_ranks = shmem.get_num_ranks()
    cur_rank = shmem.get_rank()

    shmem.log("Starting")

    # Check if the number of ranks is 2
    if num_ranks != 2:
        raise RuntimeError(
            f"This program requires exactly 2 ranks, but {num_ranks} were found."
        )

    torch.manual_seed(0)
    size = 2
    block_size = size

    size_in_bytes = torch.tensor([], dtype=torch.int).element_size()

    # Allocate input and output on the symmetric heap
    input_data = shmem.arange(size, dtype=torch.int)
    output_data = shmem.zeros_like(input_data)

    shmem.log(f"input_data : {input_data.data_ptr():#x}")
    shmem.log(f"output_data: {output_data.data_ptr():#x}")

    if cur_rank == 0:
        # Rank 0 sends data
        heap_bases = shmem.get_heap_bases()
        shmem.log(f"heap_bases: {heap_bases}")

        for i, value in enumerate(heap_bases):
            shmem.log(f"Element {i}: {hex(value.item())}")
        shmem.log(f"input_data: {hex(input_data.data_ptr())}")
        shmem.log(f"output_data: {hex(output_data.data_ptr())}")

        device = shmem.get_device()
        shmem.log(f"Device: {device}")
        assert input_data.device == device and output_data.device == device

        n_elements = output_data.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        # Launch Triton kernel
        copy_kernel[grid](
            input_data,
            output_data,
            n_elements,
            heap_bases,
            cur_rank=0,
            to_rank=1,
            BLOCK_SIZE=block_size,
        )

    # Wait for kernels to finish
    torch.cuda.synchronize()
    # Sync the world
    shmem.barrier()

    if cur_rank == 1:
        # Rank 1 receives data
        shmem.log(
            f"The maximum difference between source and destination is "
            f"{torch.max(torch.abs(input_data - output_data))}"
        )
        shmem.log(f"expected: {input_data}")
        shmem.log(f"output  : {output_data}")

    torch.cuda.synchronize()
    shmem.barrier()
    # shmem.log(f"expected: {input_data}")
    if cur_rank == 1:
        shmem.log(f"output  : {output_data}")


if __name__ == "__main__":
    main()
