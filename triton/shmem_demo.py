import torch

import triton
import triton.language as tl
import pyrocSHMEM as pyshmem
import ctypes


@triton.jit
def add_two(first, second):
    return first + second


@triton.jit
def copy_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    heap_bases,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    cur_rank = 0
    dst_rank = 0
    input_data = pyshmem.get(input_ptr + offsets, cur_rank, dst_rank, heap_bases, mask)

    output_data = input_data
    # if pid == 0:
    #     print("cur_rank: ", input_data)
    pyshmem.put(
        output_ptr + offsets, output_data, cur_rank, dst_rank, heap_bases, mask=mask
    )


def main():
    print("Starting...")

    heap_size = 1 << 30
    shmem = pyshmem.pyrocSHMEM(heap_size)

    torch.manual_seed(0)
    size = 8
    block_size = size

    size_in_bytes = torch.tensor([], dtype=torch.int).element_size()

    # Allocate input on symmetric heap
    input_data = shmem.arange(size, dtype=torch.int)

    output_data = shmem.zeros_like(input_data)

    heap_bases = shmem.get_heap_bases()

    print(f"heap_bases: {heap_bases}")

    for i, value in enumerate(heap_bases):
        print(f"Element {i}: {hex(value.item())}")
    print(f"input_data: {hex(input_data.data_ptr())}")
    print(f"output_data: {hex(output_data.data_ptr())}")

    DEVICE = shmem.get_device()
    print(f"Device: {DEVICE}")
    assert input_data.device == DEVICE and output_data.device == DEVICE

    n_elements = output_data.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    copy_kernel[grid](
        input_data, output_data, n_elements, heap_bases, BLOCK_SIZE=block_size
    )

    torch.cuda.synchronize()
    print(f"expected: {input_data}")
    print(f"output  : {output_data}")
    print(
        f"The maximum difference between source and destination is "
        f"{torch.max(torch.abs(input_data - output_data))}"
    )


if __name__ == "__main__":
    main()
