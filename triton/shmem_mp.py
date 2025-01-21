import torch
import triton
import triton.language as tl
import pyrocSHMEM as pyshmem


@triton.jit
def producer_kernel(
    input_ptr,
    output_ptr,
    flag_ptr,
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

    # Produce data (copy from input to output)
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

    # Set the flag to signal Rank 1
    if pid == 0:
        # Why doesn't CAS  take mask?
        # Is this a scalar operation? one per block?
        pyshmem.atomic_cas(flag_ptr,
                             0, # compare
                             1, # value
                             cur_rank,
                             to_rank,
                             heap_bases,
                             sem = "release",
                             scope = "sys"
                             )


@triton.jit
def consumer_kernel(
    output_ptr,
    flag_ptr,
    n_elements,
    heap_bases,
    cur_rank: tl.constexpr,
    from_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Spin until the flag is set to 1
    # Is this a single CAS per block?
    result = 0
    while result == 0:
        result = pyshmem.atomic_cas(flag_ptr,
                                    1, # compare
                                    0, # value
                                    cur_rank,
                                    cur_rank,
                                    heap_bases,
                                    sem = "acquire",
                                    scope = "sys"
                                    )
        pass
        # print("result: ", result)


    # Consume data (read from output)
    output_data = pyshmem.get(
        output_ptr + offsets,
        from_rank,
        cur_rank,
        heap_bases,
        mask,
    )
    pyshmem.put(
        output_ptr + offsets,
        output_data,
        from_rank,
        cur_rank,
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
        raise RuntimeError(f"This program requires exactly 2 ranks, but {num_ranks} were found.")

    torch.manual_seed(0)
    size = 128  # Should deadlock if we use more than 1 block.
    block_size = 128

    # Allocate input, output, and flag on the symmetric heap
    input_data = shmem.arange(size, dtype=torch.int8)
    output_data = shmem.zeros_like(input_data)
    flag = shmem.zeros(1, dtype=torch.int)

    shmem.log(f"input_data : {input_data.data_ptr():#x}")
    shmem.log(f"output_data: {output_data.data_ptr():#x}")
    shmem.log(f"flag       : {flag.data_ptr():#x}")

    if cur_rank == 0:
        # Rank 0 produces data
        heap_bases = shmem.get_heap_bases()
        n_elements = output_data.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        # Launch producer kernel
        producer_kernel[grid](
            input_data,
            output_data,
            flag,
            n_elements,
            heap_bases,
            cur_rank=0,
            to_rank=1,
            BLOCK_SIZE=block_size,
        )
    elif cur_rank == 1:
        # Rank 1 consumes data
        heap_bases = shmem.get_heap_bases()
        n_elements = output_data.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        # Launch consumer kernel
        consumer_kernel[grid](
            output_data,
            flag,
            n_elements,
            heap_bases,
            cur_rank=1,
            from_rank=0,
            BLOCK_SIZE=block_size,
        )

    # Wait for kernels to finish
    torch.cuda.synchronize()
    shmem.barrier()

    # Verify data on Rank 1
    if cur_rank == 1:
        shmem.log(
            f"The maximum difference between source and destination is "
            f"{torch.max(torch.abs(input_data - output_data))}"
        )
        shmem.log(f"expected: {input_data}")
        shmem.log(f"output  : {output_data}")

    shmem.barrier()

    shmem.log(f"Flag: {flag}")

if __name__ == "__main__":
    main()