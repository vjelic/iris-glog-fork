import triton
import triton.language as tl
import iris

@triton.jit
def iris_allgather_push_kernel(
    local_data_ptr,         
    gathered_output_ptr,    
    signal_flags_ptr,      
    heap_bases_ptr,         
    rank: tl.constexpr,
    world_size: tl.constexpr,
    bytes_per_rank: tl.constexpr,
    stride_output_rank: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    destination_rank = tl.program_id(0)

    my_data_start_ptr = tl.cast(local_data_ptr, tl.pointer_type(tl.int8))
    
    destination_buffer_start_ptr = tl.cast(gathered_output_ptr, tl.pointer_type(tl.int8))
    
    write_offset = rank * stride_output_rank
    
    for B_offset in range(0, bytes_per_rank, BLOCK_SIZE_B):
        b_offsets = B_offset + tl.arange(0, BLOCK_SIZE_B)
        b_mask = b_offsets < bytes_per_rank

        data_block = tl.load(my_data_start_ptr + b_offsets, mask=b_mask)
        
        iris.store(
            destination_buffer_start_ptr + write_offset + b_offsets,
            data_block,
            rank,             
            destination_rank,  
            heap_bases_ptr,
            mask=b_mask,
        )

    iris.atomic_add(signal_flags_ptr + destination_rank, 1, rank, destination_rank, heap_bases_ptr, sem="release", scope="sys")

    
    # while True:
    #     current_arrivals = tl.atomic_cas(signal_flags_ptr + rank, -1, -1, sem="acquire", scope="sys")
    #     if current_arrivals >= world_size:
    #         break
        
    # current_arrivals = tl.load(signal_flags_ptr + rank, cache_modifier=".ca") # Initial relaxed read
    # while current_arrivals < world_size:
    #     current_arrivals = tl.atomic_cas(signal_flags_ptr + rank, current_arrivals, current_arrivals, sem="acquire", scope="sys")
    current_arrivals = 0
    while current_arrivals < world_size:
        current_arrivals = tl.atomic_cas(signal_flags_ptr + rank, -1, -1, sem="acquire", scope="sys")