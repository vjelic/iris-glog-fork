import triton
import triton.language as tl

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import pyrocSHMEM as pyshmem
from utils import read_realtime

@triton.jit
def tile_id_to_index_range(
    tile_id,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m

    rm_start = pid_m * BLOCK_SIZE_M
    rn_start = pid_n * BLOCK_SIZE_N

    rm = rm_start + tl.arange(0, BLOCK_SIZE_M)
    rn = rn_start + tl.arange(0, BLOCK_SIZE_N)

    rm = tl.where(rm < M, rm, M)
    rn = tl.where(rn < N, rn, N)

    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    return rm, rn, rm_start, rn_start


@triton.jit
def offset_for_tile(
    local_tile_id,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    GROUP_SIZE_M,
    M_local,
    N_local
):
    rm, rn, rm_start, rn_start = tile_id_to_index_range(
        local_tile_id, M_local, N_local,
        BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )
    c_mask = (rm[:, None] < M_local) & (rn[None, :] < N_local)
    return rm, rn, c_mask, rm_start, rn_start

@triton.jit
def extract_submask_and_offset(
    rm, rn, mask, rm_start, rn_start,
    start_row, start_col,
    SUB_BLOCK_SIZE_M: tl.constexpr,
    SUB_BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    stride_cm_local: tl.constexpr,
    stride_cn_local: tl.constexpr
):
    # Create indices for the sub-block
    sub_rm = tl.arange(0, SUB_BLOCK_SIZE_M) + start_row
    sub_rn = tl.arange(0, SUB_BLOCK_SIZE_N) + start_col

    # Create a 2D grid of indices for the sub-block
    sub_rm_2d = sub_rm[:, None]  # Shape: (SUB_BLOCK_SIZE_M, 1)
    sub_rn_2d = sub_rn[None, :]  # Shape: (1, SUB_BLOCK_SIZE_N)

    # Compute the sub-mask
    sub_mask = (sub_rm_2d < BLOCK_SIZE_M) & (sub_rn_2d < BLOCK_SIZE_N)

    # Compute the sub-offset relative to the start of the tile
    sub_offset = ((rm_start + sub_rm_2d) * stride_cm_local) + ((rn_start + sub_rn_2d) * stride_cn_local)

    return sub_mask, sub_offset



@triton.jit
def all_reduce_kernel(
    local_C_partial_ptr,
    c,
    tile_completed_ptr,
    heap_bases,
    M_local,
    N_local,
    stride_cm_local,
    stride_cn_local,
    stride_cm_global,
    stride_cn_global,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    total_tiles: tl.constexpr,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    begin_timestamp_ptr: tl.tensor = None,
    middle_min_timestamp_ptr: tl.tensor = None,
    middle_max_timestamp_ptr: tl.tensor = None,
    end_timestamp_ptr: tl.tensor = None,
    REDUCTION_TILE_M: tl.constexpr = 128,
    REDUCTION_TILE_N: tl.constexpr = 128,
    COLLECT_TIMESTAMPS: tl.constexpr = False,
):
    pid = tl.program_id(axis=0)

    for tile in range(pid, total_tiles, NUM_SMS):
        result = 0

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_min(begin_timestamp_ptr + tile, timestamp)

        while result == 0:
            compare = 1
            value = 0
            result = pyshmem.atomic_cas(
                tile_completed_ptr + tile,
                compare,
                value,
                cur_rank,
                cur_rank,
                heap_bases,
                sem="acquire",
                scope="sys",
            )

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(middle_max_timestamp_ptr + tile, timestamp)
            tl.atomic_min(middle_min_timestamp_ptr + tile, timestamp)

        # Consume the tile in sub-tiles
        rm, rn, mask, rm_start, rn_start = offset_for_tile(
            tile, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M,
            M_local, N_local
        )

        # Calculate the number of sub-tiles in each dimension
        num_sub_tiles_m = tl.cdiv(BLOCK_SIZE_M, REDUCTION_TILE_M)
        num_sub_tiles_n = tl.cdiv(BLOCK_SIZE_N, REDUCTION_TILE_N)
        total_sub_tiles = num_sub_tiles_m * num_sub_tiles_n

        # Flattened loop over all sub-tiles, triton is
        # better at handling flat loops instead of nested loops
        for sub_tile_idx in range(0, total_sub_tiles):
            # Calculate start_row and start_col for the current sub-tile
            start_row = (sub_tile_idx // num_sub_tiles_n) * REDUCTION_TILE_M
            start_col = (sub_tile_idx % num_sub_tiles_n) * REDUCTION_TILE_N

            # Extract the sub-mask and sub-offset for the current sub-block
            sub_mask, sub_offset = extract_submask_and_offset(
                rm, rn, mask, rm_start, rn_start,
                start_row, start_col,
                REDUCTION_TILE_M, REDUCTION_TILE_N,
                BLOCK_SIZE_M, BLOCK_SIZE_N,
                stride_cm_local, stride_cn_local
            )

            # Load data from the local partial result
            data = tl.load(local_C_partial_ptr + sub_offset, mask=sub_mask)

            # Store data to the global result using atomic_add
            for remote_rank in range(world_size):
                pyshmem.atomic_add(
                    c + sub_offset,
                    data,
                    cur_rank,
                    remote_rank,
                    heap_bases,
                    mask=sub_mask,
                    scope="sys",
                )

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(end_timestamp_ptr + tile, timestamp)

@triton.jit
def all_scatter_kernel(
    local_C_partial_ptr,
    c,
    tile_completed_ptr,
    heap_bases,
    M_local,
    N_local,
    stride_cm_local,
    stride_cn_local,
    stride_cm_global,
    stride_cn_global,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M_global: tl.constexpr,
    total_tiles: tl.constexpr,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    begin_timestamp_ptr: tl.tensor = None,
    middle_min_timestamp_ptr: tl.tensor = None,
    middle_max_timestamp_ptr: tl.tensor = None,
    end_timestamp_ptr: tl.tensor = None,
    COMMUNICATION_TILE_M: tl.constexpr = 128,
    COMMUNICATION_TILE_N: tl.constexpr = 128,
    COLLECT_TIMESTAMPS: tl.constexpr = False,
):
    pid = tl.program_id(axis=0)

    for tile in range(pid, total_tiles, NUM_SMS):
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_min(begin_timestamp_ptr + tile, timestamp)

        result = 0
        # Spin till tile is produced
        while result == 0:
            compare = 1
            value = 0
            result = pyshmem.atomic_cas(
                tile_completed_ptr + tile,
                compare,
                value,
                cur_rank,
                cur_rank,
                heap_bases,
                sem="acquire",
                scope="sys",
            )

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(middle_max_timestamp_ptr + tile, timestamp)
            tl.atomic_min(middle_min_timestamp_ptr + tile, timestamp)

        # Consume the tile
        rm, rn, mask, rm_start, rn_start = offset_for_tile(
            tile, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M_global,
            M_local, N_local
        )
        
        
        # Calculate the number of sub-tiles in each dimension
        num_sub_tiles_m = tl.cdiv(BLOCK_SIZE_M, COMMUNICATION_TILE_M)
        num_sub_tiles_n = tl.cdiv(BLOCK_SIZE_N, COMMUNICATION_TILE_N)
        total_sub_tiles = num_sub_tiles_m * num_sub_tiles_n

        # Flattened loop over all sub-tiles, triton is
        # better at handling flat loops instead of nested loops
        for sub_tile_idx in range(0, total_sub_tiles):
            # Calculate start_row and start_col for the current sub-tile
            start_row = (sub_tile_idx // num_sub_tiles_n) * COMMUNICATION_TILE_M
            start_col = (sub_tile_idx % num_sub_tiles_n) * COMMUNICATION_TILE_N

            # Extract the sub-mask and sub-offset for the current sub-block
            sub_mask, sub_offset = extract_submask_and_offset(
                rm, rn, mask, rm_start, rn_start,
                start_row, start_col,
                COMMUNICATION_TILE_M, COMMUNICATION_TILE_N,
                BLOCK_SIZE_M, BLOCK_SIZE_N,
                stride_cm_local, stride_cn_local
            )
            
            # Load data from the local partial result
            data = tl.load(local_C_partial_ptr + sub_offset, mask=sub_mask)
            
            # Translate to global
            sub_offset_global = sub_offset + cur_rank * N_local
            
            # Store data to the global result using relaxed atomics
            for remote_rank in range(world_size):
                pyshmem.put(
                    c + sub_offset_global,
                    data,
                    cur_rank,
                    remote_rank,
                    heap_bases,
                    mask=sub_mask
                )

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(end_timestamp_ptr + tile, timestamp)

@triton.jit
def one_shot_kernel(
    partial_c,
    c,
    tile_completed_ptr,
    heap_bases,
    M_local,
    N_local,
    stride_cm_local,
    stride_cn_local,
    stride_cm_global,
    stride_cn_global,    
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    total_tiles: tl.constexpr,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    begin_timestamp_ptr: tl.tensor = None,
    middle_min_timestamp_ptr: tl.tensor = None,
    middle_max_timestamp_ptr: tl.tensor = None,
    end_timestamp_ptr: tl.tensor = None,
    REDUCTION_TILE_M: tl.constexpr = 128,
    REDUCTION_TILE_N: tl.constexpr = 128,
    COLLECT_TIMESTAMPS: tl.constexpr = False,
):
    pid = tl.program_id(axis=0)

    for tile in range(pid, total_tiles, NUM_SMS):
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_min(begin_timestamp_ptr + tile, timestamp)

    for tile in range(pid, total_tiles, NUM_SMS):
        result = 0
        while result == 0:
            compare = 1
            value = 0
            result = pyshmem.atomic_cas(
                tile_completed_ptr + tile,
                compare,
                value,
                cur_rank,
                cur_rank,
                heap_bases,
                sem="acquire",
                scope="sys",
            )

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(middle_max_timestamp_ptr + tile, timestamp)
            tl.atomic_min(middle_min_timestamp_ptr + tile, timestamp)

        # Consume the tile in sub-tiles
        rm, rn, mask, rm_start, rn_start = offset_for_tile(
            tile, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M,
            M_local, N_local
        )

        # Calculate the number of sub-tiles in each dimension
        num_sub_tiles_m = tl.cdiv(BLOCK_SIZE_M, REDUCTION_TILE_M)
        num_sub_tiles_n = tl.cdiv(BLOCK_SIZE_N, REDUCTION_TILE_N)
        total_sub_tiles = num_sub_tiles_m * num_sub_tiles_n

        # Flattened loop over all sub-tiles, triton is 
        # better at handling flat loops instead of nested loops
        for sub_tile_idx in range(0, total_sub_tiles):
            acc = tl.zeros((REDUCTION_TILE_M, REDUCTION_TILE_N), dtype=tl.float32)

            # Calculate start_row and start_col for the current sub-tile
            start_row = (sub_tile_idx // num_sub_tiles_n) * REDUCTION_TILE_M
            start_col = (sub_tile_idx % num_sub_tiles_n) * REDUCTION_TILE_N

            # Extract the sub-mask and sub-offset for the current sub-block
            sub_mask, sub_offset = extract_submask_and_offset(
                rm, rn, mask, rm_start, rn_start,
                start_row, start_col,
                REDUCTION_TILE_M, REDUCTION_TILE_N,
                BLOCK_SIZE_M, BLOCK_SIZE_N,
                stride_cm_local, stride_cn_local
            )
            
            # For all the ranks, load and accumulate
            for remote_rank in range(world_size):
                # Load data from the remote partial result
                acc += pyshmem.get(partial_c + sub_offset, cur_rank, remote_rank, heap_bases, mask=sub_mask)

            # Sub-tile completed, store the output as a sub-tile to c
            tl.store(c + sub_offset, acc, mask=sub_mask)

            if COLLECT_TIMESTAMPS:
                timestamp = read_realtime()
                tl.atomic_max(end_timestamp_ptr + tile, timestamp)