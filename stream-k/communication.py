import triton
import triton.language as tl

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import pyrocSHMEM as pyshmem


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

    return rm, rn


@triton.jit
def offset_for_tile(
    local_tile_id,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    GROUP_SIZE_M,
    M_local,
    N_local
):
    rm, rn =  tile_id_to_index_range(
            local_tile_id, M_local, N_local,
            BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
        )
    c_mask = (rm[:, None] < M_local) & (rn[None, :] < N_local)
    return rm, rn, c_mask

@triton.jit
def extract_submask_and_offset(
    rm, rn, mask, offset,
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

    # Compute the sub-offset manually using strides
    sub_offset = (sub_rm_2d * stride_cm_local) + (sub_rn_2d * stride_cn_local)

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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    total_tiles: tl.constexpr,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    REDUCTION_TILE_M: tl.constexpr = 128,
    REDUCTION_TILE_N: tl.constexpr = 128,
):
    pid = tl.program_id(axis=0)

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

        # Consume the tile in sub-tiles
        rm, rn, mask = offset_for_tile(
            tile, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M,
            M_local, N_local
        )

        offset = rm[:, None] * stride_cm_local + rn[None, :] * stride_cn_local

        # Iterate over the tile in reduction blocks
        for start_row in range(0, BLOCK_SIZE_M, REDUCTION_TILE_M):
            for start_col in range(0, BLOCK_SIZE_N, REDUCTION_TILE_N):
                # Extract the sub-mask and sub-offset for the current sub-block
                sub_mask, sub_offset = extract_submask_and_offset(
                    rm, rn, mask, offset,
                    start_row, start_col,
                    REDUCTION_TILE_M, REDUCTION_TILE_N,
                    BLOCK_SIZE_M, BLOCK_SIZE_N,
                    stride_cm_local, stride_cn_local
                )

                # data = tl.load(local_C_partial_ptr + sub_offset, mask=sub_mask)
                data = tl.load(tl.multiple_of(local_C_partial_ptr + sub_offset, (16, 16)), mask=sub_mask)

                # Store
                for remote_rank in range(world_size):
                    pyshmem.atomic_add(
                        c + sub_offset,
                        data,
                        cur_rank,
                        remote_rank,
                        heap_bases,
                        mask=sub_mask,
                        sem="relaxed"
                    )

@triton.jit
def all_scatter_kernel(
    local_C_partial_ptr,
    local_C_ptr,
    tile_completed_ptr,
    heap_bases,
    M_local,
    N_local,
    stride_cm_local,
    stride_cn_local,
    stride_cm_global,
    stride_cn_global,
    BLOCK_SIZE_M_local: tl.constexpr,
    BLOCK_SIZE_N_local: tl.constexpr,
    GROUP_SIZE_M_global: tl.constexpr,
    total_tiles: tl.constexpr,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    for tile in range(pid, total_tiles, NUM_SMS):
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
        # Consume the tile
        rm, rn, load_mask = offset_for_tile(
            tile, BLOCK_SIZE_M_local, BLOCK_SIZE_N_local, GROUP_SIZE_M_global,
            M_local, N_local
        )
        offset_local = rm[:, None] * stride_cm_local + rn[None, :] * stride_cn_local
        data = tl.load(local_C_partial_ptr + offset_local, mask=load_mask)

        rm_global = rm
        rn_global = rn + cur_rank * N_local
        store_mask = load_mask
        offset_global = rm_global[:, None] * stride_cm_global + rn_global[None, :] * stride_cn_global

        # Store
        for remote_rank in range(world_size):
            if True:
                 pyshmem.put(
                    local_C_ptr + offset_global,
                    data,
                    cur_rank,
                    remote_rank,
                    heap_bases,
                    mask=store_mask,
                )