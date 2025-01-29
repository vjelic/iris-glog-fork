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
def all_reduce_kernel(
    local_C_partial_ptr,
    local_C_ptr,
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
):
    pid = tl.program_id(axis=0)

    # TODO: Parallel over PIDs.
    if pid != 0:
        return

    # TODO: Match producer loop
    for tile in range(total_tiles):
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

        # Consume the tile
        rm, rn, mask = offset_for_tile(
            tile, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M,
            M_local, N_local
        )
        offset = rm[:, None] * stride_cm_local + rn[None, :] * stride_cn_local

        # TODO Tile communication differently than GEMM
        data = tl.load(local_C_partial_ptr + offset, mask=mask)

        # Store
        for remote_rank in range(world_size):
            pyshmem.atomic_add(
                local_C_ptr + offset,
                data,
                cur_rank,
                remote_rank,
                heap_bases,
                mask=mask,
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
):
    pid = tl.program_id(axis=0)
    if pid != 0:
        return

    for tile in range(total_tiles):
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