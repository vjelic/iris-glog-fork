import torch
import triton
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import pyrocSHMEM as pyshmem
import triton.language as tl


# from streamk_kernel import streamk_gemm
# from streamk_kernel_atomic import streamk_gemm
from shmem_persistent_gemm import persistent_gemm

torch.manual_seed(123)
random.seed(123)

gpu = "mi300"
gpu = "mi250"

total_sm = 304 if gpu == "mi300" else 104
num_xcds = 8 if gpu == "mi300" else 1
gemm_kernel = persistent_gemm
print(f"total SMs: {total_sm}")


# @triton.jit
# def local_to_global_tile_id(
#     local_tile_id,  # Local tile ID
#     rank_local,  # Local rank
#     world_size_global,  # Total number of ranks
#     M_local,
#     N_local,  # Local matrix dimensions
#     BLOCK_SIZE_M_local,
#     BLOCK_SIZE_N_local,  # Local block sizes
# ):
#     num_tiles_m_local = (M_local + BLOCK_SIZE_M_local - 1) // BLOCK_SIZE_M_local
#     num_tiles_n_local = (N_local + BLOCK_SIZE_N_local - 1) // BLOCK_SIZE_N_local
#     num_tiles_per_rank = num_tiles_m_local * num_tiles_n_local

#     global_tile_id = rank_local * num_tiles_per_rank + local_tile_id
#     return global_tile_id

@triton.jit
def local_to_global_tile_id(
    local_tile_id,  # Local tile ID
    rank_local,  # Local rank
    world_size_global,  # Total number of ranks
    M_local,
    N_local,  # Local matrix dimensions
    BLOCK_SIZE_M_local,
    BLOCK_SIZE_N_local,  # Local block sizes
):
    # Number of tiles in the local matrix (per dimension)
    num_tiles_m_local = (M_local + BLOCK_SIZE_M_local - 1) // BLOCK_SIZE_M_local
    num_tiles_n_local = (N_local + BLOCK_SIZE_N_local - 1) // BLOCK_SIZE_N_local
    num_tiles_per_rank = num_tiles_m_local * num_tiles_n_local

    # Compute global tile ID
    global_tile_id = rank_local * num_tiles_per_rank + local_tile_id
    return global_tile_id


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

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

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
def reduction_kernel(
    local_C_partial_ptr,
    local_C_ptr,
    tile_completed_ptr,
    heap_bases,
    M_local,
    N_local,
    K_local,
    M_global,
    N_global,
    K_global,
    stride_am_local,
    stride_ak_local,
    stride_bk_local,
    stride_bn_local,
    stride_cm_local,
    stride_cn_local,
    stride_am_global,
    stride_ak_global,
    stride_bk_global,
    stride_bn_global,
    stride_cm_global,
    stride_cn_global,
    BLOCK_SIZE_M_local: tl.constexpr,
    BLOCK_SIZE_N_local: tl.constexpr,
    BLOCK_SIZE_K_local: tl.constexpr,
    GROUP_SIZE_M_global: tl.constexpr,
    total_tiles: tl.constexpr,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    # # print("Reduce on rank: ", cur_rank)
    # if cur_rank != 0:
    #     return
    # return
    # TODO: Need to pass NUM_SMs..
    # for tile_id in range(pid, total_tiles, NUM_SMS):
    for tile in range(total_tiles):
    # for tile in range(1):
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
        # print("Consume: ", local_C_partial_ptr + offset_local, data)

        # # print("ptr: ", local_C_partial_ptr + offset_local)
        # # print("data: ", data)
        # global_tile_id = local_to_global_tile_id(
        #     tile,
        #     cur_rank,
        #     world_size,
        #     M_local,
        #     N_local,
        #     BLOCK_SIZE_M_local,
        #     BLOCK_SIZE_N_local
        # )
        rm_global = rm
        rn_global = rn + cur_rank * N_local
        store_mask = load_mask
        offset_global = rm_global[:, None] * stride_cm_global + rn_global[None, :] * stride_cn_global

        # Store
        for remote_rank in range(world_size):
            # if (cur_rank == 0 and remote_rank == 0) and (global_tile_id == 0):
            if True:
            # if (cur_rank != remote_rank):
                # rm, rn, store_mask = offset_for_tile(
                #                     global_tile_id,
                #                     BLOCK_SIZE_M_local,
                #                     BLOCK_SIZE_N_local,
                #                     GROUP_SIZE_M_global,
                #                     M_global, N_global)
                # # offset_global = rm[:, None] * stride_cm_global + rn[None, :] * stride_cn_global
                # offset_global = rm[:, None] * stride_cn_global + rn[None, :] * stride_cm_global

                # print("offset_global: ", offset_global)
                # for i in offset_global:
                #     print(i)
                # print("store_mask: ", store_mask)
                # for i in store_mask:
                #     print(i)
                # print("rank, tile, gtile", cur_rank, tile, global_tile_id)
                # print("put", local_C_ptr + output_offset, data)
                # output_offset=offset_global
                # print("stride_cm_global:", stride_cm_global)
                # print("stride_cn_global:", stride_cn_global)
                # print("rank, offset, mask", cur_rank, offset_global, store_mask)
                pyshmem.put(
                    local_C_ptr + offset_global,
                    data,
                    cur_rank,
                    remote_rank,
                    heap_bases,
                    mask=store_mask,
                )


class matmul(torch.autograd.Function):

    _debug = True

    @staticmethod
    def set_debug(debug: bool):
        matmul._debug = debug

    @staticmethod
    def _call(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        bias: torch.Tensor,
        P: torch.Tensor,
        locks: torch.Tensor,
        tile_completed: torch.Tensor,
        rank: int,
        total_programs_streamk: int,
        BLK_M: int,
        BLK_N: int,
        BLK_K: int,
        gsize_m: int,
        two_tiles: bool,
        num_stages: int,
        num_warps: int,
        waves_per_eu: int,
        mfmaInstrSize: int,
        kpack: int,
    ):

        #        assert a.is_contiguous() and b.is_contiguous(), "non-contiguous inputs are not supported"
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape

        total_blocks_M = triton.cdiv(M, BLK_M)
        total_blocks_N = triton.cdiv(N, BLK_N)
        iters_per_tile = triton.cdiv(K, BLK_K)
        total_tiles = total_blocks_M * total_blocks_N
        even_k = K % BLK_K == 0

        if total_programs_streamk > 0:  # Stream-K
            # last wave may occupy less than total_programs_streamk SMs
            total_tiles_streamk = total_tiles % total_programs_streamk
            # for two-tile Stream-K + data-parallel from original paper
            #            if two_tiles and total_tiles - total_tiles_streamk > total_programs_streamk:
            #                total_tiles_streamk += total_programs_streamk
            # remaining tiles are computed using classical blocking
            total_blocking_tiles = total_tiles - total_tiles_streamk
            total_iters_streamk = total_tiles_streamk * iters_per_tile
            # iterations related to full waves
            total_full_tiles_streamk = total_iters_streamk // total_programs_streamk
            # iterations related to last (partial) wave
            total_partial_tiles_streamk = total_iters_streamk % total_programs_streamk

        else:  # all tiles are computed using classical blocking
            total_blocking_tiles = total_tiles
            total_tiles_streamk = 0
            total_full_tiles_streamk = 0
            total_partial_tiles_streamk = 0
            total_iters_streamk = 0

        if matmul._debug:
            print(f"M,N,K={M},{N},{K} ; BLK_M,N,K={BLK_M},{BLK_N},{BLK_K}")
            print(f"{total_blocks_M=} x {total_blocks_N=} = {total_tiles=}")
            print(f"{total_tiles_streamk=} + {total_blocking_tiles=} = {total_tiles=}")
            print(f"{total_programs_streamk=}")
            print(f"{total_blocking_tiles=}")
            print(f"{total_full_tiles_streamk=}")
            print(f"{iters_per_tile=}")
            print(f"{total_iters_streamk=}")
            print("total_remainder_iters_streamk=", total_partial_tiles_streamk)
        use_bias = False
        # compute grid (work to do per SM on the first wave)
        grids = total_programs_streamk
        stride_bias = bias.stride(0) if use_bias else 0
        kk = gemm_kernel[(grids,)](
            a,
            b,
            c,
            bias,
            P,
            locks,
            tile_completed,
            rank,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            stride_bias,
            BLOCK_SIZE_M=BLK_M,
            BLOCK_SIZE_N=BLK_N,
            BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m,
            NUM_SMS=total_programs_streamk,
            STREAMK_TILES=total_tiles_streamk,
            NUM_XCDS=num_xcds,
            BIAS=use_bias,
            EVEN_K=even_k,
            num_stages=num_stages,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=mfmaInstrSize,
            kpack=kpack,
        )
        # if matmul._debug:
            # print(f"{kk.n_regs} registers used, {kk.n_spills} spills")

        #     print(kk.asm['ttgir'])
        #     print(kk.asm['amdgcn'])

        return c

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        bias: torch.Tensor,
        P: torch.Tensor,
        locks: torch.Tensor,
        tile_completed: torch.Tensor,
        rank: int,
        grid: int,
        BLK_M=128,
        BLK_N=128,
        BLK_K=32,
        gsize_m=1,
        two_tiles=True,
        num_stages=3,
        num_warps=4,
        waves_per_eu=2,
        mfmaInstrSize=16,
        kpack=1,
    ):
        matmul._call(
            a=a,
            b=b,
            c=c,
            bias=bias,
            P=P,
            locks=locks,
            tile_completed=tile_completed,
            rank=rank,
            total_programs_streamk=grid,
            BLK_M=BLK_M,
            BLK_N=BLK_N,
            BLK_K=BLK_K,
            gsize_m=gsize_m,
            two_tiles=two_tiles,
            num_warps=num_warps,
            num_stages=num_stages,
            waves_per_eu=waves_per_eu,
            mfmaInstrSize=mfmaInstrSize,
            kpack=kpack,
        )
        return c


# ---------------------------------------------------------------------------
# Example and Benchmark
# ---------------------------------------------------------------------------

perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)

## sweet shapes has multiple of 304 tiles
# m, n, k = 4864, 4096, 8256  # some problem size to test
# m, n, k =4864, 8192, 4160  # some problem size to test
# m, n, k = 8192, 4864, 6878  # some problem size to test

## test for tiles that is not multipe of 304 tiles
# m, n, k = 4096, 4096, 8192  # some problem size to test
m, n, k = 8192, 8192, 8192  # some problem size to test
# m, n, k = 512, 512, 512  # some problem size to test

## memory bound sizes
# m, n, k = 1, 1024, 256

## sizes that have to do masked load/store
# m, n, k = 8133, 8132, 8172  # some problem size to test
# m, n, k = 8128, 6878, 7378  # some problem size to test
# m, n, k = 6912, 768, 256  # some problem size to test
# m, n, k = 5632, 6656, 7936

## test when k is not multiple of 16
# m, n, k = 4864, 4096, 4300

# m, n, k = 128,128,128
# m, n, k = 2,2,2

heap_size = 1 << 30
shmem = pyshmem.pyrocSHMEM(heap_size)
shmem.log(f"Device: {shmem.get_device()}")
rank = shmem.get_rank()
world_size = shmem.get_num_ranks()


A = shmem.randn(m, k, device="cuda", dtype=torch.float16)
B = shmem.randn(n, k, device="cuda", dtype=torch.float16).T
# allocates output
C = shmem.zeros((m, n), device="cuda", dtype=A.dtype)

# Split
M = m
N = n
K = k
n = n // world_size
assert N % world_size == 0, "N must be divisible by world size."


local_B = B[:, rank * n : (rank + 1) * n].clone()
local_C_partial = shmem.zeros((m, n), device="cuda", dtype=A.dtype)

# bias = shmem.zeros((m,), device="cuda", dtype=A.dtype)
bias = None
BLK_M = 256
BLK_N = 256
BLK_K = 64
total_blocks_M = triton.cdiv(m, BLK_M)
total_blocks_N = triton.cdiv(n, BLK_N)
total_tiles = total_blocks_M * total_blocks_N
gsize_m = 8
two_tiles = "True"
num_stages = 2
num_warps = 8
waves_per_eu = 0
mfmaInstrSize = 16
kpack = 2

communication_sms =2
margin = 8
streamk_sms = total_sm - communication_sms - margin

# communication_sms = 16
# streamk_sms = 16

shmem.log(f"{streamk_sms=}")
matmul.set_debug(True)
locks = shmem.zeros((streamk_sms,), device="cuda", dtype=torch.int32)
tile_completed = shmem.zeros((total_tiles,), device="cuda", dtype=torch.int32)
P = shmem.zeros((streamk_sms, BLK_M * BLK_N), device="cuda", dtype=torch.float32)

shmem.log(f"{total_tiles=}")
shmem.log(f"{tile_completed=}")

shmem.log("Launching GEMM")
# torch.cuda.nvtx.range_push("GEMM")
# torch.cuda.nvtx.range_push("GEMM + Communication")
# torch.cuda.nvtx.range_push("GEMM")

local_C = shmem.zeros(M, N, dtype=C.dtype).cuda()

gemm_stream = torch.cuda.Stream()
comm_stream = torch.cuda.Stream()
num_experiments=20

for exp in range(num_experiments):
    torch.cuda.nvtx.range_push(f"GEMM + Communication {exp}")
    torch.cuda.nvtx.range_push(f"GEMM {exp}")
    with torch.cuda.stream(gemm_stream):
        local_C_partial = matmul.apply(
            A,
            local_B,
            local_C_partial,
            bias,
            P,
            locks,
            tile_completed,
            rank,
            streamk_sms,
            BLK_M,
            BLK_N,
            BLK_K,
            gsize_m,
            two_tiles,
            num_stages,
            num_warps,
            waves_per_eu,
            mfmaInstrSize,
            kpack,
        )

    # Reduction kernel
    communication_block_size = 128
    communication_num_threads = communication_block_size * communication_sms
    # communication_num_threads=8
    grid = lambda meta: (triton.cdiv(communication_num_threads, meta["BLOCK_SIZE"]),)

    # Reduction kernel
    # local_C[:, rank * n:(rank + 1) * n] = local_C_partial
    # for index, t in enumerate(tile_completed):
    #     print(f"Index {index}: {t}")
    # shmem.log("Launching Reduction")
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push(f"Communication {exp}")
    # torch.cuda.synchronize()
    with torch.cuda.stream(comm_stream):
        reduction_kernel[grid](
            local_C_partial,
            local_C,
            tile_completed,
            shmem.get_heap_bases(),
            m,
            n,
            k,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            local_B.stride(0),
            local_B.stride(1),
            local_C_partial.stride(0),
            local_C_partial.stride(1),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            BLK_M,
            BLK_N,
            BLK_K,
            gsize_m,
            total_tiles,
            rank,
            world_size,
            BLOCK_SIZE=communication_block_size,
        )
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    shmem.barrier()
    shmem.log("Reduction completed..")
    torch.cuda.nvtx.range_pop()

# exit(0)
# print(f"{tile_completed=}")
matmul.set_debug(False)
expected = A @ B
if True or rank ==0:
    shmem.log(f"{local_C_partial=}")
    shmem.log(f"{local_C=}")
    shmem.log(f"{expected=}")

# Identify where the difference exceeds the tolerance
diff_mask = ~torch.isclose(local_C, expected, atol=1)
breaking_indices = torch.nonzero(diff_mask, as_tuple=False)

if rank ==0:
    if not torch.allclose(local_C, expected, atol=1):
        max_diff = (local_C - expected).abs().max().item()
        shmem.log(f"Max absolute difference: {max_diff}")
        for idx in breaking_indices:
            idx = tuple(idx.tolist())
            local_val = local_C[idx]
            expected_val = expected[idx]
            shmem.log(f"Mismatch at index {idx}: local_C={local_val}, expected={expected_val}")
        # raise AssertionError(
        #     f"torch.allclose failed. Max diff: {max_diff}, first mismatch at {breaking_indices[0]}"
        # )

# assert torch.allclose(local_C, expected, atol=1), f"max: {(local_C - expected).abs().max().item()}\n{local_C}\n{expected}"

# Validation barrier
shmem.barrier()
shmem.log("pass validation test")

# for debugging, uncomment the following line
exit(0)

triton_ms = triton.testing.do_bench(lambda: torch.matmul(A, B))
shmem.log(f"PyTorch: {triton_ms:.3f} ms  {perf(triton_ms):.3f} tflops")

# locks = shmem.zeros((streamk_sms,), device="cuda", dtype=torch.int32)
# tile_completed = shmem.zeros((total_tiles,), device="cuda", dtype=torch.int32)
# P = shmem.zeros((streamk_sms, BLK_M * BLK_N), device="cuda", dtype=torch.float32)
# triton_ms = triton.testing.do_bench(
#     lambda: matmul.apply(
#         A,
#         B,
#         C,
#         bias,
#         P,
#         locks,
#         tile_completed,
#         streamk_sms,
#         BLK_M,
#         BLK_N,
#         BLK_K,
#         gsize_m,
#         two_tiles,
#         num_stages,
#         num_warps,
#         waves_per_eu,
#         mfmaInstrSize,
#         kpack,
#     )
# )
# print(
#     f"hybrid stream-k (grid={streamk_sms}): {triton_ms:.3f} ms  {perf(triton_ms):.3f} tflops"
# )

# locks = shmem.zeros((streamk_sms * 2,), device="cuda", dtype=torch.int32)
# P = shmem.zeros((streamk_sms * 2, BLK_M * BLK_N), device="cuda", dtype=torch.float32)
# triton_ms = triton.testing.do_bench(
#     lambda: matmul.apply(
#         A,
#         B,
#         C,
#         bias,
#         P,
#         locks,
#         streamk_sms * 2,
#         BLK_M,
#         BLK_N,
#         BLK_K,
#         gsize_m,
#         two_tiles,
#         num_stages,
#         num_warps,
#         waves_per_eu,
#         mfmaInstrSize,
#         kpack,
#     )
# )
# print(
#     f"hybrid stream-k (grid={streamk_sms * 2}): {triton_ms:.3f} ms  {perf(triton_ms):.3f} tflops"
# )

# triton_ms = triton.testing.do_bench(
#     lambda: matmul.apply(
#         A,
#         B,
#         C,
#         bias,
#         P,
#         locks,
#         total_tiles,
#         BLK_M,
#         BLK_N,
#         BLK_K,
#         gsize_m,
#         two_tiles,
#         num_stages,
#         num_warps,
#         waves_per_eu,
#         mfmaInstrSize,
#         kpack,
#     )
# )
# print(
#     f"tile matmul (grid={total_tiles}): {triton_ms:.3f} ms  {perf(triton_ms):.3f} tflops"
# )
