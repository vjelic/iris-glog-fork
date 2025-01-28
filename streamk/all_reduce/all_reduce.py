import torch
import triton
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pyrocSHMEM as pyshmem
import triton.language as tl


# from streamk_kernel import streamk_gemm
# from streamk_kernel_atomic import streamk_gemm
from gemm import persistent_gemm

torch.manual_seed(123)
random.seed(123)

gpu = "mi300"
gpu = "mi250"

total_sm = 304 if gpu == "mi300" else 104
num_xcds = 8 if gpu == "mi300" else 1
gemm_kernel = persistent_gemm
print(f"total SMs: {total_sm}")


# @triton.jit
# def tile_id_to_index_range(
#     tile_id,
#     M,
#     N,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
# ):
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n

#     group_id = tile_id // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

#     pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
#     pid_n = (tile_id % num_pid_in_group) // group_size_m

#     rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#     rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

#     rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
#     rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # return rm, rn


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

    rm = tl.where(rm < M, rm, M)  # Clip indices to [0, M)
    rn = tl.where(rn < N, rn, N)  # Clip indices to [0, N)

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
            tile, BLOCK_SIZE_M_local, BLOCK_SIZE_N_local, GROUP_SIZE_M_global,
            M_local, N_local
        )
        offset = rm[:, None] * stride_cm_local + rn[None, :] * stride_cn_local
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
# m, n, k = 8192, 8192, 8192  # some problem size to test
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

m, n, k = 128,128,128
# m, n, k = 2,2,2

heap_size = 1 << 30
shmem = pyshmem.pyrocSHMEM(heap_size)
shmem.log(f"Device: {shmem.get_device()}")
rank = shmem.get_rank()
world_size = shmem.get_num_ranks()

# data_type = torch.float16
data_type = torch.float32

A = shmem.randn(m, k, device="cuda", dtype=data_type)
B = shmem.randn(n, k, device="cuda", dtype=data_type).T
# allocates output
C = shmem.zeros((m, n), device="cuda", dtype=A.dtype)

# Split
rows_per_gpu = k // world_size
start_row = rank * rows_per_gpu
end_row = start_row + rows_per_gpu
assert k % world_size == 0, "N must be divisible by world size."


local_B = B[start_row:end_row, :]
local_A = A[:, start_row:end_row]
local_C = shmem.zeros((m, n), device="cuda", dtype=A.dtype)
global_C = shmem.zeros((m, n), device="cuda", dtype=A.dtype)

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

#triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 65536. Reducing block sizes or `num_stages` may help.
# num_stages = 2 #
num_stages=1
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

gemm_stream = torch.cuda.Stream()
comm_stream = torch.cuda.Stream()
num_experiments=1

for exp in range(num_experiments):
    torch.cuda.nvtx.range_push(f"GEMM + Communication {exp}")
    torch.cuda.nvtx.range_push(f"GEMM {exp}")
    with torch.cuda.stream(gemm_stream):
        local_C = matmul.apply(
            local_A,
            local_B,
            local_C,
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
            local_C,
            global_C,
            tile_completed,
            shmem.get_heap_bases(),
            m,
            n,
            k,
            A.stride(0),
            A.stride(1),
            local_B.stride(0),
            local_B.stride(1),
            local_C.stride(0),
            local_C.stride(1),
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
    shmem.log(f"{local_C=}")
    shmem.log(f"{global_C=}")
    shmem.log(f"{expected=}")

# Identify where the difference exceeds the tolerance
print(expected.shape)
print(global_C.shape)
diff_mask = ~torch.isclose(global_C, expected, atol=1)
breaking_indices = torch.nonzero(diff_mask, as_tuple=False)

if rank ==0:
    if not torch.allclose(global_C, expected, atol=1):
        max_diff = (global_C - expected).abs().max().item()
        shmem.log(f"Max absolute difference: {max_diff}")
        for idx in breaking_indices:
            idx = tuple(idx.tolist())
            local_val = global_C[idx]
            expected_val = expected[idx]
            shmem.log(f"Mismatch at index {idx}: global_C={local_val}, expected={expected_val}, ratio={local_val/expected_val}")
            break

assert torch.allclose(global_C, expected, atol=1), f"max: {(global_C - expected).abs().max().item()}\n{local_C}\n{expected}"

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
