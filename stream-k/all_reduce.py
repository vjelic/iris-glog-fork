import torch
import triton
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import pyrocSHMEM as pyshmem


torch.manual_seed(123)
random.seed(123)

gpu = "mi300"
gpu = "mi250"

total_sm = 304 if gpu == "mi300" else 104

from communication import all_reduce_kernel
from matmul_wrapper import matmul

# ---------------------------------------------------------------------------
# Example and Benchmark
# ---------------------------------------------------------------------------

debug = False
perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
m, n, k = 4864, 4096, 8256

heap_size = 1 << 30
shmem = pyshmem.pyrocSHMEM(heap_size)
rank = shmem.get_rank()
world_size = shmem.get_num_ranks()

shmem.log(f"total SMs: {total_sm}")
shmem.log(f"Device: {shmem.get_device()}")

# data_type = torch.float16
data_type = torch.float32

A = shmem.randn(m, k, device="cuda", dtype=data_type)
B = shmem.randn(n, k, device="cuda", dtype=data_type).T
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

bias = None
BLK_M = 256
BLK_N = 256
BLK_K = 64
total_blocks_M = triton.cdiv(m, BLK_M)
total_blocks_N = triton.cdiv(n, BLK_N)
total_tiles = total_blocks_M * total_blocks_N
gsize_m = 8
two_tiles = "True"

num_stages=1
num_warps = 8
waves_per_eu = 0
mfmaInstrSize = 16
kpack = 2

communication_sms = 2
streamk_sms = total_sm - communication_sms


shmem.log(f"{streamk_sms=}")
shmem.log(f"{communication_sms=}")

matmul.set_debug(debug)
locks = shmem.zeros((streamk_sms,), device="cuda", dtype=torch.int32)
tile_completed = shmem.zeros((total_tiles,), device="cuda", dtype=torch.int32)
P = shmem.zeros((streamk_sms, BLK_M * BLK_N), device="cuda", dtype=torch.float32)

shmem.log(f"{total_tiles=}")

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
    grid = lambda meta: (triton.cdiv(communication_num_threads, meta["BLOCK_SIZE"]),)

    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push(f"Communication {exp}")
    with torch.cuda.stream(comm_stream):
        rr = all_reduce_kernel[grid](
            local_C,
            global_C,
            tile_completed,
            shmem.get_heap_bases(),
            m,
            n,
            local_C.stride(0),
            local_C.stride(1),
            BLK_M,
            BLK_N,
            gsize_m,
            total_tiles,
            rank,
            world_size,
            BLOCK_SIZE=communication_block_size,
        )

        if debug:
            shmem.log(f"{rr.n_regs} registers used, {rr.n_spills} spills")
            # print(rr.asm['ttgir'])
            # print(rr.asm['amdgcn'])

    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    shmem.barrier()
    shmem.log("Reduction completed.")
    torch.cuda.nvtx.range_pop()

# Validation
matmul.set_debug(False)
expected = A @ B
diff_mask = ~torch.isclose(global_C, expected, atol=1)
breaking_indices = torch.nonzero(diff_mask, as_tuple=False)

if True or rank ==0:
    shmem.log(f"{local_C=}")
    shmem.log(f"{global_C=}")
    shmem.log(f"{expected=}")

# Identify where the difference exceeds the tolerance
print(expected.shape)
print(global_C.shape)

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
shmem.log("Validation passed.")

exit(0)
