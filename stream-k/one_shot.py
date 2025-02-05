import torch
import triton
import random
import sys
import os
import json
import numpy as np
from utils import dump_timers

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import pyrocSHMEM as pyshmem

torch.manual_seed(123)
random.seed(123)

gpu = "mi300"
gpu = "mi250"

total_sm = 304
streamk_sms = 256
# if gpu == "mi300" else 104

# total_sm = 104
# streamk_sms = 80

from communication import one_shot_kernel
from matmul_wrapper import matmul
from validation import validate_gemm

# ---------------------------------------------------------------------------
# Example and Benchmark
# ---------------------------------------------------------------------------

debug = False
validate = True
benchmark = True

COLLECT_TIMESTAMPS = False

m, n, k = 4864, 4096, 8256
# m, n, k = 512, 512, 256 # one tile

heap_size = 1 << 30
shmem = pyshmem.pyrocSHMEM(heap_size)
rank = shmem.get_rank()
world_size = shmem.get_num_ranks()

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

bias = None
BLK_M = 256
BLK_N = 256
BLK_K = 32
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

communication_sms = total_sm - streamk_sms
communication_block_size = 256
communication_num_threads = communication_block_size * communication_sms
grid = lambda meta: (triton.cdiv(communication_num_threads, meta["BLOCK_SIZE"]),)


shmem.log(f"Device: {shmem.get_device()}")
shmem.log(f"total SMs: {total_sm}")
shmem.log(f"{streamk_sms=}")
shmem.log(f"{communication_sms=}")
shmem.log(f"{total_tiles=}")


matmul.set_debug(debug)
locks = shmem.zeros((streamk_sms,), device="cuda", dtype=torch.int32)
tile_completed = shmem.zeros((total_tiles,), device="cuda", dtype=torch.int32)
P = shmem.zeros((streamk_sms, BLK_M * BLK_N), device="cuda", dtype=torch.float32)

max_ts = torch.iinfo(torch.int64).max
min_ts = 0
mm_begin_timestamp = torch.empty(total_tiles, dtype=torch.int64, device='cuda')
mm_end_timestamp = torch.zeros(total_tiles, dtype=torch.int64, device='cuda')

comm_begin_timestamp = torch.empty(total_tiles, dtype=torch.int64, device='cuda')
comm_middle_min_timestamp = torch.zeros(total_tiles, dtype=torch.int64, device='cuda')
comm_middle_max_timestamp = torch.zeros(total_tiles, dtype=torch.int64, device='cuda')
comm_end_timestamp = torch.zeros(total_tiles, dtype=torch.int64, device='cuda')

def reset_timers():
    mm_begin_timestamp.fill_(max_ts)
    mm_end_timestamp.fill_(min_ts)
    
    comm_begin_timestamp.fill_(max_ts)
    comm_middle_min_timestamp.fill_(max_ts)
    comm_middle_max_timestamp.fill_(min_ts)
    comm_end_timestamp.fill_(min_ts)

def reset_buffers():
    C.fill_(0)
    tile_completed.fill_(0)
    
gemm_stream = torch.cuda.Stream()
comm_stream = torch.cuda.Stream()

def run_experiment():
    global local_C
    torch.cuda.nvtx.range_push(f"GEMM + Communication")
    torch.cuda.nvtx.range_push(f"GEMM")
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
            world_size,
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
            mm_begin_timestamp,
            mm_end_timestamp,
            COLLECT_TIMESTAMPS,
            shmem.get_heap_bases(),
            True
        )

    # Reduction kernel
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push(f"Communication")
    with torch.cuda.stream(comm_stream):
        rr = one_shot_kernel[grid](
            local_C,
            C,
            tile_completed,
            shmem.get_heap_bases(),
            m,
            n,
            local_C.stride(0),
            local_C.stride(1),
            C.stride(0),
            C.stride(1),
            BLK_M,
            BLK_N,
            gsize_m,
            total_tiles,
            rank,
            world_size,
            BLOCK_SIZE=communication_block_size,
            NUM_SMS=communication_sms,
            begin_timestamp_ptr=comm_begin_timestamp,
            middle_min_timestamp_ptr=comm_middle_min_timestamp,
            middle_max_timestamp_ptr=comm_middle_max_timestamp,
            end_timestamp_ptr=comm_end_timestamp,
            COLLECT_TIMESTAMPS=COLLECT_TIMESTAMPS
        )

        shmem.log_debug(f"{rr.n_regs} registers used, {rr.n_spills} spills")
        # if rank == 0:
            # print(rr.asm['amdgcn'])
            # print(rr.asm['ttgir'])

    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    shmem.barrier()
    torch.cuda.nvtx.range_pop()


run_experiment()


if COLLECT_TIMESTAMPS:
    num_timer_experiments = 10
    for experiment in range(num_timer_experiments):
        reset_timers()
        reset_buffers()
        run_experiment()

if validate:
    matmul.set_debug(False)
    validate_gemm(A, B, C, shmem)
    shmem.barrier()
    shmem.log("Validation passed.")

if COLLECT_TIMESTAMPS and rank == 0:
    gpu_freq = shmem.wall_clock_rate(rank) * 1e-3 
    filename = f"gemm_tiles_one_shot_trace_rank{rank}.json"
    dump_timers(mm_begin_timestamp,
                    mm_end_timestamp,
                    comm_begin_timestamp,
                    comm_middle_max_timestamp,
                    comm_middle_min_timestamp,
                    comm_end_timestamp,
                    gpu_freq,
                    filename)
if benchmark:
    perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
    triton_ms = triton.testing.do_bench(lambda: run_experiment())
    shmem.log_stats(f"tile matmul (grid={total_tiles}): {triton_ms:.3f} ms  {perf(triton_ms):.3f} tflops")

exit(0)
