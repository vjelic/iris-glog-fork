import torch
import triton
import random
import sys
import os
from utils import dump_timers

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import pyrocSHMEM as pyshmem


torch.manual_seed(123)
random.seed(123)

gpu = "mi300"
gpu = "mi250"

total_sm = 304
# if gpu == "mi300" else 104

from communication import all_scatter_kernel
from matmul_wrapper import matmul
from validation import validate_gemm

# ---------------------------------------------------------------------------
# Example and Benchmark
# ---------------------------------------------------------------------------

debug = False
validate = True
benchmark = True

COLLECT_TIMESTAMPS = True

m, n, k = 4864, 4096, 8256
SCATTER_TILE_M=128
SCATTER_TILE_N=128

heap_size = 1 << 30
shmem = pyshmem.pyrocSHMEM(heap_size)
rank = shmem.get_rank()
world_size = shmem.get_num_ranks()

# data_type = torch.float16
data_type = torch.float32

A = shmem.randn(m, k, device="cuda", dtype=data_type)
B = shmem.randn(n, k, device="cuda", dtype=data_type).T
C = shmem.zeros((m, n), device="cuda", dtype=A.dtype) # TODO: Do we need this?

# Split
M = m
N = n
K = k
n = n // world_size
assert N % world_size == 0, "N must be divisible by world size."

local_B = B[:, rank * n : (rank + 1) * n].clone()
local_C_partial = shmem.zeros((m, n), device="cuda", dtype=A.dtype)
local_C = shmem.zeros(M, N, dtype=C.dtype).cuda()

bias = None
BLK_M = 256
BLK_N = 256
BLK_K = 32
total_blocks_M = triton.cdiv(m, BLK_M)
total_blocks_N = triton.cdiv(n, BLK_N)
total_tiles = total_blocks_M * total_blocks_N
gsize_m = 8
two_tiles = "True"

num_stages = 1
num_warps = 8
waves_per_eu = 0
mfmaInstrSize = 16
kpack = 2

streamk_sms = 256
communication_sms = total_sm - streamk_sms
communication_block_size = 128
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

gemm_stream = torch.cuda.Stream()
comm_stream = torch.cuda.Stream()

def run_experiment():
    global local_C_partial
    torch.cuda.nvtx.range_push(f"GEMM + Communication")
    torch.cuda.nvtx.range_push(f"GEMM")
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
            None,
            False,
            COLLECT_TIMESTAMPS,
            mm_begin_timestamp,
            mm_end_timestamp,
        )

    # All scatter kernel
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push(f"Communication")
    with torch.cuda.stream(comm_stream):
        ss = all_scatter_kernel[grid](
            local_C_partial,
            local_C,
            tile_completed,
            m,
            n,
            local_C_partial.stride(0),
            local_C_partial.stride(1),
            C.stride(0),
            C.stride(1),
            BLK_M,
            BLK_N,
            gsize_m,
            total_tiles,
            BLOCK_SIZE=communication_block_size,
            NUM_SMS=communication_sms,
            heap_bases=shmem.get_heap_bases(),
            cur_rank=rank,
            world_size=world_size,
            SCATTER_TILE_M=SCATTER_TILE_M,
            SCATTER_TILE_N=SCATTER_TILE_N,
            COLLECT_TIMESTAMPS=COLLECT_TIMESTAMPS,            
            begin_timestamp_ptr=comm_begin_timestamp,
            middle_min_timestamp_ptr=comm_middle_min_timestamp,
            middle_max_timestamp_ptr=comm_middle_max_timestamp,
            end_timestamp_ptr=comm_end_timestamp,
        )

        shmem.log_debug(f"{ss.n_regs} registers used, {ss.n_spills} spills")

    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    shmem.barrier()
    torch.cuda.nvtx.range_pop()



run_experiment()

if COLLECT_TIMESTAMPS:
    num_timer_experiments = 10
    for experiment in range(num_timer_experiments):
        reset_timers()
        run_experiment()

if COLLECT_TIMESTAMPS and rank == 0:
    gpu_freq = shmem.wall_clock_rate(rank) * 1e-3 
    filename = f"gemm_tiles_all_scatter_trace_rank{rank}.json"
    dump_timers(mm_begin_timestamp,
                    mm_end_timestamp,
                    comm_begin_timestamp,
                    comm_middle_max_timestamp,
                    comm_middle_min_timestamp,
                    comm_end_timestamp,
                    gpu_freq,
                    filename)
            
if validate:
    matmul.set_debug(False)
    success = validate_gemm(A, B, local_C, shmem)
    shmem.barrier()
    shmem.log("Validation passed.") if success else shmem.log("Validation failed.")

if benchmark:
    perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)

    triton_ms = triton.testing.do_bench(lambda: run_experiment())
    shmem.log_stats(f"tile matmul (grid={total_tiles}): {triton_ms:.3f} ms  {perf(triton_ms):.3f} tflops")


exit(0)
