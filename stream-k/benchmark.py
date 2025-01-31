import torch
import triton
import random
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import pyrocSHMEM as pyshmem

from communication import all_scatter_kernel
from communication import all_reduce_kernel
from matmul_wrapper import matmul
from validation import validate_gemm

torch.manual_seed(123)
random.seed(123)

gpu = "mi300"
gpu = "mi250"

total_sm = 304 if gpu == "mi300" else 104


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse matrix dimensions and configuration."
    )
    parser.add_argument("-m", type=int, default=4864, help="Number of rows in matrix A")
    parser.add_argument(
        "-n", type=int, default=4096, help="Number of columns in matrix B"
    )
    parser.add_argument(
        "-k", type=int, default=8256, help="Common dimension between matrices A and B"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--validate", action="store_true", help="Enable validation mode"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Enable benchmarking mode"
    )
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp32",
        choices=["fp16", "fp32", "int8", "bf16", "tf32"],
        help="Datatype of computation",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="all_reduce",
        choices=["all_reduce", "all_scatter"],
        help="Datatype of computation",
    )
    parser.add_argument("--BLK_M", type=int, default=256, help="Block size M")
    parser.add_argument("--BLK_N", type=int, default=256, help="Block size N")
    parser.add_argument("--BLK_K", type=int, default=32, help="Block size K")
    parser.add_argument("--gsize_m", type=int, default=8, help="Grid size M")
    parser.add_argument("--two_tiles", type=str, default="True", help="Use two tiles")
    parser.add_argument("--num_stages", type=int, default=1, help="Number of stages")
    parser.add_argument("--num_warps", type=int, default=8, help="Number of warps")
    parser.add_argument(
        "--waves_per_eu", type=int, default=0, help="Waves per execution unit"
    )
    parser.add_argument(
        "--mfmaInstrSize", type=int, default=16, help="MFMA instruction size"
    )
    parser.add_argument("--kpack", type=int, default=2, help="K packing size")
    parser.add_argument(
        "--heap_size", type=int, default=1 << 30, help="pyrocSHMEM heap size"
    )
    default_streamk_sms = 256 if gpu == "mi300" else 64
    parser.add_argument(
        "--streamk_sms", type=int, default=default_streamk_sms, help="pyrocSHMEM heap size"
    )
    parser.add_argument(
        "--communication_block_size", type=int, default=256, help="pyrocSHMEM heap size"
    )

    return vars(parser.parse_args())


def main():
    args = parse_args()

    shmem = pyshmem.pyrocSHMEM(args["heap_size"])
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # GEMM
    datatype = torch.float32
    if args["datatype"] == "fp16":
        datatype = torch.float16
    elif args["datatype"] == "fp32":
        datatype = torch.float32
    elif args["datatype"] == "int8":
        datatype = torch.int8
    elif args["datatype"] == "bf16":
        datatype = torch.bfloat16
    else:
        print("Unknown datatype.")
        exit(1)

    assert args["n"] % world_size == 0, f"N ({args['n']}) must be divisible by world size ({world_size})."
    assert args["k"] % world_size == 0, f"K ({args['k']}) must be divisible by world size ({world_size})."
    
    A = shmem.randn(args["m"], args["k"], device="cuda", dtype=datatype)
    B = shmem.randn(args["n"], args["k"], device="cuda", dtype=datatype).T
    C = shmem.zeros((args["m"], args["n"]), device="cuda", dtype=A.dtype)
    
    args["M"] = args["m"]
    args["N"] = args["n"]
    args["K"] = args["k"]
    
    # Splitting
    if args["algorithm"] == "all_scatter":
        args["n"] = args["n"]  // world_size
        local_B = B[:, rank * args["n"] : (rank + 1) * args["n"]].clone()
        local_A = A
    elif args["algorithm"] == "all_reduce":
        rows_per_gpu = args["k"] // world_size
        start_row = rank * rows_per_gpu
        end_row = start_row + rows_per_gpu
        local_B = B[start_row:end_row, :]
        local_A = A[:, start_row:end_row]
    else:
        print("Unknown algorithm.")
        exit(1)
    global_C = shmem.zeros((args["M"], args["N"]), device="cuda", dtype=A.dtype)
    local_C = shmem.zeros((args["m"], args["n"]), device="cuda", dtype=A.dtype)

    total_blocks_M = triton.cdiv(args["m"], args["BLK_M"])
    total_blocks_N = triton.cdiv(args["n"], args["BLK_N"])
    total_tiles = total_blocks_M * total_blocks_N
    
    if args["streamk_sms"] >= total_sm:
        print(f"Invalid number of stream-K SMs. {args['streamk_sms']} >= {total_sm}")
        exit(1)

        
    communication_sms = total_sm - args["streamk_sms"] 
    
    communication_num_threads = args["communication_block_size"]  * communication_sms
    grid = lambda meta: (triton.cdiv(communication_num_threads, meta["BLOCK_SIZE"]),)
        
    locks = shmem.zeros((args["streamk_sms"] ,), device="cuda", dtype=torch.int32)
    tile_completed = shmem.zeros((total_tiles,), device="cuda", dtype=torch.int32)
    P = shmem.zeros((args["streamk_sms"] , args["BLK_M"] * args["BLK_N"]), device="cuda", dtype=torch.float32)
    bias = None

    gemm_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()

    communication_kernel = all_scatter_kernel if args["algorithm"] == "all_scatter" else all_reduce_kernel

    
    def run_experiment():
        nonlocal local_C
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
                args["streamk_sms"] ,
                args["BLK_M"],
                args["BLK_N"],
                args["BLK_K"],
                args["gsize_m"],
                args["two_tiles"] ,
                args["num_stages"] ,
                args["num_warps"] ,
                args["waves_per_eu"] ,
                args["mfmaInstrSize"] ,
                args["kpack"] ,
            )

        # All scatter kernel
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(f"Communication")
        with torch.cuda.stream(comm_stream):
            ss = communication_kernel[grid](
                local_C,
                global_C,
                tile_completed,
                shmem.get_heap_bases(),
                args["m"],
                args["n"],
                local_C.stride(0),
                local_C.stride(1),
                C.stride(0),
                C.stride(1),
                args["BLK_M"],
                args["BLK_N"],
                args["gsize_m"],
                total_tiles,
                rank,
                world_size,
                BLOCK_SIZE=args["communication_block_size"] ,
                NUM_SMS=communication_sms,
            )

            shmem.log_debug(f"{ss.n_regs} registers used, {ss.n_spills} spills")

        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        shmem.barrier()
        torch.cuda.nvtx.range_pop()

    run_experiment()

    if args['validate']:
        matmul.set_debug(False)
        validate_gemm(A, B, global_C, shmem)
        shmem.barrier()
        shmem.log("Validation passed.")

    if args['benchmark']:
        perf = lambda ms: 2 * args["M"] * args["N"] * args["K"] * 1e-12 / (ms * 1e-3)
        triton_ms = triton.testing.do_bench(lambda: run_experiment())
        shmem.log_stats(f"tile matmul (grid={total_tiles}): {triton_ms:.3f} ms  {perf(triton_ms):.3f} tflops")



if __name__ == "__main__":
    main()