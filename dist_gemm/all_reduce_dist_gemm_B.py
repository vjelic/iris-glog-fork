import torch
import torch.distributed as dist
import time
import random

if __name__ == "__main__":
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    print(f"Starting distributed GEMM on Rank {rank} of {world_size} on device cuda:{rank}")

    torch.manual_seed(42)
    random.seed(42)

    # Matrix dimensions
    M, K, N = 8192, 8192, 8192
    cols_per_rank = N // world_size
    assert N % world_size == 0, "N must be divisible by world size."

    dtype = torch.float16
    A = torch.randn(M, K, dtype=dtype).cuda()
    B = torch.randn(K, N, dtype=dtype).cuda()

    print(f"Rank {rank}: A shape = {A.shape}, B shape = {B.shape}")

    # Partition the weights matrix B across ranks (column-wise)
    local_B = B[:, rank * cols_per_rank:(rank + 1) * cols_per_rank].clone()
    print(f"Rank {rank}: local_B shape = {local_B.shape}")

    # Allocate memory for the full matrix C on all ranks
    local_C = torch.zeros(M, N, dtype=dtype).cuda()

    # Synchronize before starting timing
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Perform local GEMM (A @ local_B)
    local_C_partial = torch.matmul(A, local_B)

    # Place the partial result into the correct section of C
    local_C[:, rank * cols_per_rank:(rank + 1) * cols_per_rank] = local_C_partial

    # Perform an all-reduce to sum contributions across ranks
    dist.all_reduce(local_C, op=dist.ReduceOp.SUM)

    # Synchronize after computation and communication
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    elapsed_time_ms = (end_time - start_time) * 1e3  # Convert to milliseconds

    # Validation and performance metrics
    if rank == 0:
        perf_tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        tflops = perf_tflops(elapsed_time_ms)
        expected_C = torch.matmul(A, B)
        atol = 1
        rtol = 1e-6

        if torch.allclose(local_C, expected_C, atol=atol, rtol=rtol):
            print("Validation passed! Distributed result matches the expected result.")
        else:
            max_abs_error = torch.max(torch.abs(local_C - expected_C))
            max_rel_error = torch.max(torch.abs((local_C - expected_C) / expected_C))
            print(f"Validation failed!")
            print(f"Max Absolute Error: {max_abs_error:.6f}")
            print(f"Max Relative Error: {max_rel_error:.6e}")

        print("\nPerformance Metrics:")
        print(f"Elapsed Time (GEMM + Communication): {elapsed_time_ms:.3f} ms")
        print(f"TFLOPs Achieved: {tflops:.2f}")

    dist.destroy_process_group()