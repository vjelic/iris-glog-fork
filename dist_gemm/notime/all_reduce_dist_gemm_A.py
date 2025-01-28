import torch
import torch.distributed as dist
import time

if __name__ == "__main__":
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    print(f"Starting distributed GEMM on Rank {rank} of {world_size} on device cuda:{rank}")

    torch.manual_seed(42)

    M, K, N = 8192, 8192, 8192
    rows_per_rank = M // world_size
    assert M % world_size == 0, "M must be divisible by world size."

    dtype=torch.float16
    A = torch.randn(M, K, dtype=dtype).cuda()
    B = torch.randn(K, N, dtype=dtype).cuda()

    print(f"Rank {rank}: A shape = {A.shape}, B shape = {B.shape}")

    local_A = A[rank * rows_per_rank:(rank + 1) * rows_per_rank].clone()
    local_C_partial = torch.matmul(local_A, B)

    print(f"Rank {rank}: local_A shape = {local_A.shape}, local_C_partial shape = {local_C_partial.shape}")

    local_C = torch.zeros_like(A, dtype=dtype).cuda()
    local_C[rank * rows_per_rank:(rank + 1) * rows_per_rank, :] = local_C_partial

    print(f"Rank {rank}: local_C shape = {local_C.shape}")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    dist.all_reduce(local_C, op=dist.ReduceOp.SUM)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    elapsed_time_ms = (end_time - start_time) * 1e3

    if rank == 0:
        perf_tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        tflops = perf_tflops(elapsed_time_ms)
        expected_C = torch.matmul(A, B)
        atol = 1e-8
        rtol = 1e-6
        max_abs_error = torch.max(torch.abs(local_C - expected_C))
        max_rel_error = torch.max(torch.abs((local_C - expected_C) / expected_C))

        if torch.allclose(local_C, expected_C, atol=atol, rtol=rtol):
            print("Validation passed! Distributed result matches the expected result.")
        else:
            print(f"Validation failed!")
            print(f"Max Absolute Error: {max_abs_error:.6f}")
            print(f"Max Relative Error: {max_rel_error:.6e}")

        print("\nPerformance Metrics:")
        print(f"Elapsed Time: {elapsed_time_ms:.3f} ms")
        print(f"TFLOPs Achieved: {tflops:.2f}")

    dist.destroy_process_group()