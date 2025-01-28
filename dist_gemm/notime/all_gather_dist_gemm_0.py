import torch
import torch.distributed as dist
import os

if __name__ == "__main__":
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    print(f"Starting distributed GEMM on Rank {rank} of {world_size} on device cuda:{rank}")

    torch.manual_seed(42)

    M, K, N = 128, 32, 512
    rows_per_rank = M // world_size

    assert M % world_size == 0, "M must be divisible by world size."

    A = torch.randn(M, K).cuda()
    B = torch.randn(K, N).cuda()

    local_A = A[rank * rows_per_rank:(rank + 1) * rows_per_rank].clone()
    local_C = torch.matmul(local_A, B)

    gathered_C = [torch.zeros_like(local_C) for _ in range(world_size)]
    dist.all_gather(gathered_C, local_C)

    global_C = torch.cat(gathered_C, dim=0)
    print(f"Rank {rank} local result:\n{local_C}")

    if rank == 0:
        expected_C = torch.matmul(A, B)
        if torch.allclose(global_C, expected_C, atol=1e-6):
            print("Validation passed! Distributed result matches the expected result.")
        else:
            print(f"Expected:\n{expected_C}")
            print(f"Global C:\n{global_C}")
            print("Validation failed! Results do not match.")

    dist.destroy_process_group()