import torch
import torch.distributed as dist

def main():
    torch.manual_seed(42)

    M, K, N = 1024, 64, 512

    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    print(f"Starting distributed GEMM on Rank {rank} of {world_size} on device cuda:{rank}")

    A_full = torch.randn(M, K, device=f"cuda:{rank}")
    B_full = torch.randn(K, N, device=f"cuda:{rank}")

    # Split B column-wise
    cols_per_gpu = N // world_size
    start_col = rank * cols_per_gpu
    end_col = start_col + cols_per_gpu
    B_local = B_full[:, start_col:end_col]

    # Perform local computation
    C_partial = A_full @ B_local

    # Allocate tensor for gathered results
    C_global = torch.empty(M, N, device=f"cuda:{rank}")

    # Manually specify slices to gather into
    gathered_parts = [C_global[:, i * cols_per_gpu: (i + 1) * cols_per_gpu] for i in range(world_size)]

    dist.all_gather(gathered_parts, C_partial)

    # Validation
    C_full = A_full @ B_full
    valid = torch.allclose(C_global, C_full, atol=1e-5)
    if valid:
        print(f"Rank {rank}: Validation passed! Distributed GEMM matches full GEMM.")
    else:
        print(f"Rank {rank}: Validation failed! Results do not match.")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()