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

    A_full = torch.randn(M, K).cuda(rank)
    B_full = torch.randn(K, N).cuda(rank)

    # Split B column-wise
    cols_per_gpu = N // world_size
    start_col = rank * cols_per_gpu
    end_col = start_col + cols_per_gpu
    B_local = B_full[:, start_col:end_col]

    # Perform the local computation
    C_partial = A_full @ B_local

    # Prepare a tensor to gather all partial results
    C_gathered = torch.zeros(M, N).cuda(rank)

    # All-gather the results
    C_parts = list(torch.chunk(C_gathered, world_size, dim=1))
    dist.all_gather(C_parts, C_partial)

    # Combine the gathered parts
    C_global = torch.cat(C_parts, dim=1)

    # Validation step
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
