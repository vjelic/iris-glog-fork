import torch
import torch.distributed as dist

def main():

    torch.manual_seed(42)

    M, K, N = 8, 8, 8

    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    print(f"Starting distributed GEMM on Rank {rank} of {world_size} on device cuda:{rank}")

    A_full = torch.randn(M, K).cuda(rank)
    B_full = torch.randn(K, N).cuda(rank)
    rows_per_gpu = K // world_size
    start_row = rank * rows_per_gpu
    end_row = start_row + rows_per_gpu
    B_local = B_full[start_row:end_row, :]
    A_local = A_full[:, start_row:end_row]
    C_partial = A_local @ B_local
    C_global = torch.zeros_like(C_partial)
    dist.all_reduce(C_partial, op=dist.ReduceOp.SUM)
    C_global.copy_(C_partial)

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