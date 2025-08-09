import torch
import iris
import os
import argparse
import torch.distributed as dist
import triton

# Use the matmul wrapper function as requested
from matmul_wrapper import matmul

def dist_print(iris_instance, msg, allowed_ranks="all"):
    rank = iris_instance.get_rank()
    if allowed_ranks == "all" or rank in allowed_ranks:
        print(f"[RANK {rank}]: {msg}")

def test_correctness(_iris, args):
    """
    Verifies the fused kernel inside the matmul wrapper.
    """
    # Configuration
    M, K, N_global, TP, dtype = 4, 8192, 28672, 8, torch.float16
    world_size = _iris.get_num_ranks()
    rank = _iris.get_rank()
    assert world_size == TP
    dist_print(_iris, f"🔬 Starting correctness test for matmul wrapper...", allowed_ranks=[0])
    
    K_local = K // TP
    N_local = N_global // TP

    # --- Tensor Initialization ---
    torch.manual_seed(1234 + rank)
    local_A = torch.randn((M, K_local), dtype=dtype)
    local_B = torch.randn((K, N_local), dtype=dtype)
    _iris.barrier()
    
    # --- Reference Implementation (RCCL) ---
    gathered_tensors = [torch.empty_like(local_A) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, local_A.contiguous())
    global_A = torch.cat(gathered_tensors, dim=1)
    ref_output = torch.matmul(global_A, local_B)
    _iris.barrier()

    # --- Fused Implementation via matmul wrapper ---
    # Create all necessary arguments for the matmul.apply function signature
    local_C = torch.empty((M, N_local), device="cuda", dtype=dtype)
    global_C = torch.empty((M, N_global), device="cuda", dtype=dtype) # Placeholder
    BLK_M, BLK_N, BLK_K = 128, 64, 64
    total_tiles = triton.cdiv(M, BLK_M) * triton.cdiv(N_local, BLK_N)
    tile_completed = torch.zeros((total_tiles,), device="cuda", dtype=torch.int32)
    locks = torch.zeros((args.gemm_sms,), device="cuda", dtype=torch.int32)
    P = torch.zeros((args.gemm_sms, BLK_M * BLK_N), device="cuda", dtype=torch.float32)

    fused_output = matmul.apply(
        _iris,
        local_A, local_B, local_C, global_C, None, P, locks, tile_completed,
        rank, world_size, args.gemm_sms, BLK_M, BLK_N, BLK_K, args.gsize_m,
        True, 2, 8, 16, 1, _iris.get_heap_bases(), _iris.get_cu_count()
    )
    _iris.barrier()

    # --- Verification ---
    try:
        torch.testing.assert_close(fused_output, ref_output, rtol=1e-2, atol=1e-2)
        dist_print(_iris, f"✅ TEST PASSED on Rank {rank}.", allowed_ranks="all")
    except AssertionError as e:
        # (Detailed failure analysis logic would go here)
        dist_print(_iris, f"❌ TEST FAILED on Rank {rank}.", allowed_ranks="all")
        print(e)
    _iris.barrier()

def test_performance(_iris, args):
    """
    Benchmarks the fused kernel inside the matmul wrapper.
    """
    M, K, N_global, TP, dtype = 4, 8192, 28672, 8, torch.float16
    world_size = _iris.get_num_ranks()
    rank = _iris.get_rank()
    assert world_size == TP
    dist_print(_iris, f"🚀 Starting Performance Benchmark for matmul wrapper...", allowed_ranks=[0])
    
    K_local, N_local = K // TP, N_global // TP
    torch.manual_seed(1234 + rank)
    local_A = torch.randn((M, K_local), dtype=dtype)
    local_B = torch.randn((K, N_local), dtype=dtype)

    # --- Setup for matmul.apply call ---
    local_C = torch.empty((M, N_local), device="cuda", dtype=dtype)
    global_C = torch.empty((M, N_global), device="cuda", dtype=dtype)
    BLK_M, BLK_N, BLK_K = 128, 64, 64
    total_tiles = triton.cdiv(M, BLK_M) * triton.cdiv(N_local, BLK_N)
    tile_completed = torch.zeros((total_tiles,), device="cuda", dtype=torch.int32)
    locks = torch.zeros((args.gemm_sms,), device="cuda", dtype=torch.int32)
    P = torch.zeros((args.gemm_sms, BLK_M * BLK_N), device="cuda", dtype=torch.float32)
    heap_bases = _iris.get_heap_bases()
    cu_count = _iris.get_cu_count()

    # --- Benchmark Fused Kernel ---
    def fused_op():
        matmul.apply(
            local_A, local_B, local_C, global_C, None, P, locks, tile_completed,
            rank, world_size, args.gemm_sms, BLK_M, BLK_N, BLK_K, args.gsize_m,
            True, 2, 8, 16, 1, heap_bases, cu_count
        )

    fused_time_ms = iris.do_bench(
        fn=fused_op,
        barrier_fn=_iris.barrier,
        n_warmup=10, n_repeat=50, return_mode="mean"
    )
    dist_print(_iris, f"   Done. Fused matmul wrapper time: {fused_time_ms:.3f} ms", allowed_ranks=[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, choices=['correctness', 'performance'], default='performance')
    # Add arguments expected by the test functions
    parser.add_argument("--gemm_sms", type=int, default=112)
    parser.add_argument("--gsize_m", type=int, default=8)
    args = parser.parse_args()

    # --- Environment Setup ---
    _iris = iris.iris()
    os.environ['RANK'] = str(_iris.get_rank())
    os.environ['WORLD_SIZE'] = str(_iris.get_num_ranks())
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="nccl")
    try:
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    except KeyError:
        local_rank = _iris.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    torch.set_default_device("cuda")

    # --- Run selected test ---
    if args.case == 'correctness':
        test_correctness(_iris, args)
    elif args.case == 'performance':
        test_performance(_iris, args)

    dist.destroy_process_group()