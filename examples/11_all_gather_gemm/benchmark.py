import torch
import iris
import os
import numpy as np
import argparse
import torch.distributed as dist
import triton
import triton.language as tl

from all_gather_gemm import FusedAGGemm
from all_gather_gemm_fused import FusedAGGemmFused
from all_gather_gemm_2 import PipelinedGemm

def dist_print(iris_instance, msg, allowed_ranks="all"):
    rank = iris_instance.get_rank()
    if allowed_ranks == "all" or rank in allowed_ranks:
        print(f"[RANK {rank}]: {msg}")

def test_correctness2(_iris, args):
    """
    Verifies both the fused kernel and the separate Triton GEMM kernel.
    """
    B, D, F, TP, dtype = 4, 8192, 28672, 8, torch.float16
    world_size = _iris.get_num_ranks()
    rank = _iris.get_rank()
    assert world_size == TP
    dist_print(_iris, f"🔬 Starting 3-way correctness test...", allowed_ranks=[0])
    
    D_local, F_local = D // TP, F // TP
    torch.manual_seed(1234 + rank)
    local_act = torch.randn((B, D_local), dtype=dtype)
    local_W_up = torch.randn((D, F_local), dtype=dtype)
    _iris.barrier()
    
    # --- 1. Compute Reference (RCCL + Torch GEMM) ---
    dist_print(_iris, "1. Computing reference (RCCL AG + Torch GEMM)...", allowed_ranks=[0])
    gathered_tensors = [torch.empty_like(local_act) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, local_act.contiguous())
    global_act = torch.cat(gathered_tensors, dim=1)
    ref_output = torch.matmul(global_act, local_W_up)
    _iris.barrier()

    # --- 2. Compute Fused Kernel Output ---
    dist_print(_iris, "2. Computing fused kernel output...", allowed_ranks=[0])
    # fused_op = FusedAGGemm(_iris, B, D, F, TP, dtype=dtype)
    fused_op = PipelinedGemm(_iris, TP) # No longer needs incorrect M,N,K

    # The forward call is now consistent with the GEMM you want to perform
    # For a standard GEMM test, A and B should have compatible shapes, for example:
    A_test = torch.randn((M, K), dtype=dtype)
    B_test = torch.randn((K, N), dtype=dtype)
    fused_output = fused_op.forward(A_test, B_test)
    _iris.barrier()

    # # --- 3. Compute RCCL AG + Triton GEMM Output ---
    # dist_print(_iris, "3. Computing sequential (RCCL AG + Triton GEMM) output...", allowed_ranks=[0])
    # triton_gemm_output = torch.empty((B, F_local), device="cuda", dtype=dtype)
    # BLK_M, BLK_N, BLK_K, GROUP_M, NUM_SMS = 128, 128, 32, 8, 112
    # persistent_gemm_local_store[(NUM_SMS,)](
    #     global_act, local_W_up, triton_gemm_output,
    #     B, F_local, D,
    #     global_act.stride(0), global_act.stride(1),
    #     local_W_up.stride(0), local_W_up.stride(1),
    #     triton_gemm_output.stride(0), triton_gemm_output.stride(1),
    #     BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
    #     GROUP_SIZE_M=GROUP_M, NUM_SMS=NUM_SMS, EVEN_K=(D % BLK_K == 0)
    # )
    # _iris.barrier()
    
    # --- 4. Verification ---
    dist_print(_iris, "\n4. Verifying outputs...", allowed_ranks=[0])
    try:
        torch.testing.assert_close(fused_output, ref_output, rtol=1e-2, atol=1e-2)
        dist_print(_iris, f"✅ Fused Kernel PASSED on Rank {rank}.", allowed_ranks="all")
    except AssertionError as e:
        dist_print(_iris, f"❌ Fused Kernel FAILED on Rank {rank}.", allowed_ranks="all")
        print(e)

    # try:
    #     torch.testing.assert_close(triton_gemm_output, ref_output, rtol=1e-2, atol=1e-2)
    #     dist_print(_iris, f"✅ Triton GEMM Kernel PASSED on Rank {rank}.", allowed_ranks="all")
    # except AssertionError as e:
    #     dist_print(_iris, f"❌ Triton GEMM Kernel FAILED on Rank {rank}.", allowed_ranks="all")
    #     print(e)
        
    _iris.barrier()

    # --- NEW: Detailed Failure Analysis for each rank ---
    # Calculate the absolute difference between the two outputs
    diff = torch.abs(fused_output - ref_output)
    
    # Find the value and location of the largest error
    max_diff_val = torch.max(diff)
    max_diff_idx_flat = torch.argmax(diff)
    
    # Convert the 1D index to 2D coordinates
    num_cols = fused_output.shape[1]
    row = max_diff_idx_flat // num_cols
    col = max_diff_idx_flat % num_cols
    
    # Get the specific values at the point of maximum difference
    fused_val = fused_output[row, col]
    ref_val = ref_output[row, col]
    
    # Print a formatted report
    print(f"---[ Rank {rank} Difference Details ]---")
    print(f"  Max difference      : {max_diff_val.item():.6f}")
    # print(f"  Location (row, col) : ({row.item()}, {col.item()})")
    # print(f"  Fused Kernel Value  : {fused_val.item():.6f}")
    # print(f"  Reference Value     : {ref_val.item():.6f}")
    # print(f"---------------------------------")
        
    _iris.barrier()
    
def test_correctness(_iris, args):
    """
    Verifies the new PipelinedGemm kernel.
    """
    TP = 8 # Tensor Parallelism, from your example
    dtype = torch.float16
    
    world_size = _iris.get_num_ranks()
    rank = _iris.get_rank()
    assert world_size == TP, f"This test requires TP={TP} ranks."
    
    # --- 1. Define Dimensions for a Standard GEMM Test ---
    # We will test a simple M,N,K that is compatible with the consumer kernel's logic.
    M = 512
    N = 512
    K = 1024
    
    # Ensure torch seed is different for each rank to create different local matrices
    torch.manual_seed(1234 + rank)
    
    # Create the input tensors for the consumer GEMM on each rank
    A_test = torch.randn((M, K), dtype=dtype, device="cuda")
    B_test = torch.randn((K, N), dtype=dtype, device="cuda")
    
    dist_print(_iris, f"🔬 Starting pipelined GEMM correctness test (M,N,K = {M},{N},{K})...", allowed_ranks=[0])
    _iris.barrier()

    # --- 2. Compute the Fused Pipelined Kernel Output ---
    dist_print(_iris, "1. Computing pipelined kernel output...", allowed_ranks=[0])
    pipelined_op = PipelinedGemm(_iris, TP)
    fused_output = pipelined_op.forward(A_test, B_test)
    _iris.barrier()

    # --- 3. Compute the Reference Output on the CPU ---
    # This must match the logic of both the producer and consumer kernels.
    dist_print(_iris, "2. Computing reference output...", allowed_ranks=[0])
    
    # 3a. This is what the consumer kernel computes locally before waiting
    ref_local_gemm = torch.matmul(A_test, B_test)
    
    # 3b. This is what the mock producer kernels send.
    # Each rank `r` produces tiles filled with the value `r + 1`.
    # The consumer adds the tiles from ALL ranks.
    producer_sum = sum(range(1, world_size + 1)) # Sum of (0+1) + (1+1) + ...
    
    # 3c. The final result is the sum of the local GEMM and all incoming data.
    ref_output = ref_local_gemm + producer_sum
    _iris.barrier()
    
    # --- 4. Verification ---
    dist_print(_iris, "\n3. Verifying outputs...", allowed_ranks=[0])
    try:
        # The tolerance might need to be adjusted depending on the hardware.
        torch.testing.assert_close(fused_output, ref_output, rtol=1e-2, atol=1e-2)
        dist_print(_iris, f"✅ Pipelined Kernel PASSED on Rank {rank}.", allowed_ranks="all")
    except AssertionError as e:
        dist_print(_iris, f"❌ Pipelined Kernel FAILED on Rank {rank}.", allowed_ranks="all")
        # Print a more detailed error report
        diff = torch.abs(fused_output - ref_output)
        max_diff_val = torch.max(diff)
        print(f"   Max difference on Rank {rank}: {max_diff_val.item():.6f}")
        print(e)
        
    _iris.barrier()

@triton.jit
def tile_id_to_index_range(tile_id, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    tile_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (tile_in_group % group_size_m)
    pid_n = tile_in_group // group_size_m
    rm_start = pid_m * BLOCK_SIZE_M
    rn_start = pid_n * BLOCK_SIZE_N
    rm = rm_start + tl.arange(0, BLOCK_SIZE_M)
    rn = rn_start + tl.arange(0, BLOCK_SIZE_N)
    return rm, rn, rm_start, rn_start

@triton.jit()
def persistent_gemm_local_store(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr, EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
    num_tiles_total = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    acc_dtype = tl.float32
    for tile_id in range(pid, num_tiles_total, NUM_SMS):
        rm, rn, _, _ = tile_id_to_index_range(tile_id, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
        rk = tl.arange(0, BLOCK_SIZE_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1
        for k in range(0, loop_k):
            a = tl.load(A_BASE, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
            b = tl.load(B_BASE, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk
        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rn[None, :] < N, other=0.0)
            acc += tl.dot(a, b)
        c = acc.to(C.type.element_ty)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_ptr, c, mask=c_mask)

def dist_print(iris_instance, msg, allowed_ranks="all"):
    rank = iris_instance.get_rank()
    if allowed_ranks == "all" or rank in allowed_ranks:
        print(f"[RANK {rank}]: {msg}")

def test_performance(_iris, args):
    # --- Benchmark Configuration ---
    B, D, F, TP, dtype = 16, 8192, 28672, 8, torch.float16
    world_size = _iris.get_num_ranks()
    rank = _iris.get_rank()
    assert world_size == TP
    dist_print(_iris, f"🚀 Starting Performance Benchmark...", allowed_ranks=[0])
    
    D_local, F_local = D // TP, F // TP
    torch.manual_seed(1234 + rank)
    local_act = torch.randn((B, D_local), dtype=dtype)
    local_W_up = torch.randn((D, F_local), dtype=dtype)
    _iris.barrier()

    # --- 1. Benchmark Reference (RCCL + torch.matmul) ---
    dist_print(_iris, "\n1. Benchmarking Reference (RCCL + Torch GEMM)...", allowed_ranks=[0])
    def reference_op_rccl_torch():
        gathered_tensors = [torch.empty_like(local_act) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, local_act.contiguous())
        global_act = torch.cat(gathered_tensors, dim=1)
        _ = torch.matmul(global_act, local_W_up)
    
    for _ in range(10): reference_op_rccl_torch()
    _iris.barrier(); torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(50): reference_op_rccl_torch()
    end_event.record()
    _iris.barrier(); torch.cuda.synchronize()
    ref_torch_ms = start_event.elapsed_time(end_event) / 50
    dist_print(_iris, f"   Done. Average time: {ref_torch_ms:.3f} ms", allowed_ranks=[0])

    # --- 2. Benchmark Reference (RCCL AG + Triton GEMM) ---
    dist_print(_iris, "\n2. Benchmarking Reference (RCCL AG + Triton GEMM)...", allowed_ranks=[0])
    triton_gemm_output = torch.empty((B, F_local), device="cuda", dtype=dtype)
    BLK_M, BLK_N, BLK_K, GROUP_M, NUM_SMS = 256, 64, 64, 4, 304
    def reference_op_rccl_triton():
        # Step 1: All-Gather with RCCL
        gathered_tensors = [torch.empty_like(local_act) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, local_act.contiguous())
        global_act = torch.cat(gathered_tensors, dim=1)
        
        # Step 2: GEMM with local Triton kernel
        persistent_gemm_local_store[(NUM_SMS,)](
            global_act, local_W_up, triton_gemm_output,
            B, F_local, D,
            global_act.stride(0), global_act.stride(1),
            local_W_up.stride(0), local_W_up.stride(1),
            triton_gemm_output.stride(0), triton_gemm_output.stride(1),
            BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=GROUP_M, NUM_SMS=NUM_SMS, EVEN_K=(D % BLK_K == 0)
        )

    for _ in range(10): reference_op_rccl_triton()
    _iris.barrier(); torch.cuda.synchronize()
    start_event_2 = torch.cuda.Event(enable_timing=True)
    end_event_2 = torch.cuda.Event(enable_timing=True)
    start_event_2.record()
    for _ in range(50): reference_op_rccl_triton()
    end_event_2.record()
    _iris.barrier(); torch.cuda.synchronize()
    ref_triton_ms = start_event_2.elapsed_time(end_event_2) / 50
    dist_print(_iris, f"   Done. Average time: {ref_triton_ms:.3f} ms", allowed_ranks=[0])

    # --- 3. Benchmark Fused Kernel ---
    dist_print(_iris, "\n3. Benchmarking Fused Kernel...", allowed_ranks=[0])
    fused_op = FusedAGGemm(_iris, B, D, F, TP, dtype=dtype)
    fused_time_ms = iris.do_bench(
        fn=lambda: fused_op.forward(local_act, local_W_up),
        preamble_fn=fused_op.clear_flags, barrier_fn=_iris.barrier,
        n_warmup=10, n_repeat=50, return_mode="mean"
    )
    dist_print(_iris, f"   Done. Average time: {fused_time_ms:.3f} ms", allowed_ranks=[0])


     # --- 3. Benchmark Fully Fused Kernel ---
    dist_print(_iris, "\n3. Benchmarking Fused Kernel...", allowed_ranks=[0])
    full_fused_op = FusedAGGemmFused(_iris, B, D, F, TP, dtype=dtype)
    full_fused_time_ms = iris.do_bench(
        fn=lambda: full_fused_op.forward(local_act, local_W_up),
        preamble_fn=full_fused_op.clear_flags, barrier_fn=_iris.barrier,
        n_warmup=10, n_repeat=50, return_mode="mean"
    )
    dist_print(_iris, f"   Done. Average time: {fused_time_ms:.3f} ms", allowed_ranks=[0])
    
    # --- 4. Summary ---
    if rank == 0:
        print("\n--- Performance Summary (Rank 0) ---")
        print(f"{'Implementation':<35} | {'Avg. Time (ms)':<20}")
        print("-" * 60)
        print(f"{'1. RCCL AG + Torch GEMM':<35} | {ref_torch_ms:<20.3f}")
        print(f"{'2. RCCL AG + Triton GEMM':<35} | {ref_triton_ms:<20.3f}")
        print(f"{'3. Fused Kernel (AG + GEMM)':<35} | {fused_time_ms:<20.3f}")
        print(f"{'4. Full Fused Kernel (AG + GEMM)':<35} | {full_fused_time_ms:<20.3f}")
        print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, choices=['correctness', 'performance'], default='performance')
    args = parser.parse_args()

    _iris = iris.iris()
    
    os.environ['RANK'] = str(_iris.get_rank())
    os.environ['WORLD_SIZE'] = str(_iris.get_num_ranks())
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group(backend="nccl")

    # --- FIX IS HERE ---
    # Get the local rank from the MPI environment variable to ensure each
    # process is assigned to a unique GPU.
    try:
        # This is the standard variable for OpenMPI
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    except KeyError:
        # Fallback for other MPI implementations
        num_gpus_on_node = torch.cuda.device_count()
        local_rank = _iris.get_rank() % num_gpus_on_node
        
    torch.cuda.set_device(local_rank)
    torch.set_default_device("cuda")

    if args.case == 'correctness':
        test_correctness(_iris, args)
    elif args.case == 'performance':
        test_performance(_iris, args)

    dist.destroy_process_group()