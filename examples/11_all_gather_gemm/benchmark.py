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
from ag_gemm_separate import AGThenGemm

IMPLEMENTATIONS = ["ag_then_gemm", "pipelined"]
PERF_IMPLEMENTATIONS = ["rccl", "fused", "pipelined", "ag_then_gemm"]


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
    
def test_correctness_old(_iris, args):
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
    ref_output = (A_test.float() @ B_test.float()) + float(producer_sum)
    ref_output = ref_output.half()
    # ref_output = ref_local_gemm + producer_sum
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
    
def test_correctness_fused(_iris, args=None):
    """
    Verifies the fused (1 + 2) kernel: all_gather(act) + matmul_up
    for LLaMA-70B decode shapes across several batch sizes.
    """
    import torch
    import torch.distributed as dist

    # --- LLaMA-70B decode shapes ---
    D = 8192
    F = 28672
    TP = 8
    dtype = torch.float16

    world_size = _iris.get_num_ranks()
    rank = _iris.get_rank()
    assert world_size == TP, f"Requires TP={TP} ranks, got world_size={world_size}"

    # Recommended: keep TF32 off for a cleaner numeric match with the kernel
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Batch sizes to test (decode). For prefill, multiply these by 512 as you noted.
    batch_list = [1, 2, 4, 8, 16, 32]

    from all_gather_gemm_fused import FusedAGGemmFused  # uses (1+2) fusion

    def dist_print(msg, allowed_ranks="all"):
        if allowed_ranks == "all" or rank in allowed_ranks:
            print(f"[RANK {rank}]: {msg}")

    for B in batch_list:
        dist_print(f"🔬 Testing (1+2) fused AG+UP with B={B}, D={D}, F={F}, TP={TP}", allowed_ranks=[0])

        # Per-rank local activation and local W_up shard
        D_local = D // TP           # 1024
        F_local = F // TP           # 3584
        torch.manual_seed(1234 + rank)  # different across ranks, deterministic

        # Local activation has shape (B, D/TP)
        act_local = torch.randn((B, D_local), device="cuda", dtype=dtype)

        # Local W_up shard has shape (D, F/TP)  (full-D rows, sharded columns)
        W_up_local = torch.randn((D, F_local), device="cuda", dtype=dtype)

        _iris.barrier()

        # --- Reference: AG via NCCL + matmul (float accum, single fp16 cast at end) ---
        gathered_parts = [torch.empty_like(act_local) for _ in range(world_size)]
        dist.all_gather(gathered_parts, act_local.contiguous())
        act_global = torch.cat(gathered_parts, dim=1)  # (B, D)

        # Do matmul in fp32 then cast once (matches kernel's fp32 accum + fp16 store)
        ref_output = (act_global.float() @ W_up_local.float()).half()  # (B, F/TP)
        _iris.barrier()

        # --- Fused kernel under test (all_gather + up matmul) ---
        fused_op = FusedAGGemmFused(_iris, M=B, K=D, N_global=F, TP=TP, dtype=dtype)
        fused_op.clear_flags()  # if implemented; harmless otherwise
        fused_output = fused_op.forward(act_local, W_up_local)  # (B, F/TP)
        _iris.barrier()

        # --- Check ---
        try:
            # With the fp32->fp16 single-cast reference, this should pass with a tight tol.
            torch.testing.assert_close(fused_output, ref_output, rtol=1e-2, atol=1e-2)
            dist_print(f"✅ PASSED for B={B}", allowed_ranks="all")
        except AssertionError as e:
            dist_print(f"❌ FAILED for B={B}", allowed_ranks="all")
            # Helpful diff dump
            diff = (fused_output - ref_output).abs()
            max_diff = diff.max().item()
            idx = diff.argmax().item()
            rows, cols = fused_output.shape
            r, c = divmod(idx, cols)
            fv = fused_output[r, c].item()
            rv = ref_output[r, c].item()
            dist_print(f"   Max |diff| = {max_diff:.6f} at (r={r}, c={c}); fused={fv:.6f}, ref={rv:.6f}", allowed_ranks="all")
            # If you still see fp16 quantization noise on some GPUs, relax atol a hair:
            # torch.testing.assert_close(fused_output, ref_output, rtol=1e-2, atol=3.2e-2)

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

def test_performance_old(_iris, args):
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
        
def test_performance_old(_iris, args=None, *, repeats=50, warmup=10):
    """
    Benchmark fused (1+2) kernel (all_gather + up-projection) vs.
    reference (RCCL all_gather + torch matmul) for LLaMA-70B decode shapes.
    Pretty prints per-B results and a summary table.
    """
    import torch
    import torch.distributed as dist
    from all_gather_gemm_fused import FusedAGGemmFused

    # ---------- Shapes & env ----------
    D = 8192
    F = 28672
    TP = 8
    dtype = torch.float16

    world_size = _iris.get_num_ranks()
    rank = _iris.get_rank()
    assert world_size == TP, f"Requires TP={TP} ranks, got world_size={world_size}"

    # Keep matmul paths aligned with kernel’s “fp32 accum → fp16 store”
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    batch_list = [1, 2, 4, 8, 16, 32,1024]
    D_local = D // TP           # 1024
    F_local = F // TP           # 3584

    def dist_print(msg, allowed="all"):
        if allowed == "all" or rank in allowed:
            print(f"[RANK {rank}]: {msg}")

    def time_ms(fn):
        # Warmup
        for _ in range(warmup):
            fn()
        _iris.barrier(); torch.cuda.synchronize()
        # Timed
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        _iris.barrier(); torch.cuda.synchronize()
        return start.elapsed_time(end) / repeats

    results = []  # (B, ref_ms, fused_ms, speedup)

    dist_print("🚀 Starting (1+2) performance benchmarks for LLaMA-70B decode…", allowed=[0])
    for B in batch_list:
        # Fresh inputs per B (deterministic per-rank)
        torch.manual_seed(1234 + rank)
        act_local  = torch.randn((B, D_local), device="cuda", dtype=dtype)
        W_up_local = torch.randn((D, F_local), device="cuda", dtype=dtype)

        # ---------- Reference: RCCL AG + torch matmul (fp32 -> fp16) ----------
        def ref_once():
            # AG
            parts = [torch.empty_like(act_local) for _ in range(world_size)]
            dist.all_gather(parts, act_local.contiguous())
            act_global = torch.cat(parts, dim=1)  # (B, D)
            # Matmul in fp32 then single cast to fp16
            _ = (act_global.float() @ W_up_local.float()).half()  # (B, F/TP)

        ref_ms = time_ms(ref_once)
        _iris.barrier()

        # ---------- Fused: AG+UP ----------
        fused_op = FusedAGGemmFused(_iris, M=B, K=D, N_global=F, TP=TP, dtype=dtype)

        def fused_once():
            # If your module exposes clear_flags(), call it to avoid reuse
            if hasattr(fused_op, "clear_flags"):
                fused_op.clear_flags()
            _ = fused_op.forward(act_local, W_up_local)

        fused_ms = time_ms(fused_once)
        _iris.barrier()

        speedup = ref_ms / fused_ms if fused_ms > 0 else float("inf")
        results.append((B, ref_ms, fused_ms, speedup))

        # Pretty per-B line
        if rank == 0:
            print(f"\nB={B:>2} | Ref (AG+torch) = {ref_ms:7.3f} ms  | Fused (AG+UP) = {fused_ms:7.3f} ms  | "
                  f"Speedup ×{speedup:0.2f}")

    # ---------- Summary table ----------
    if rank == 0:
        print("\n" + "═" * 70)
        print(f"{'LLaMA-70B Decode (TP=8) — (1+2) Performance Summary':^70}")
        print("═" * 70)
        print(f"{'B':>4} | {'Ref (AG+torch) ms':>18} | {'Fused (AG+UP) ms':>18} | {'Speedup':>8}")
        print("-" * 70)
        for B, ref_ms, fused_ms, speedup in results:
            print(f"{B:>4} | {ref_ms:18.3f} | {fused_ms:18.3f} | {speedup:>7.2f}×")
        print("─" * 70)
        best = max(results, key=lambda r: r[3])
        print(f"🏁 Best speedup: B={best[0]}  →  ×{best[3]:.2f} ("
              f"{best[1]:.3f} ms → {best[2]:.3f} ms)")
        print("═" * 70)

def test_correctness(_iris, args=None):
    """
    Runs correctness for any implementations listed in IMPLEMENTATIONS.
    - "pipelined": consumer GEMM + sum of mock producer tiles
                   Reference: (A @ B) + sum_{r=1..TP} r  (fp32 -> fp16 cast once)
    - "fused":     (1+2) all_gather(act) + up-projection matmul
                   Reference: NCCL all_gather + (fp32 matmul -> fp16 cast once)
    - "ag_then_gemm": Stage0(prev GEMM)->act_local, Stage1(explicit AG), Stage2(GEMM)
                   Reference: (A_prev @ W_prev_local)  -> AG ->  (act_global @ W_up_local)
    """
    import torch
    import torch.distributed as dist

    # --- LLaMA-70B decode constants ---
    D = 8192
    F = 28672
    TP = 8
    dtype = torch.float16
    K_PREV = 4096  # prev-GEMM K (choose as you like; must match AGThenGemm ctor)

    world_size = _iris.get_num_ranks()
    rank = _iris.get_rank()
    assert world_size == TP, f"Requires TP={TP}, got world_size={world_size}"

    # Align reference math with kernels: fp32 accum, single fp16 cast
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Decode batches. (For prefill, scale B by 512 as you noted.)
    batch_list = [1, 2, 4, 8, 16, 32]

    # Lazy imports of the impls
    from all_gather_gemm_2 import PipelinedGemm             # pipelined
    from all_gather_gemm_fused import FusedAGGemmFused      # fused (1+2)

    def dist_print(msg, allowed="all"):
        if allowed == "all" or rank in allowed:
            print(f"[RANK {rank}]: {msg}")

    dist_print("🔬 Starting correctness tests…", allowed=[0])

    for impl in IMPLEMENTATIONS:
        if impl not in {"pipelined", "fused", "ag_then_gemm"}:
            if rank == 0:
                print(f"⚠️ Skipping unknown implementation: {impl}")
            continue

        if rank == 0:
            print("\n" + "=" * 72)
            print(f"Checking implementation: {impl.upper()}")
            print("=" * 72)

        if impl == "pipelined":
            # Shapes for the consumer GEMM path
            K = D
            N = F // TP
            producer_sum = float(sum(range(1, world_size + 1)))  # 36 for TP=8

            for B in batch_list:
                dist_print(f"— {impl}: B={B}  (A:{B}x{K},  B:{K}x{N})", allowed=[0])

                torch.manual_seed(1234 + rank)
                A = torch.randn((B, K), device="cuda", dtype=dtype)
                Bmat = torch.randn((K, N), device="cuda", dtype=dtype)
                _iris.barrier()

                ref = (A.float() @ Bmat.float()).add_(producer_sum).half()
                _iris.barrier()

                pipe = PipelinedGemm(_iris, TP)
                out = pipe.forward(A, Bmat)
                _iris.barrier()

                try:
                    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
                    dist_print(f"✅ PASSED for B={B}", allowed="all")
                except AssertionError:
                    dist_print(f"❌ FAILED for B={B}", allowed="all")
                    diff = (out - ref).abs()
                    mx = float(diff.max()); idx = int(diff.argmax())
                    rows, cols = out.shape; r, c = divmod(idx, cols)
                    dist_print(f"   Max |diff| = {mx:.6f} at (r={r}, c={c}); "
                               f"pipelined={out[r,c].item():.6f}, ref={ref[r,c].item():.6f}", allowed="all")
                _iris.barrier()

        elif impl == "fused":
            D_local = D // TP
            F_local = F // TP

            for B in batch_list:
                dist_print(f"— {impl}: B={B}  (act_local:{B}x{D_local},  W_up:{D}x{F_local})", allowed=[0])

                torch.manual_seed(1234 + rank)
                act_local  = torch.randn((B, D_local), device="cuda", dtype=dtype)
                W_up_local = torch.randn((D, F_local),  device="cuda", dtype=dtype)
                _iris.barrier()

                parts = [torch.empty_like(act_local) for _ in range(world_size)]
                dist.all_gather(parts, act_local.contiguous())
                act_global = torch.cat(parts, dim=1)  # (B, D)
                ref = (act_global.float() @ W_up_local.float()).half()
                _iris.barrier()

                fused = FusedAGGemmFused(_iris, M=B, K=D, N_global=F, TP=TP, dtype=dtype)
                if hasattr(fused, "clear_flags"):
                    fused.clear_flags()
                out = fused.forward(act_local, W_up_local)
                _iris.barrier()

                try:
                    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
                    dist_print(f"✅ PASSED for B={B}", allowed="all")
                except AssertionError:
                    dist_print(f"❌ FAILED for B={B}", allowed="all")
                    diff = (out - ref).abs()
                    mx = float(diff.max()); idx = int(diff.argmax())
                    rows, cols = out.shape; r, c = divmod(idx, cols)
                    dist_print(f"   Max |diff| = {mx:.6f} at (r={r}, c={c}); "
                               f"fused={out[r,c].item():.6f}, ref={ref[r,c].item():.6f}", allowed="all")
                _iris.barrier()

        elif impl == "ag_then_gemm":
            D_local = D // TP
            F_local = F // TP

            for B in batch_list:
                dist_print(f"— {impl}: B={B}  (A_prev:{B}x{K_PREV},  W_prev:{K_PREV}x{D_local},  W_up:{D}x{F_local})",
                           allowed=[0])

                torch.manual_seed(1234 + rank)
                A_prev        = torch.randn((B, K_PREV),   device="cuda", dtype=dtype)
                W_prev_local  = torch.randn((K_PREV, D_local), device="cuda", dtype=dtype)
                W_up_local    = torch.randn((D, F_local),  device="cuda", dtype=dtype)
                _iris.barrier()

                # Reference: prev GEMM -> AG -> up GEMM (fp32, then fp16 once)
                act_local_ref = (A_prev.float() @ W_prev_local.float()).half()
                parts = [torch.empty_like(act_local_ref) for _ in range(world_size)]
                dist.all_gather(parts, act_local_ref.contiguous())
                act_global_ref = torch.cat(parts, dim=1)                   # (B, D)
                ref = (act_global_ref.float() @ W_up_local.float()).half() # (B, F/TP)
                _iris.barrier()

                # DUT
                op = AGThenGemm(_iris, B=B, D=D, F=F, TP=TP, K_prev=K_PREV, dtype=dtype)
                op.clear_flags()
                out = op.forward(A_prev, W_prev_local, W_up_local)
                _iris.barrier()

                try:
                    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
                    dist_print(f"✅ PASSED for B={B}", allowed="all")
                except AssertionError:
                    dist_print(f"❌ FAILED for B={B}", allowed="all")
                    diff = (out - ref).abs()
                    mx = float(diff.max()); idx = int(diff.argmax())
                    rows, cols = out.shape; r, c = divmod(idx, cols)
                    dist_print(f"   Max |diff| = {mx:.6f} at (r={r}, c={c}); "
                               f"ag_then={out[r,c].item():.6f}, ref={ref[r,c].item():.6f}", allowed="all")
                _iris.barrier()
        
def test_performance(_iris, args=None, *, repeats=50, warmup=10):
    """
    Benchmarks implementations listed in PERF_IMPLEMENTATIONS for LLaMA-70B decode sizes.
    Columns: RCCL(AG+torch GEMM), Fused(AG+UP), Pipelined, AG-Then-GEMM (+ its own RCCL baseline).
    Pretty per-B lines and a summary table at the end.
    """
    import torch
    import torch.distributed as dist
    from all_gather_gemm_fused import FusedAGGemmFused   # (1+2) fused AG+UP
    from all_gather_gemm_2 import PipelinedGemm         # pipelined impl

    # ---- LLaMA-70B decode shapes ----
    D  = 8192
    F  = 28672
    TP = 8
    dtype = torch.float16
    K_PREV = 4096

    world_size = _iris.get_num_ranks()
    rank = _iris.get_rank()
    assert world_size == TP, f"Requires TP={TP}, got world_size={world_size}"

    # Keep math consistent with kernels
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    batch_list = [1, 2, 4, 8, 16, 32]
    D_local = D // TP
    F_local = F // TP
    K = D
    N = F_local
    producer_sum = float(sum(range(1, world_size + 1)))  # 36 for TP=8

    def dist_print(msg, allowed="all"):
        if allowed == "all" or rank in allowed:
            print(f"[RANK {rank}]: {msg}")

    def time_ms(fn):
        # Warmup
        for _ in range(warmup):
            fn()
        _iris.barrier(); torch.cuda.synchronize()
        # Timed
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        _iris.barrier(); torch.cuda.synchronize()
        return start.elapsed_time(end) / repeats

    # results[B] = dict with keys present in PERF_IMPLEMENTATIONS (+ rccl_ag_then if used)
    results = {B: {} for B in batch_list}

    if rank == 0:
        print("🚀 Performance benchmarks (decode) — running:", ", ".join(PERF_IMPLEMENTATIONS))

    for B in batch_list:
        # Fresh inputs per B, deterministic per rank
        torch.manual_seed(1234 + rank)

        # Inputs for fused path
        act_local  = torch.randn((B, D_local), device="cuda", dtype=dtype)
        W_up_local = torch.randn((D, F_local),  device="cuda", dtype=dtype)

        # Inputs for pipelined path
        A = torch.randn((B, K), device="cuda", dtype=dtype)
        Bmat = torch.randn((K, N), device="cuda", dtype=dtype)

        # Inputs for ag_then_gemm path
        A_prev       = torch.randn((B, K_PREV),          device="cuda", dtype=dtype)
        W_prev_local = torch.randn((K_PREV, D_local),    device="cuda", dtype=dtype)
        W_up_local_2 = torch.randn((D, F_local),         device="cuda", dtype=dtype)  # separate to avoid reuse bias

        # ---------- RCCL baseline: AG + torch GEMM (for the fused scenario) ----------
        if "rccl" in PERF_IMPLEMENTATIONS:
            def rccl_once():
                parts = [torch.empty_like(act_local) for _ in range(world_size)]
                dist.all_gather(parts, act_local.contiguous())
                act_global = torch.cat(parts, dim=1)  # (B, D)
                _ = (act_global.float() @ W_up_local.float()).half()  # (B, F/TP)
            rccl_ms = time_ms(rccl_once)
            results[B]["rccl"] = rccl_ms
            _iris.barrier()
        else:
            rccl_ms = None

        # ---------- Fused (AG+UP) ----------
        if "fused" in PERF_IMPLEMENTATIONS:
            fused_op = FusedAGGemmFused(_iris, M=B, K=D, N_global=F, TP=TP, dtype=dtype)
            def fused_once():
                if hasattr(fused_op, "clear_flags"):
                    fused_op.clear_flags()
                _ = fused_op.forward(act_local, W_up_local)
            fused_ms = time_ms(fused_once)
            results[B]["fused"] = fused_ms
            _iris.barrier()

        # ---------- Pipelined (consumer GEMM + mock producer tiles) ----------
        if "pipelined" in PERF_IMPLEMENTATIONS:
            pipe = PipelinedGemm(_iris, TP)
            def pipelined_once():
                _ = pipe.forward(A, Bmat)
            pip_ms = time_ms(pipelined_once)
            results[B]["pipelined"] = pip_ms
            _iris.barrier()

        # ---------- AG-Then-GEMM (prev GEMM -> explicit AG -> up GEMM) ----------
        if "ag_then_gemm" in PERF_IMPLEMENTATIONS:
            # RCCL baseline for this path: (A_prev @ W_prev_local) -> AG -> torch GEMM
            def rccl_ag_then_once():
                act_local_ref = (A_prev.float() @ W_prev_local.float()).half()
                parts = [torch.empty_like(act_local_ref) for _ in range(world_size)]
                dist.all_gather(parts, act_local_ref.contiguous())
                act_global_ref = torch.cat(parts, dim=1)
                _ = (act_global_ref.float() @ W_up_local_2.float()).half()
            rccl_ag_then_ms = time_ms(rccl_ag_then_once)
            results[B]["rccl_ag_then"] = rccl_ag_then_ms
            _iris.barrier()

            # DUT timing
            op = AGThenGemm(_iris, B=B, D=D, F=F, TP=TP, K_prev=K_PREV, dtype=dtype)
            def ag_then_once():
                op.clear_flags()
                _ = op.forward(A_prev, W_prev_local, W_up_local_2)
            ag_then_ms = time_ms(ag_then_once)
            results[B]["ag_then_gemm"] = ag_then_ms
            _iris.barrier()

        # ---------- Per-B pretty print ----------
        if rank == 0:
            print(f"\nB={B:>2} | ", end="")
            if "rccl" in PERF_IMPLEMENTATIONS:
                print(f"RCCL(AG+torch) = {results[B]['rccl']:7.3f} ms  | ", end="")
            if "fused" in PERF_IMPLEMENTATIONS:
                if rccl_ms:
                    sp = results[B]['rccl'] / results[B]['fused']
                    print(f"Fused(AG+UP) = {results[B]['fused']:7.3f} ms  | ×{sp:0.2f} vs RCCL  | ", end="")
                else:
                    print(f"Fused(AG+UP) = {results[B]['fused']:7.3f} ms  | ", end="")
            if "pipelined" in PERF_IMPLEMENTATIONS:
                if rccl_ms:
                    print(f"Pipelined = {results[B]['pipelined']:7.3f} ms  | ×{results[B]['rccl']/results[B]['pipelined']:0.2f} vs RCCL  | ", end="")
                else:
                    print(f"Pipelined = {results[B]['pipelined']:7.3f} ms  | ", end="")
            if "ag_then_gemm" in PERF_IMPLEMENTATIONS:
                print(f"AG-Then = {results[B]['ag_then_gemm']:7.3f} ms  | ×{results[B]['rccl_ag_then']/results[B]['ag_then_gemm']:0.2f} vs RCCL(AG-Then)", end="")
            print()

    # ---------- Summary table ----------
    if rank == 0:
        cols = ["rccl", "fused", "pipelined", "rccl_ag_then", "ag_then_gemm"]
        active = [c for c in cols if c in PERF_IMPLEMENTATIONS or c.startswith("rccl_ag_then")]

        # Only include rccl_ag_then column if ag_then_gemm was requested
        if "ag_then_gemm" not in PERF_IMPLEMENTATIONS and "rccl_ag_then" in active:
            active.remove("rccl_ag_then")

        print("\n" + "═" * (16 + 18 * len(active)))
        title = f"Decode Performance Summary (TP={TP})"
        print(f"{title:^{16 + 18 * len(active)}}")
        print("═" * (16 + 18 * len(active)))

        header = f"{'B':>4} "
        for c in active:
            label = {
                "rccl":"RCCL(AG+torch)",
                "fused":"Fused(AG+UP)",
                "pipelined":"Pipelined",
                "rccl_ag_then":"RCCL(AG-Then)",
                "ag_then_gemm":"AG-Then"
            }[c]
            header += f"| {label:>14} "
        print(header)
        print("-" * (16 + 18 * len(active)))

        for B in batch_list:
            line = f"{B:>4} "
            for c in active:
                val = results[B].get(c, float('nan'))
                line += f"| {val:14.3f} "
            print(line)

        print("─" * (16 + 18 * len(active)))

        # Best speedups
        if "rccl" in active and "fused" in active:
            best = max(batch_list, key=lambda b: results[b]["rccl"] / results[b]["fused"])
            sp = results[best]["rccl"] / results[best]["fused"]
            print(f"🏁 Best fused speedup vs RCCL: B={best} → ×{sp:.2f} "
                  f"({results[best]['rccl']:.3f} → {results[best]['fused']:.3f} ms)")
        if "ag_then_gemm" in active and "rccl_ag_then" in active:
            best = max(batch_list, key=lambda b: results[b]["rccl_ag_then"] / results[b]["ag_then_gemm"])
            sp = results[best]["rccl_ag_then"] / results[best]["ag_then_gemm"]
            print(f"🏁 Best AG-Then speedup vs RCCL(AG-Then): B={best} → ×{sp:.2f} "
                  f"({results[best]['rccl_ag_then']:.3f} → {results[best]['ag_then_gemm']:.3f} ms)")

        print("═" * (16 + 18 * len(active)))



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