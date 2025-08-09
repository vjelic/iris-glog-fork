import torch
import iris
import os
import numpy as np

# Import the refactored FusedAGGemm class
from all_gather_gemm import FusedAGGemm

def dist_print(iris_instance, msg, allowed_ranks="all"):
    rank = iris_instance.get_rank()
    if allowed_ranks == "all" or rank in allowed_ranks:
        print(f"[RANK {rank}]: {msg}")

def test_correctness():
    # --- Test Configuration ---
    B = 4
    D = 2048
    F = 7168
    TP = 8
    dtype = torch.float16
    
    _iris = iris.iris()
    world_size = _iris.get_num_ranks()
    rank = _iris.get_rank()
    
    assert world_size == TP, f"This test requires TP={TP} GPUs, but got {world_size}."
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.set_default_device("cuda")

    dist_print(_iris, f"🔬 Starting correctness test with refactored AG+GEMM.", allowed_ranks=[0])
    dist_print(_iris, f"   Config (Small): B={B}, D={D}, F={F}, TP={TP}", allowed_ranks=[0])
    
    D_local = D // TP
    F_local = F // TP

    torch.manual_seed(1234 + rank)
    local_act = torch.randn((B, D_local), dtype=dtype)
    local_W_up = torch.randn((D, F_local), dtype=dtype)

    _iris.barrier()

    # --- Reference Implementation ---
    dist_print(_iris, "1. Computing reference output (manual all-gather + gemm)...", allowed_ranks=[0])
    local_act_numpy = local_act.cpu().numpy()
    gathered_act_numpy_flat = iris._mpi_helpers.mpi_allgather(local_act_numpy.flatten())
    gathered_act_numpy = gathered_act_numpy_flat.reshape(world_size, B, D_local).transpose(1, 0, 2).reshape(B, D)
    global_act = torch.from_numpy(gathered_act_numpy).to(device="cuda", dtype=dtype)
    ref_output = torch.matmul(global_act, local_W_up)
    
    _iris.barrier()
    dist_print(_iris, "   Reference computation complete.", allowed_ranks=[0])

    # --- Fused Implementation ---
    dist_print(_iris, "2. Computing fused kernel output...", allowed_ranks=[0])
    fused_op = FusedAGGemm(_iris, B, D, F, TP, dtype=dtype)
    fused_output = fused_op.forward(local_act, local_W_up)
    
    _iris.barrier()
    dist_print(_iris, "   Fused computation complete.", allowed_ranks=[0])
    
    # --- Verification ---
    dist_print(_iris, "3. Comparing outputs...", allowed_ranks=[0])
    
    try:
        torch.testing.assert_close(fused_output, ref_output, rtol=1e-2, atol=1e-2)
        dist_print(_iris, f"✅ TEST PASSED on Rank {rank}. Outputs are numerically close.", allowed_ranks="all")
    except AssertionError as e:
        dist_print(_iris, f"❌ TEST FAILED on Rank {rank}.", allowed_ranks="all")
        print(e)

    _iris.barrier()

if __name__ == "__main__":
    test_correctness()