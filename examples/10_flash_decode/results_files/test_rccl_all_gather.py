import argparse
import os
import torch
import torch.distributed as dist

class RCCLAllGatherLayer:
    """
    A simple wrapper for PyTorch's distributed all_gather_into_tensor
    to facilitate benchmarking against other communication libraries.
    """
    def __init__(self):
        # Initialize with the world size from the distributed environment.
        self.world_size = dist.get_world_size()

    def forward(self, local_data: torch.Tensor):
        """
        Performs the all-gather operation.

        Args:
            local_data (torch.Tensor): The tensor shard from the current rank.

        Returns:
            torch.Tensor: The fully gathered tensor containing data from all ranks.
        """
        # Calculate the total number of elements for the output tensor.
        total_elements = self.world_size * local_data.numel()
        
        # Pre-allocate the destination tensor to hold the gathered data.
        gathered_tensor = torch.empty(
            total_elements, dtype=local_data.dtype, device=local_data.device
        )
        
        # Perform the collective all-gather operation.
        dist.all_gather_into_tensor(gathered_tensor, local_data.contiguous())
        
        return gathered_tensor

def setup_distributed():
    """
    Initializes the PyTorch distributed process group using environment variables.
    """
    try:
        # These environment variables are typically set by launchers like torchrun or mpirun.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError:
        print("Distributed environment variables not found. Please use a launcher like torchrun.")
        exit(1)

    # Initialize the process group with the NCCL backend for GPU communication.
    dist.init_process_group(backend="nccl", init_method="env://")
    # Pin the current process to a specific GPU.
    torch.cuda.set_device(local_rank)
    return rank, world_size

def perf_func(func, warmup_iters, iters, sync_func=None):
    """
    A simple performance measurement utility.
    """
    # Warmup iterations to let the GPU reach a stable state.
    for _ in range(warmup_iters):
        func()
    if sync_func:
        sync_func()
    
    # Use CUDA events for accurate timing of GPU operations.
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        func()
    end.record()
    
    # Wait for all kernels to finish before calculating the elapsed time.
    torch.cuda.synchronize()
    if sync_func:
        sync_func()
    return None, start.elapsed_time(end) / iters

def parse_args():
    """
    Parses command-line arguments for the benchmark script.
    """
    parser = argparse.ArgumentParser(description="Test and benchmark NCCL AllGather in PyTorch.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["int8", "int32", "float16", "float32", "bfloat16"])
    parser.add_argument("--warmup_iters", type=int, default=25, help="Number of warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Number of main iterations for performance measurement.")
    parser.add_argument("--verify", default=True, action=argparse.BooleanOptionalAction, help="Run correctness verification before performance test.")
    return parser.parse_args()

def get_torch_dtype(name: str):
    """
    Converts a string dtype name to a torch.dtype object.
    """
    return {"int8": torch.int8, "int32": torch.int32, "float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[name]

def perf_rccl_ag(ag_op: RCCLAllGatherLayer, rank: int, world_size: int, num_tokens: int, nbytes: int, dtype: torch.dtype, args: argparse.Namespace):
    """
    Runs the verification and performance benchmark for a given size.
    """
    nbytes_per_rank = nbytes // world_size
    if nbytes_per_rank % dtype.itemsize != 0:
        if rank == 0:
            print(f"Skipping {num_tokens} tokens: size {nbytes} Bytes not divisible by world size and data type.")
        return None, None
    elements_per_rank = nbytes_per_rank // dtype.itemsize
    
    def _verify():
        """
        Verifies the correctness of the all-gather operation.
        """
        if rank == 0:
            print(f"Verifying for {num_tokens} tokens ({nbytes / (1024 * 1024):.2f} MB)...")
        
        # Create a reference tensor on rank 0 and broadcast it to all other ranks.
        if rank == 0:
            ref_tensor = torch.randint(0, 1000, (nbytes // dtype.itemsize,), device="cuda", dtype=dtype)
        else:
            ref_tensor = torch.empty((nbytes // dtype.itemsize,), device="cuda", dtype=dtype)
        
        dist.broadcast(ref_tensor, src=0)
        
        # Each rank takes its corresponding shard from the reference tensor.
        local_shard = ref_tensor[rank * elements_per_rank : (rank + 1) * elements_per_rank]
        
        # Perform the all-gather.
        result = ag_op.forward(local_shard)
        
        # Check if the gathered result matches the original reference tensor.
        try:
            torch.testing.assert_close(result, ref_tensor, atol=0, rtol=0)
        except Exception as e:
            print(f"❌ RANK[{rank}] check FAILED!")
            raise e
        
        if rank == 0:
            print(f"✅ Verification PASSED!")

    if args.verify:
        _verify()
    
    dist.barrier()

    # Create a tensor for the performance benchmark.
    local_perf_shard = torch.ones(elements_per_rank, device="cuda", dtype=dtype)
    _run_op = lambda: ag_op.forward(local_perf_shard)

    # Run the benchmark.
    _, ag_time_ms = perf_func(
        _run_op,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
        sync_func=dist.barrier
    )
    
    # Calculate bus bandwidth based on the timing.
    bus_gbps_calc = (lambda ms: nbytes * 1e-9 / (ms * 1e-3) * (world_size - 1) / world_size)
    
    latency_ms = ag_time_ms
    bus_gbps = bus_gbps_calc(ag_time_ms)
    
    if rank == 0:
        print(
            f"Tokens = {num_tokens: <8}, Size = {nbytes / (1024*1024):.2f} MB, Latency = {latency_ms:7.3f} ms, Bus Bandwidth = {bus_gbps:6.2f} GB/s"
        )
    
    return latency_ms, bus_gbps

if __name__ == "__main__":
    args = parse_args()
    dtype = get_torch_dtype(args.dtype)
    
    rank, world_size = setup_distributed()
    ag_op = RCCLAllGatherLayer()

    # --- Benchmark Configuration (Mirrors the Iris test) ---
    token_counts = [2, 4, 8, 16, 32, 64, 8192, 16384]
    bytes_per_token_per_rank = 12 * 128 * 2 * 2  # num_kv_heads * head_size * (k+v) * bytes_per_element
    
    results = []

    for num_tokens in token_counts:
        # Calculate the total number of bytes to be all-gathered across all ranks.
        nbytes = num_tokens * bytes_per_token_per_rank * world_size
        
        if rank == 0:
            print("-" * 80)
        dist.barrier()
        
        latency_ms, bus_gbps = perf_rccl_ag(ag_op, rank, world_size, num_tokens, nbytes, dtype, args)
        
        if rank == 0 and latency_ms is not None:
            nbytes_mb = nbytes / (1024 * 1024)
            results.append({
                "num_tokens": num_tokens,
                "size_mb": nbytes_mb,
                "latency_ms": latency_ms,
                "bus_gbps": bus_gbps
            })
            
    if rank == 0 and results:
        print("\n" + "="*80)
        print(" " * 24 + "RCCL All-Gather Performance Summary")
        print("="*80)
        
        header = f"{'Token Count (Size)':<25} | {'Latency (ms)':<20} | {'Bus Bandwidth (GB/s)':<25}"
        print(header)
        print("-" * (len(header)))
        
        for res in results:
            token_str = f"{res['num_tokens']} ({res['size_mb']:.2f} MB)"
            print(f"{token_str:<25} | {res['latency_ms']:<20.3f} | {res['bus_gbps']:<25.2f}")
        print("-" * (len(header)))
    
    # Clean up the distributed environment.
    dist.destroy_process_group()
