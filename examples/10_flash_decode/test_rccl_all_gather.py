import argparse
import os
import torch
import torch.distributed as dist

class RCCLAllGatherLayer:
    def __init__(self):
        self.world_size = dist.get_world_size()

    def forward(self, local_data: torch.Tensor):
        total_elements = self.world_size * local_data.numel()
        gathered_tensor = torch.empty(
            total_elements, dtype=local_data.dtype, device=local_data.device
        )
        dist.all_gather_into_tensor(gathered_tensor, local_data.contiguous())
        return gathered_tensor

def setup_distributed():
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError:
        print("Distributed environment variables not found. Please use a launcher like torchrun.")
        exit(1)

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return rank, world_size

def perf_func(func, warmup_iters, iters, sync_func=None):
    for _ in range(warmup_iters):
        func()
    if sync_func:
        sync_func()
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        func()
    end.record()
    
    torch.cuda.synchronize()
    if sync_func:
        sync_func()
    return None, start.elapsed_time(end) / iters

def parse_args():
    parser = argparse.ArgumentParser(description="Test and benchmark NCCL AllGather in PyTorch.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["int8", "int32", "float16", "float32", "bfloat16"])
    parser.add_argument("--warmup_iters", type=int, default=30, help="Number of warmup iterations.")
    parser.add_argument("--iters", type=int, default=20, help="Number of main iterations for performance measurement.")
    parser.add_argument("--verify", default=True, action=argparse.BooleanOptionalAction, help="Run correctness verification before performance test.")
    return parser.parse_args()

def get_torch_dtype(name: str):
    return {"int8": torch.int8, "int32": torch.int32, "float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[name]

def perf_rccl_ag(ag_op: RCCLAllGatherLayer, rank: int, world_size: int, nbytes: int, dtype: torch.dtype, args: argparse.Namespace):
    nbytes_per_rank = nbytes // world_size
    if nbytes_per_rank % dtype.itemsize != 0:
        if rank == 0:
            print(f"Skipping size {nbytes // (1024*1024)} MB: not evenly divisible by dtype and world size.")
        return None, None
    elements_per_rank = nbytes_per_rank // dtype.itemsize
    
    def _verify():
        if rank == 0:
            print(f"Verifying for {nbytes // (1024 * 1024)} MB...")
        
        if rank == 0:
            ref_tensor = torch.randint(0, 1000, (nbytes // dtype.itemsize,), device="cuda", dtype=dtype)
        else:
            ref_tensor = torch.empty((nbytes // dtype.itemsize,), device="cuda", dtype=dtype)
        
        dist.broadcast(ref_tensor, src=0)
        local_shard = ref_tensor[rank * elements_per_rank : (rank + 1) * elements_per_rank]
        result = ag_op.forward(local_shard)
        
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

    local_perf_shard = torch.ones(elements_per_rank, device="cuda", dtype=dtype)
    _run_op = lambda: ag_op.forward(local_perf_shard)

    _, ag_time_ms = perf_func(
        _run_op,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
        sync_func=dist.barrier
    )
    
    bus_gbps_calc = (lambda ms: nbytes * 1e-9 / (ms * 1e-3) * (world_size - 1) / world_size)
    
    latency_ms = ag_time_ms
    bus_gbps = bus_gbps_calc(ag_time_ms)
    
    if rank == 0:
        print(
            f"Size = {nbytes // (1024 * 1024): <4} MB, Latency = {latency_ms:7.3f} ms, Bus Bandwidth = {bus_gbps:6.2f} GB/s"
        )
    
    return latency_ms, bus_gbps

if __name__ == "__main__":
    args = parse_args()
    dtype = get_torch_dtype(args.dtype)
    
    rank, world_size = setup_distributed()
    ag_op = RCCLAllGatherLayer()

    sizes_mb = [1, 4, 16, 32, 64, 128, 256, 512]
    results = []

    for size_mb in sizes_mb:
        nbytes = size_mb * 1024 * 1024
        
        if rank == 0:
            print("-" * 80)
        dist.barrier()
        
        latency_ms, bus_gbps = perf_rccl_ag(ag_op, rank, world_size, nbytes, dtype, args)
        
        if rank == 0 and latency_ms is not None:
            results.append({
                "size_mb": size_mb,
                "latency_ms": latency_ms,
                "bus_gbps": bus_gbps
            })
            
    if rank == 0 and results:
        print("\n" + "="*80)
        print(" " * 24 + "RCCL All-Gather Performance Summary")
        print("="*80)
        
        header = f"{'Size (MB)':<15} | {'Latency (ms)':<20} | {'Bus Bandwidth (GB/s)':<25}"
        print(header)
        print("-" * len(header))
        
        for res in results:
            print(f"{res['size_mb']:<15} | {res['latency_ms']:<20.3f} | {res['bus_gbps']:<25.2f}")
        print("-" * len(header))

    dist.destroy_process_group()