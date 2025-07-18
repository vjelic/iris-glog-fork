import argparse
import os
import torch
import iris
import numpy as np

class MPIAllGatherLayer:
    def __init__(self, iris_instance: iris.Iris):
        self.iris = iris_instance
        self.rank = self.iris.get_rank()
        self.num_ranks = self.iris.get_num_ranks()

    def forward(self, local_data: torch.Tensor):
        original_dtype = local_data.dtype
        original_device = local_data.device
        
        local_data_numpy = local_data.cpu().numpy().astype(np.float32).flatten()
        
        gathered_numpy_flattened = iris._mpi_helpers.mpi_allgather(local_data_numpy)
        
        gathered_tensor = torch.from_numpy(gathered_numpy_flattened).to(original_device)
        
        reshaped_tensor = gathered_tensor.view(self.num_ranks, *local_data.shape)
        concatenated_tensor = torch.cat([reshaped_tensor[i] for i in range(self.num_ranks)])
        
        return concatenated_tensor.to(original_dtype)

def perf_func(func, warmup_iters, iters, sync_func=None):
    for _ in range(warmup_iters):
        func()
    if sync_func:
        sync_func()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        func()
    end.record()
    
    if sync_func:
        sync_func()
        
    torch.cuda.synchronize()
    return None, start.elapsed_time(end) / iters

def parse_args():
    parser = argparse.ArgumentParser(description="Test and benchmark MPI AllGather in Iris.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["int8", "int32", "float16", "float32"])
    parser.add_argument("--warmup_iters", type=int, default=10, help="Number of warmup iterations.")
    parser.add_argument("--iters", type=int, default=5, help="Number of main iterations for performance measurement.")
    parser.add_argument("--verify", default=True, action=argparse.BooleanOptionalAction, help="Run correctness verification before performance test.")
    return parser.parse_args()

def get_torch_dtype(name: str):
    return {"int8": torch.int8, "int32": torch.int32, "float16": torch.float16, "float32": torch.float32}[name]

def perf_mpi_ag(ag_op: MPIAllGatherLayer, iris_instance: iris.Iris, nbytes: int, dtype: torch.dtype, args: argparse.Namespace):
    world_size = iris_instance.get_num_ranks()
    rank = iris_instance.get_rank()
    
    nbytes_per_rank = nbytes // world_size
    if nbytes_per_rank % dtype.itemsize != 0:
        if rank == 0:
            print(f"Skipping size {nbytes // (1024*1024)} MB: not evenly divisible by dtype and world size.")
        return None, None
    elements_per_rank = nbytes_per_rank // dtype.itemsize
    
    def _verify():
        print(f"RANK[{rank}] Verifying for {nbytes // (1024 * 1024)} MB...")
        if rank == 0:
            ref_tensor = torch.randint(0, 1000, (nbytes // dtype.itemsize,), device="cuda", dtype=dtype)
        else:
            ref_tensor = torch.empty((nbytes // dtype.itemsize,), device="cuda", dtype=dtype)
        
        ref_tensor = torch.from_numpy(
            iris_instance.broadcast_tensor(ref_tensor, source_rank=0)
        ).to(ref_tensor.device)

        local_shard = ref_tensor[rank * elements_per_rank : (rank + 1) * elements_per_rank]
        result = ag_op.forward(local_shard)
        
        try:
            torch.testing.assert_close(result, ref_tensor, atol=1e-5, rtol=1e-5)
        except Exception as e:
            print(f"❌ RANK[{rank}] check FAILED!")
            raise e
        print(f"✅ RANK[{rank}] check PASSED!")

    if args.verify:
        _verify()
    
    iris_instance.barrier()

    local_perf_shard = torch.ones(elements_per_rank, device="cuda", dtype=dtype)
    _run_op = lambda: ag_op.forward(local_perf_shard)

    _, ag_time_ms = perf_func(
        _run_op,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
        sync_func=iris_instance.barrier
    )

    bus_gbps_calc = (lambda ms: nbytes * 1e-9 / (ms * 1e-3) * (world_size - 1) / world_size)
    
    latency_ms = ag_time_ms
    bus_gbps = bus_gbps_calc(ag_time_ms)

    print(
        f"RANK = {rank}, Size = {nbytes // (1024 * 1024): <4} MB, Latency = {latency_ms:7.3f} ms, Bus Bandwidth = {bus_gbps:6.2f} GB/s"
    )

    return latency_ms, bus_gbps

if __name__ == "__main__":
    args = parse_args()
    dtype = get_torch_dtype(args.dtype)
    
    iris_instance = iris.iris()
    rank = iris_instance.get_rank()
    torch.cuda.set_device(iris_instance.gpu_id)

    ag_op = MPIAllGatherLayer(iris_instance)

    sizes_mb = [1, 4, 16, 32, 64, 128, 256, 512]
    max_buffer_size_for_alloc = 512 * 1024 * 1024
    results = []

    for size_mb in sizes_mb:
        if size_mb * 1024 * 1024 > max_buffer_size_for_alloc:
            if rank == 0:
                print(f"Stopping benchmark. Next size {size_mb} MB exceeds max buffer size.")
            break
            
        nbytes = size_mb * 1024 * 1024
        
        if rank == 0:
            print("-" * 80)
        iris_instance.barrier()
        
        latency_ms, bus_gbps = perf_mpi_ag(ag_op, iris_instance, nbytes, dtype, args)
        
        if rank == 0 and latency_ms is not None:
            results.append({
                "size_mb": size_mb,
                "latency_ms": latency_ms,
                "bus_gbps": bus_gbps
            })

    if rank == 0 and results:
        print("\n" + "="*80)
        print(" " * 25 + "MPI All-Gather Performance Summary")
        print("="*80)
        
        header = f"{'Size (MB)':<15} | {'Latency (ms)':<20} | {'Bus Bandwidth (GB/s)':<25}"
        print(header)
        print("-" * len(header))
        
        for res in results:
            print(f"{res['size_mb']:<15} | {res['latency_ms']:<20.3f} | {res['bus_gbps']:<25.2f}")
        print("-" * len(header))