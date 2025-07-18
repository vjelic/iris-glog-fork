import argparse
import os
import torch
import iris
from all_gather_layer import IrisAllGatherLayer

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
    parser = argparse.ArgumentParser()
    # Note: minbytes and stepfactor are no longer used for the main loop
    parser.add_argument("-b", "--minbytes", type=int, default=1 * 1024 * 1024)
    parser.add_argument("-e", "--maxbytes", type=int, default=4096 * 1024 * 1024, help="Maximum buffer size in bytes for the test.")
    parser.add_argument("-f", "--stepfactor", default=2, type=int)
    parser.add_argument("--dtype", type=str, default="int32", choices=["int8", "int32", "float16", "float32"])
    parser.add_argument("--warmup_iters", type=int, default=30, help="Number of warmup iterations for performance measurement.")
    parser.add_argument("--iters", type=int, default=20, help="Number of main iterations for performance measurement.")
    parser.add_argument("--verify", default=True, action=argparse.BooleanOptionalAction, help="Run correctness verification before performance test.")
    args = parser.parse_args()
    return args


def get_torch_dtype(name: str):
    return {"int8": torch.int8, "int32": torch.int32, "float16": torch.float16, "float32": torch.float32}[name]


def perf_iris_ag(ag_op: IrisAllGatherLayer, iris_instance: iris.Iris, nbytes: int, dtype: torch.dtype, args: argparse.Namespace):
    world_size = iris_instance.get_num_ranks()
    rank = iris_instance.get_rank()
    
    nbytes_per_rank = nbytes // world_size
    if nbytes_per_rank % dtype.itemsize != 0: return
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
            torch.testing.assert_close(result, ref_tensor, atol=0, rtol=0)
        except Exception as e:
            print(f"❌ RANK[{rank}] check FAILED!")
            raise e
        print(f"✅ RANK[{rank}] check PASSED!")

    if args.verify:
        _verify()
    
    iris_instance.barrier()

    local_perf_shard = torch.ones(elements_per_rank, device="cuda", dtype=dtype)

    def _run_op():
        ag_op.forward(local_perf_shard)

    _, ag_time_ms = perf_func(
        _run_op,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
        sync_func=iris_instance.barrier
    )

    bus_gbps = (lambda ms: nbytes * 1e-9 / (ms * 1e-3) * (world_size - 1) / world_size)
    print(
        f"RANK = {rank}, Size = {nbytes // (1024 * 1024): <4} MB, Latency = {ag_time_ms * 1000:7.2f} us, Bus Bandwidth = {bus_gbps(ag_time_ms):6.2f} GB/s"
    )

if __name__ == "__main__":
    args = parse_args()
    dtype = get_torch_dtype(args.dtype)
    
    iris_instance = iris.iris()
    rank = iris_instance.get_rank()
    torch.cuda.set_device(iris_instance.gpu_id)

    max_buffer_size_for_alloc = 512 * 1024 * 1024
    ag_op = IrisAllGatherLayer(iris_instance, max_buffer_size=max_buffer_size_for_alloc, dtype=dtype)

  
    sizes_mb = [1, 4, 16, 32, 64, 128, 256, 512]


    for size_mb in sizes_mb:
        if size_mb * 1024 * 1024 > max_buffer_size_for_alloc:
            if rank == 0:
                print(f"Stopping benchmark. Next size {size_mb} MB exceeds allocated buffer of {max_buffer_size_for_alloc // (1024*1024)} MB.")
            break

        nbytes = size_mb * 1024 * 1024
        
        if rank == 0:
            print("-" * 80)
        iris_instance.barrier()
        perf_iris_ag(ag_op, iris_instance, nbytes, dtype, args)