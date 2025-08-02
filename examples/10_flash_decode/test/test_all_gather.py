import argparse
import os
import torch
import iris
from all_gather_layer import IrisAllGatherLayer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default="float16", choices=["int8", "int32", "float16", "float32"])
    parser.add_argument("--n_warmup", type=int, default=4, help="Number of warmup iterations for performance measurement.")
    parser.add_argument("--n_repeat", type=int, default=10, help="Number of main iterations for performance measurement.")
    parser.add_argument("--verify", default=True, action=argparse.BooleanOptionalAction, help="Run correctness verification before performance test.")
    args = parser.parse_args()
    return args


def get_torch_dtype(name: str):
    return {"int8": torch.int8, "int32": torch.int32, "float16": torch.float16, "float32": torch.float32}[name]


def perf_iris_ag(ag_op: IrisAllGatherLayer, num_tokens: int, nbytes: int, dtype: torch.dtype, args: argparse.Namespace):
    world_size = iris_instance.get_num_ranks()
    rank = iris_instance.get_rank()
    
    nbytes_per_rank = nbytes // world_size
    if nbytes_per_rank % dtype.itemsize != 0: 
        if rank == 0:
            print(f"Skipping {num_tokens} tokens: size {nbytes} Bytes not divisible by world size and data type.")
        return None, None
        
    elements_per_rank = nbytes_per_rank // dtype.itemsize
    
    def _verify():
        print(f"RANK[{rank}] Verifying for {num_tokens} tokens ({nbytes / (1024*1024):.2f} MB)...")
        if rank == 0:
            ref_tensor = torch.randint(0, 1000, (nbytes // dtype.itemsize,), device="cuda", dtype=dtype)
        else:
            ref_tensor = torch.empty((nbytes // dtype.itemsize,), device="cuda", dtype=dtype)
        
        ref_tensor = torch.from_numpy(
            iris_instance.broadcast_tensor(ref_tensor, source_rank=0)
        ).to(ref_tensor.device)

        local_shard = ref_tensor[rank * elements_per_rank : (rank + 1) * elements_per_rank]

        ag_op.clear_flags()
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

    fn_to_benchmark = lambda: ag_op.forward(local_perf_shard)
    preamble_fn = ag_op.clear_flags
    barrier_fn = iris_instance.barrier

    ag_time_ms = iris.do_bench(
        fn=fn_to_benchmark,
        preamble_fn=preamble_fn,
        barrier_fn=barrier_fn,
        n_warmup=args.n_warmup,
        n_repeat=args.n_repeat,
        return_mode="mean",
    )

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

    iris_instance = iris.iris()
    rank = iris_instance.get_rank()
    torch.cuda.set_device(iris_instance.gpu_id)

    token_counts = [2, 4, 8, 16, 32, 64, 8192, 16384]
    bytes_per_token_per_rank = 12 * 128 * 2 * 2  # num_kv_heads * head_size * (k+v) * bytes_per_element

    results = []

    for num_tokens in token_counts:
        nbytes = num_tokens * bytes_per_token_per_rank * iris_instance.get_num_ranks()

        ag_op = IrisAllGatherLayer(iris_instance, max_buffer_size=nbytes, dtype=dtype)

        if rank == 0:
            print("-" * 80)
        iris_instance.barrier()

        latency_ms, bus_gbps = perf_iris_ag(ag_op, num_tokens, nbytes, dtype, args)

        if rank == 0 and latency_ms is not None:
            nbytes_mb = nbytes / (1024 * 1024)
            results.append({
                "num_tokens": num_tokens,
                "size_mb": nbytes_mb,
                "latency_ms": latency_ms,
                "bus_gbps": bus_gbps
            })
        
        del ag_op
        torch.cuda.empty_cache()

    if rank == 0 and results:
        print("\n" + "="*80)
        print(" " * 25 + "Iris All-Gather Performance Summary")
        print("="*80)

        header = f"{'Token Count (Size)':<25} | {'Latency (ms)':<20} | {'Bus Bandwidth (GB/s)':<25}"
        print(header)
        print("-" * (len(header)))

        for res in results:
            token_str = f"{res['num_tokens']} ({res['size_mb']:.2f} MB)"
            print(f"{token_str:<25} | {res['latency_ms']:<20.3f} | {res['bus_gbps']:<25.2f}")
        print("-" * (len(header)))
