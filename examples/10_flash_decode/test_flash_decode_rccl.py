import argparse
import datetime
import os
import sys
from typing import List, Optional
import json # Added for JSON output

import numpy as np
import torch
import torch.distributed as dist

from sp_flash_decode_layer_rccl import SpGQAFlashDecodeAttentionRCCL

ALL_TESTS = {}
TP_GROUP = None

def perf_func(func, iters, warmup_iters=10):
    for _ in range(warmup_iters):
        func()

    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(iters):
        func()

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / iters
    return None, avg_time_ms

def register_test(name):
    def wrapper(func):
        assert name not in ALL_TESTS
        ALL_TESTS[name] = func
        return func
    return wrapper

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Test and benchmark the RCCL distributed attention implementation.")
    parser.add_argument("--list", action="store_true", help="List available test cases and exit.")
    parser.add_argument("--case", type=str, choices=list(ALL_TESTS.keys()), help="The test case to run: 'correctness' or 'perf'.")
    args = parser.parse_args()
    return args

def help():
    """Prints help message with available tests and run commands."""
    print(f"""
Usage:
------
Use 'torchrun' to launch the script.

torchrun --nproc_per_node=<num_gpus> {os.path.abspath(__file__)} --case <test_name>

Arguments:
----------
--case: The test to run.
  Available choices: {list(ALL_TESTS.keys())}

Examples:
---------
# Run the correctness test on 4 GPUs
torchrun --nproc_per_node=4 {os.path.abspath(__file__)} --case correctness

# Run the performance benchmark on 8 GPUs
torchrun --nproc_per_node=8 {os.path.abspath(__file__)} --case perf
""")

def dist_print(msg, allowed_ranks="all", is_error=False):
    """Prints a message only on the specified ranks."""
    if not dist.is_initialized():
        print(msg)
        return

    rank = dist.get_rank()
    if allowed_ranks == "all" or rank in allowed_ranks:
        prefix = f"[rank{rank}]: "
        stream = sys.stderr if is_error else sys.stdout
        print(f"{prefix}{msg}", file=stream)
        
def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens_per_rank: List[int],
    block_tables: torch.Tensor,
    scale: float,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    """Reference implementation for paged attention."""
    num_seqs = len(query_lens)
    block_tables_cpu = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = value_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens_per_rank[i]
        q = query[start_idx:start_idx + query_len] * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_cpu[i, :num_kv_blocks]

        k = key_cache[block_indices].reshape(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].reshape(-1, num_kv_heads, head_size)[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(torch.ones(query_len, kv_len), diagonal=kv_len - query_len + 1).bool().to(q.device)

        if soft_cap and soft_cap > 0.0:
            attn = soft_cap * torch.tanh(attn / soft_cap)

        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len
        
    return torch.cat(outputs, dim=0)

@register_test("correctness")
def test_correctness(args) -> None:
    """Verifies the output of the RCCL implementation against a reference."""
    kv_lens_per_rank = [1536]
    num_heads, head_size, block_size = 96, 128, 1
    dtype, soft_cap = torch.float16, 0.0
    torch.set_default_device("cuda")

    num_seqs = len(kv_lens_per_rank)
    num_query_heads = num_heads
    num_kv_heads = num_query_heads // 8
    scale = head_size**-0.5
    NUM_BLOCKS_PER_RANK = max(kv_lens_per_rank) + 1
    NUM_BLOCKS = NUM_BLOCKS_PER_RANK * args.num_ranks
    
    dist_print("🔬 Verifying correctness for RCCL implementation...", allowed_ranks=[0])

    ths_op = SpGQAFlashDecodeAttentionRCCL(args.rank, args.num_ranks, num_query_heads, num_kv_heads, head_size, head_size, TP_GROUP, page_size=block_size, scale=scale, soft_cap=soft_cap, max_allowed_batch=num_seqs)

    for i in range(2):
        dist_print(f"\n--- Correctness Test: Iteration {i+1} ---", allowed_ranks=[0])

        seed = 1234 + i
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if args.rank == 0:
            query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype) / 10
            key_value_cache = torch.randn(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype) / 10
        else:
            query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
            key_value_cache = torch.empty(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype)
        
        dist.broadcast(query, src=0, group=TP_GROUP)
        dist.broadcast(key_value_cache, src=0, group=TP_GROUP)

        key_cache = key_value_cache[:, 0].contiguous()
        value_cache = key_value_cache[:, 1].contiguous()
        key_cache_this_rank = key_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) * NUM_BLOCKS_PER_RANK].contiguous()
        value_cache_this_rank = value_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) * NUM_BLOCKS_PER_RANK].contiguous()
        block_tables_this_rank = torch.arange(NUM_BLOCKS_PER_RANK, device="cuda").unsqueeze(0)
        
        gathered_tables_list = [torch.empty_like(block_tables_this_rank) for _ in range(args.num_ranks)]
        dist.all_gather(gathered_tables_list, block_tables_this_rank.contiguous(), group=TP_GROUP)
        block_tables_for_ref = torch.cat([tbl + r * NUM_BLOCKS_PER_RANK for r, tbl in enumerate(gathered_tables_list)], dim=-1)

        global_kv_lens_list = [k * args.num_ranks for k in kv_lens_per_rank]
        kv_lens_tensor = torch.tensor(kv_lens_per_rank, dtype=torch.int32, device=query.device)
        global_kv_lens_tensor = kv_lens_tensor.repeat(args.num_ranks, 1)

        output = ths_op(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)
        ref_output = ref_paged_attn(query.clone(), key_cache, value_cache, [1] * num_seqs, global_kv_lens_list, block_tables_for_ref, scale, soft_cap)
        
        dist.barrier()

        if args.rank == 0:
            header = f"{'Index':<8} | {'Computed':<15} | {'Reference':<15} | {'Abs. Diff':<15}"
            print("\n--- Comparison of First 16 Values (Head 0) ---")
            print(header)
            print("-" * len(header))
            comp_slice, ref_slice = output[0, 0, :16].cpu(), ref_output[0, 0, :16].cpu()
            diff_slice = torch.abs(comp_slice - ref_slice)
            for j in range(len(comp_slice)):
                print(f"{j:<8} | {comp_slice[j]:<15.6f} | {ref_slice[j]:<15.6f} | {diff_slice[j]:<15.6f}")
            print("-" * len(header))

        try:
            torch.testing.assert_close(output, ref_output, atol=1e-3, rtol=1e-2)
            max_diff = torch.max(torch.abs(output - ref_output))
            dist_print(f"✅ TEST PASSED. Max absolute difference: {max_diff:.6f}", allowed_ranks="all")
        except AssertionError as e:
            dist_print(f"❌ TEST FAILED.\n{e}", allowed_ranks="all", is_error=True)

        dist.barrier()

@register_test("perf")
def test_performance(args):
    """Benchmarks the RCCL implementation across various model shapes and sequence lengths."""
    
    # Define model shapes to test
    model_configs = [
    # --- 1. Stress Latency vs. Bandwidth (Varying Batch Size) ---
    # Fused kernel should excel at low batch sizes.
      {"batch_size": 1, "num_q_heads": 96, "head_size": 128},
    {"batch_size": 4, "num_q_heads": 96, "head_size": 128},
    
    # The crossover point where All-Gather may become faster is likely here.
    {"batch_size": 8, "num_q_heads": 96, "head_size": 128},
    {"batch_size": 1, "num_q_heads": 384, "head_size": 128},
    # {"batch_size": 1, "num_q_heads": 768, "head_size": 128},
    # All-Gather's bandwidth advantage should be clear at larger batches.
    # {"batch_size": 32, "num_q_heads": 96, "head_size": 128},

    # --- 2. Stress Message Size vs. Count (Varying Head Dims/Counts) ---
    # Scenario A: More, smaller messages (stresses latency).
    {"batch_size": 1, "num_q_heads": 96, "head_size": 32},
    
    # Scenario B: Fewer, larger messages (stresses bandwidth).
    # {"batch_size": 1, "num_q_heads": 96, "head_size": 1024},
    ]
    # kv_len_configs = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    kv_len_configs = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    dist_print(f"🚀 Benchmarking RCCL Performance on {args.num_ranks} GPUs 🚀", allowed_ranks=[0])
    torch.set_default_device("cuda")

    # ---- NEW: List to hold all data for final JSON dump ----
    all_benchmark_data_for_json = []

    for model_config in model_configs:
        results_summary = []
        
        batch_size = model_config["batch_size"]
        num_query_heads = model_config["num_q_heads"]
        head_size = model_config["head_size"]
        
        # Derived parameters
        num_kv_heads = num_query_heads // 8
        scale = head_size**-0.5
        dtype = torch.float16
        soft_cap = 0.0
        block_size = 1
        bytes_per_token = num_kv_heads * head_size * 2 * 2
        
        dist_print(f"\n\n{'#'*80}", allowed_ranks=[0])
        dist_print(f"### New Benchmark Suite ###", allowed_ranks=[0])
        dist_print(f"# Config: Batch={batch_size}, Q-Heads={num_query_heads}, Head-Dim={head_size}", allowed_ranks=[0])
        dist_print(f"{'#'*80}", allowed_ranks=[0])

        warmup_iters = 5
        benchmark_iters = 20
        
        for kv_len_per_rank in kv_len_configs:
            kv_cache_size_bytes = kv_len_per_rank * bytes_per_token * batch_size
            kv_cache_size_gb = kv_cache_size_bytes / (1024 * 1024 * 1024)
            
            dist_print(f"\n----- Benchmarking @ KV Cache/GPU: {kv_cache_size_gb:.2f} GB (Len: {kv_len_per_rank}) -----", allowed_ranks=[0])

            NUM_BLOCKS_PER_RANK = kv_len_per_rank + 1
            query = torch.randn(batch_size, num_query_heads, head_size, dtype=dtype)
            key_cache_this_rank = torch.randn(NUM_BLOCKS_PER_RANK, block_size, num_kv_heads, head_size, dtype=dtype)
            value_cache_this_rank = torch.randn(NUM_BLOCKS_PER_RANK, block_size, num_kv_heads, head_size, dtype=dtype)
            block_tables_this_rank = torch.randint(0, NUM_BLOCKS_PER_RANK, (batch_size, NUM_BLOCKS_PER_RANK), dtype=torch.int32)
            kv_lens_tensor = torch.tensor([kv_len_per_rank] * batch_size, dtype=torch.int32)
            global_kv_lens_tensor = kv_lens_tensor.unsqueeze(0).repeat(args.num_ranks, 1)

            ths_op = SpGQAFlashDecodeAttentionRCCL(
                args.rank, args.num_ranks, num_query_heads, num_kv_heads, head_size, head_size, 
                TP_GROUP, page_size=block_size, scale=scale, soft_cap=soft_cap, 
                max_allowed_batch=batch_size
            )

            def func_to_benchmark():
                return ths_op(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)
            
            _, time_ms = perf_func(func=func_to_benchmark, iters=benchmark_iters, warmup_iters=warmup_iters)
            dist_print(f"         ✅ Result captured: {time_ms:.3f} ms", allowed_ranks=[0])
            
            if args.rank == 0:
                results_summary.append({
                    "kv_len": kv_len_per_rank,
                    "cache_gb": kv_cache_size_gb,
                    "time_ms": time_ms,
                })
            dist.barrier()

        if args.rank == 0:
            # --- Print final summary to console ---
            print("\n\n--- Final RCCL Performance Summary ---")
            print(f"--- Config: Batch={batch_size}, Q-Heads={num_query_heads}, Head-Dim={head_size} ---")
            header = f"{'KV Length/GPU (GB)':<28} | #GPU={args.num_ranks:<4} | {'Average Time (ms)':<25}"
            separator = "-" * len(header)
            print(separator)
            print(header)
            print(separator)
            for r in results_summary:
                kv_len_str = f"{r['kv_len']} ({r['cache_gb']:.2f} GB)"
                print(f"{kv_len_str:<28} | {'':<9} | {r['time_ms']:<25.3f}")
            print(separator)

            # --- Append final summary to a log file ---
            filename = "rccl_perf_results_log.txt"
            print(f"\nAppending detailed results to {filename}...")
            
            try:
                with open(filename, 'a') as f:
                    f.write(f"\n\n{'='*80}\n")
                    f.write(f"--- New Benchmark Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                    f.write(f"{'='*80}\n\n")

                    f.write("--- Configuration ---\n")
                    f.write(f"Number of GPUs: {args.num_ranks}\n")
                    f.write(f"Batch Size: {batch_size}\n")
                    f.write(f"Number of Query Heads: {num_query_heads}\n")
                    f.write(f"Head Dimension: {head_size}\n\n")

                    f.write("--- Final RCCL Performance Summary ---\n")
                    f.write(separator + '\n')
                    f.write(header + '\n')
                    f.write(separator + '\n')

                    for r in results_summary:
                        kv_len_str = f"{r['kv_len']} ({r['cache_gb']:.2f} GB)"
                        row_str = f"{kv_len_str:<28} | {'':<9} | {r['time_ms']:<25.3f}"
                        f.write(row_str + '\n')

                    f.write(separator + '\n')
                
                print(f"Results successfully appended to {filename}.")
            except IOError as e:
                print(f"Error: Could not write to file {filename}. \n{e}", is_error=True)

            # ---- NEW: Aggregate data for this config for the JSON file ----
            benchmark_suite_data = {
                "config": {
                    "batch_size": batch_size,
                    "num_q_heads": num_query_heads,
                    "head_dim": head_size,
                    "num_gpus": args.num_ranks
                },
                "results": results_summary
            }
            all_benchmark_data_for_json.append(benchmark_suite_data)

    # ---- NEW: Write the final JSON file after all configs are done ----
    if args.rank == 0 and all_benchmark_data_for_json:
        json_filename = "rccl_perf_results.json"
        dist_print(f"\nWriting structured results to {json_filename}...", allowed_ranks=[0])
        try:
            with open(json_filename, 'w') as f:
                json.dump(all_benchmark_data_for_json, f, indent=4)
            dist_print(f"Successfully wrote to {json_filename}", allowed_ranks=[0])
        except IOError as e:
            dist_print(f"Error writing to {json_filename}: {e}", allowed_ranks=[0], is_error=True)


if __name__ == "__main__":
    args = get_args()
    if args.list:
        help()
        sys.exit(0)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        args.rank = int(os.environ["RANK"])
        args.num_ranks = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        dist_print("Warning: Not running in a distributed environment. Running on a single GPU.", is_error=True)
        args.rank, args.num_ranks = 0, 1
    
    TP_GROUP = dist.new_group(ranks=range(args.num_ranks)) if dist.is_initialized() else None
    
    if not args.case:
        if args.rank == 0:
            print("Error: --case is a required argument ('correctness' or 'perf')."); help()
        sys.exit(1)
    
    func = ALL_TESTS[args.case]
    func(args)
    
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()