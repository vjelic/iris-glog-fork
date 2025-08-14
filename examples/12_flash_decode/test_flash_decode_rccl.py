import argparse
import datetime
import os
import sys
from typing import List, Optional, Dict, Any
import json

import numpy as np
import torch
import torch.distributed as dist

from fd_layer_rccl import SpGQAFlashDecodeAttentionRCCL

# ==============================================================================
# Test Configurations
# ==============================================================================

class TestConfigs:
    """Centralized configurations for correctness and performance tests."""
    CORRECTNESS = {
        "kv_len": 16384,
        "num_heads": 96,
        "head_size": 128,
        "block_size": 1,
        "dtype": torch.float16,
        "soft_cap": 0.0,
        "num_seqs": 1,
    }
    PERF = {
        "model_configs": [
            # {"batch_size": 1, "num_q_heads": 96, "head_size": 64},
            {"batch_size": 1, "num_q_heads": 96, "head_size": 128},
            # {"batch_size": 4, "num_q_heads": 96, "head_size": 128},
            # {"batch_size": 8, "num_q_heads": 96, "head_size": 128},
            # {"batch_size": 1, "num_q_heads": 96, "head_size": 32},
        ],
        "kv_len_configs": [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
        "warmup_iters": 5,
        "benchmark_iters": 20,
    }

# ==============================================================================
# Test Registration & Setup
# ==============================================================================

ALL_TESTS = {}
TP_GROUP = None

def register_test(name):
    def wrapper(func):
        assert name not in ALL_TESTS, f"Test '{name}' already registered."
        ALL_TESTS[name] = func
        return func
    return wrapper

def get_args():
    parser = argparse.ArgumentParser(description="Test and benchmark the RCCL distributed attention implementation.")
    parser.add_argument("--list", action="store_true", help="List available test cases and exit.")
    parser.add_argument("--case", type=str, choices=list(ALL_TESTS.keys()), help="The test case to run: 'correctness' or 'perf'.")
    args = parser.parse_args()
    return args

def help():
    print(f"""
Usage:
------
Use 'torchrun' to launch the script.

torchrun --nproc_per_node=<num_gpus> {os.path.abspath(__file__)} --case <test_name>

Arguments:
----------
--case: The test to run.
  Available choices: {list(ALL_TESTS.keys())}
""")

def dist_print(msg, allowed_ranks="all", is_error=False):
    if not dist.is_initialized() or dist.get_rank() == 0:
        if allowed_ranks == "all" or (dist.is_initialized() and dist.get_rank() in allowed_ranks):
            prefix = f"[rank{dist.get_rank() if dist.is_initialized() else 0}]: " if allowed_ranks == "all" else ""
            stream = sys.stderr if is_error else sys.stdout
            print(f"{prefix}{msg}", file=stream)

# ==============================================================================
# Data Preparation Functions
# ==============================================================================

def prepare_correctness_data(cfg, args, num_query_heads, num_kv_heads, NUM_BLOCKS):
    """Prepares and distributes data for the correctness test."""
    if args.rank == 0:
        query = torch.randn(cfg['num_seqs'], num_query_heads, cfg['head_size'], dtype=cfg['dtype']) / 10
        key_value_cache = torch.randn(NUM_BLOCKS, 2, cfg['block_size'], num_kv_heads, cfg['head_size'], dtype=cfg['dtype']) / 10
    else:
        query = torch.empty(cfg['num_seqs'], num_query_heads, cfg['head_size'], dtype=cfg['dtype'])
        key_value_cache = torch.empty(NUM_BLOCKS, 2, cfg['block_size'], num_kv_heads, cfg['head_size'], dtype=cfg['dtype'])

    dist.broadcast(query, src=0, group=TP_GROUP)
    dist.broadcast(key_value_cache, src=0, group=TP_GROUP)
    return {"query": query, "key_value_cache": key_value_cache}

def prepare_perf_data(model_cfg, kv_len, num_kv_heads, dtype):
    """Prepares local data for the performance test."""
    NUM_BLOCKS_PER_RANK = kv_len + 1
    query = torch.randn(model_cfg['batch_size'], model_cfg['num_q_heads'], model_cfg['head_size'], dtype=dtype)
    key_cache_this_rank = torch.randn(NUM_BLOCKS_PER_RANK, 1, num_kv_heads, model_cfg['head_size'], dtype=dtype)
    value_cache_this_rank = torch.randn(NUM_BLOCKS_PER_RANK, 1, num_kv_heads, model_cfg['head_size'], dtype=dtype)
    block_tables_this_rank = torch.randint(0, NUM_BLOCKS_PER_RANK, (model_cfg['batch_size'], NUM_BLOCKS_PER_RANK), dtype=torch.int32)
    return {
        "query": query,
        "key_cache_this_rank": key_cache_this_rank,
        "value_cache_this_rank": value_cache_this_rank,
        "block_tables_this_rank": block_tables_this_rank,
    }

# ==============================================================================
# Performance Utilities
# ==============================================================================

def perf_func(func, iters, warmup_iters=10):
    for _ in range(warmup_iters):
        func()
    if dist.is_initialized(): dist.barrier()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        func()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    return elapsed_time_ms / iters

class PerfReporter:
    """Handles collecting, printing, and saving performance results."""
    def __init__(self, rank):
        self.rank = rank
        self.all_results = []

    def add_suite(self, config, results_summary):
        if self.rank == 0:
            suite_data = {"config": config, "results": results_summary}
            self.all_results.append(suite_data)
            self._print_summary_table(config, results_summary)

    def _print_summary_table(self, config, summary):
        print("\n\n--- Final RCCL Performance Summary ---")
        print(f"--- Config: Batch={config['batch_size']}, Q-Heads={config['num_q_heads']}, Head-Dim={config['head_dim']} ---")
        header = f"{'KV Length/GPU (GB)':<28} | #GPU={config['num_gpus']:<4} | {'Average Time (ms)':<25}"
        separator = "-" * len(header)
        print(separator)
        print(header)
        print(separator)
        for r in summary:
            kv_len_str = f"{r['kv_len']} ({r['cache_gb']:.2f} GB)"
            print(f"{kv_len_str:<28} | {'':<9} | {r['time_ms']:<25.3f}")
        print(separator)

    def write_log_files(self, txt_filename="rccl_perf_results_log.txt", json_filename="rccl_perf_results.json"):
        if self.rank != 0: return
        self._write_txt_log(txt_filename)
        self._write_json_log(json_filename)

    def _write_txt_log(self, filename):
        dist_print(f"\nAppending detailed results to {filename}...")
        try:
            with open(filename, 'a') as f:
                f.write(f"\n\n{'='*80}\n--- New Benchmark Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n{'='*80}\n")
                for suite in self.all_results:
                    config, summary = suite["config"], suite["results"]
                    header = f"{'KV Length/GPU (GB)':<28} | #GPU={config['num_gpus']:<4} | {'Average Time (ms)':<25}"
                    separator = "-" * len(header)
                    f.write(f"\n--- Configuration ---\n"
                            f"Number of GPUs: {config['num_gpus']}\n"
                            f"Batch Size: {config['batch_size']}\n"
                            f"Number of Query Heads: {config['num_q_heads']}\n"
                            f"Head Dimension: {config['head_dim']}\n\n")
                    f.write("--- Final RCCL Performance Summary ---\n")
                    f.write(separator + '\n' + header + '\n' + separator + '\n')
                    for r in summary:
                        kv_len_str = f"{r['kv_len']} ({r['cache_gb']:.2f} GB)"
                        f.write(f"{kv_len_str:<28} | {'':<9} | {r['time_ms']:<25.3f}\n")
                    f.write(separator + '\n')
            dist_print(f"Results successfully appended to {filename}.")
        except IOError as e:
            dist_print(f"Error: Could not write to file {filename}. \n{e}", is_error=True)

    def _write_json_log(self, filename):
        dist_print(f"\nWriting structured results to {filename}...")
        try:
            with open(filename, 'w') as f:
                json.dump(self.all_results, f, indent=4)
            dist_print(f"Successfully wrote to {filename}")
        except IOError as e:
            dist_print(f"Error writing to {filename}: {e}", is_error=True)

# ==============================================================================
# Reference Implementation
# ==============================================================================

def ref_paged_attn(query, key_cache, value_cache, query_lens, kv_lens_per_rank, block_tables, scale, soft_cap=None):
    num_seqs = len(query_lens)
    block_tables_cpu, (_, block_size, num_kv_heads, head_size) = block_tables.cpu().numpy(), value_cache.shape
    outputs, start_idx = [], 0
    for i in range(num_seqs):
        query_len, kv_len = query_lens[i], kv_lens_per_rank[i]
        q = query[start_idx:start_idx + query_len] * scale
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_cpu[i, :num_kv_blocks]
        k = key_cache[block_indices].reshape(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].reshape(-1, num_kv_heads, head_size)[:kv_len]
        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(torch.ones(query_len, kv_len, device=q.device), diagonal=kv_len - query_len + 1).bool()
        if soft_cap and soft_cap > 0.0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len
    return torch.cat(outputs, dim=0)

# ==============================================================================
# Test Cases
# ==============================================================================

@register_test("correctness")
def test_correctness(args) -> None:
    cfg = TestConfigs.CORRECTNESS
    torch.set_default_device("cuda")
    dist_print("🔬 Verifying correctness for RCCL implementation...", allowed_ranks=[0])

    num_query_heads, head_size = cfg['num_heads'], cfg['head_size']
    num_kv_heads = num_query_heads // 8
    scale = head_size**-0.5
    NUM_BLOCKS_PER_RANK = cfg['kv_len'] + 1
    NUM_BLOCKS = NUM_BLOCKS_PER_RANK * args.num_ranks
    
    ths_op = SpGQAFlashDecodeAttentionRCCL(args.rank, args.num_ranks, num_query_heads, num_kv_heads, head_size, head_size, TP_GROUP, page_size=cfg['block_size'], scale=scale, soft_cap=cfg['soft_cap'], max_allowed_batch=cfg['num_seqs'])

    for i in range(2):
        dist_print(f"\n--- Correctness Test: Iteration {i+1} ---", allowed_ranks=[0])
        torch.manual_seed(1234 + i), torch.cuda.manual_seed_all(1234 + i)
        
        tensor_data = prepare_correctness_data(cfg, args, num_query_heads, num_kv_heads, NUM_BLOCKS)
        query, key_value_cache = tensor_data['query'], tensor_data['key_value_cache']

        key_cache, value_cache = key_value_cache[:, 0].contiguous(), key_value_cache[:, 1].contiguous()
        key_cache_this_rank = key_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) * NUM_BLOCKS_PER_RANK].contiguous()
        value_cache_this_rank = value_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) * NUM_BLOCKS_PER_RANK].contiguous()
        block_tables_this_rank = torch.arange(NUM_BLOCKS_PER_RANK, device="cuda").unsqueeze(0)
        
        gathered_tables_list = [torch.empty_like(block_tables_this_rank) for _ in range(args.num_ranks)]
        dist.all_gather(gathered_tables_list, block_tables_this_rank.contiguous(), group=TP_GROUP)
        block_tables_for_ref = torch.cat([tbl + r * NUM_BLOCKS_PER_RANK for r, tbl in enumerate(gathered_tables_list)], dim=-1)

        kv_lens_tensor = torch.tensor([cfg['kv_len']], dtype=torch.int32, device=query.device)
        global_kv_lens_tensor = kv_lens_tensor.repeat(args.num_ranks, 1)

        output = ths_op(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)
        ref_output = ref_paged_attn(query.clone(), key_cache, value_cache, [1] * cfg['num_seqs'], [cfg['kv_len'] * args.num_ranks], block_tables_for_ref, scale, cfg['soft_cap'])
        dist.barrier()

        if args.rank == 0:
            header = f"{'Index':<8} | {'Computed':<15} | {'Reference':<15} | {'Abs. Diff':<15}"
            print("\n--- Comparison of First 16 Values (Head 0) ---"), print(header), print("-" * len(header))
            comp_slice, ref_slice = output[0, 0, :16].cpu(), ref_output[0, 0, :16].cpu()
            diff_slice = torch.abs(comp_slice - ref_slice)
            for j in range(len(comp_slice)):
                print(f"{j:<8} | {comp_slice[j]:<15.6f} | {ref_slice[j]:<15.6f} | {diff_slice[j]:<15.6f}")
            print("-" * len(header))
        try:
            torch.testing.assert_close(output, ref_output, atol=1e-3, rtol=1e-2)
            dist_print(f"✅ TEST PASSED. Max absolute difference: {torch.max(torch.abs(output - ref_output)):.6f}", allowed_ranks="all")
        except AssertionError as e:
            dist_print(f"❌ TEST FAILED.\n{e}", allowed_ranks="all", is_error=True)
        dist.barrier()

@register_test("perf")
def test_performance(args):
    cfg = TestConfigs.PERF
    dist_print(f"🚀 Benchmarking RCCL Performance on {args.num_ranks} GPUs 🚀", allowed_ranks=[0])
    torch.set_default_device("cuda")
    torch.manual_seed(42)

    reporter = PerfReporter(args.rank)

    for model_config in cfg['model_configs']:
        results_summary = []
        batch_size, num_query_heads, head_size = model_config["batch_size"], model_config["num_q_heads"], model_config["head_size"]
        num_kv_heads, scale, dtype, soft_cap = num_query_heads // 8, head_size**-0.5, torch.float16, 0.0
        bytes_per_token = num_kv_heads * head_size * 2 * 2

        dist_print(f"\n\n{'#'*80}\n### New Benchmark Suite: Batch={batch_size}, Q-Heads={num_query_heads}, Head-Dim={head_size} ###\n{'#'*80}", allowed_ranks=[0])
        
        for kv_len_per_rank in cfg['kv_len_configs']:
            kv_cache_size_gb = (kv_len_per_rank * bytes_per_token * batch_size) / (1024**3)
            dist_print(f"\n----- Benchmarking @ KV Cache/GPU: {kv_cache_size_gb:.2f} GB (Len: {kv_len_per_rank}) -----", allowed_ranks=[0])
            
            tensor_data = prepare_perf_data(model_config, kv_len_per_rank, num_kv_heads, dtype)
            
            kv_lens_tensor = torch.tensor([kv_len_per_rank] * batch_size, dtype=torch.int32)
            global_kv_lens_tensor = kv_lens_tensor.unsqueeze(0).repeat(args.num_ranks, 1)

            ths_op = SpGQAFlashDecodeAttentionRCCL(args.rank, args.num_ranks, num_query_heads, num_kv_heads, head_size, head_size, TP_GROUP, page_size=1, scale=scale, soft_cap=soft_cap, max_allowed_batch=batch_size)
            
            def func_to_benchmark():
                return ths_op(tensor_data['query'], tensor_data['key_cache_this_rank'], tensor_data['value_cache_this_rank'], global_kv_lens_tensor, tensor_data['block_tables_this_rank'])
            
            time_ms = perf_func(func=func_to_benchmark, iters=cfg['benchmark_iters'], warmup_iters=cfg['warmup_iters'])
            dist_print(f"✅ Result captured: {time_ms:.3f} ms", allowed_ranks=[0])
            
            if args.rank == 0:
                results_summary.append({"kv_len": kv_len_per_rank, "cache_gb": kv_cache_size_gb, "time_ms": time_ms})
            dist.barrier()
        
        suite_config = {"batch_size": batch_size, "num_q_heads": num_query_heads, "head_dim": head_size, "num_gpus": args.num_ranks}
        reporter.add_suite(suite_config, results_summary)

    reporter.write_log_files()

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    args = get_args()
    if args.list:
        help()
        sys.exit(0)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        args.rank = int(os.environ["RANK"])
        args.num_ranks = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        dist_print("Warning: Not running in a distributed environment. Running on a single GPU.", is_error=True)
        args.rank, args.num_ranks = 0, 1
    
    TP_GROUP = dist.new_group(ranks=range(args.num_ranks)) if dist.is_initialized() else None
    
    if not args.case:
        if args.rank == 0: print("Error: --case is a required argument ('correctness' or 'perf')."), help()
        sys.exit(1)
    
    try:
        func = ALL_TESTS[args.case]
        func(args)
    except Exception as e:
        dist_print(f"An uncaught error occurred during test '{args.case}': {e}", allowed_ranks="all", is_error=True)
        import traceback
        traceback.print_exc()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()