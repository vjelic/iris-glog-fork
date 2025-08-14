import argparse
import os
import sys
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import iris

from fd_layer_iris_fused_full import SpGQAFlashDecodeAttentionIrisFusedFull
from utils.utils import dist_print, print_correctness_report, PerfLogger

# ==============================================================================
# Test Configurations
# ==============================================================================

class TestConfigs:
    """Centralized configurations for performance and correctness tests."""
    CORRECTNESS = {
        "kv_len": 16384,
        "num_heads": 96,
        "head_size": 128,
        "block_size": 1,
        "dtype": torch.float16,
        "soft_cap": 0,
        "num_seqs": 1,
    }
    PERF = {
        "kv_len_configs": [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
        "num_heads": 96,
        "head_size": 128,
        "block_size": 1,
        "dtype": torch.float16,
        "soft_cap": 0,
        "num_seqs": 1,
        "n_warmup": 100,
        "n_repeat": 1000,
    }

# ==============================================================================
# Test Registration & Reference Implementation
# ==============================================================================

ALL_TESTS = {}

def register_test(name):
    def wrapper(func):
        assert name not in ALL_TESTS, f"Test '{name}' already registered."
        ALL_TESTS[name] = func
        return func
    return wrapper

def ref_paged_attn(
    query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
    query_lens: List[int], kv_lens_per_rank: List[int], block_tables: torch.Tensor,
    scale: float, soft_cap: Optional[float] = None
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables_cpu = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len, kv_len = query_lens[i], kv_lens_per_rank[i]
        q = query[start_idx:start_idx + query_len]
        q *= scale
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_cpu[i, :num_kv_blocks]
        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        if q.shape[1] != k.shape[1]:
            gqa_ratio = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, gqa_ratio, dim=1)
            v = torch.repeat_interleave(v, gqa_ratio, dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len, device=query.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if soft_cap is not None and soft_cap > 0.0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len
    return torch.cat(outputs, dim=0)

# ==============================================================================
# Data Preparation Functions
# ==============================================================================

def prepare_correctness_data(cfg, args, num_query_heads, num_kv_heads, NUM_BLOCKS):
    """Prepares and distributes data needed for the correctness test."""
    if args.rank == 0:
        query = torch.randn(cfg['num_seqs'], num_query_heads, cfg['head_size'], dtype=cfg['dtype']) / 10
        key_value_cache = torch.randn(NUM_BLOCKS, 2, cfg['block_size'], num_kv_heads, cfg['head_size'], dtype=cfg['dtype']) / 10
    else:
        query = torch.empty(cfg['num_seqs'], num_query_heads, cfg['head_size'], dtype=cfg['dtype'])
        key_value_cache = torch.empty(NUM_BLOCKS, 2, cfg['block_size'], num_kv_heads, cfg['head_size'], dtype=cfg['dtype'])
    
    query = torch.from_numpy(args.iris_instance.broadcast_tensor(query.cpu().numpy(), source_rank=0)).to(query.device)
    key_value_cache = torch.from_numpy(args.iris_instance.broadcast_tensor(key_value_cache.cpu().numpy(), source_rank=0)).to(key_value_cache.device)
    
    return {"query": query, "key_value_cache": key_value_cache}

def prepare_perf_data(cfg, num_query_heads, num_kv_heads, NUM_BLOCKS_PER_RANK):
    """Prepares local data needed for the performance test."""
    query = torch.randn(cfg['num_seqs'], num_query_heads, cfg['head_size'], dtype=cfg['dtype'])
    key_cache_this_rank = torch.randn(NUM_BLOCKS_PER_RANK, cfg['block_size'], num_kv_heads, cfg['head_size'], dtype=cfg['dtype'])
    value_cache_this_rank = torch.randn(NUM_BLOCKS_PER_RANK, cfg['block_size'], num_kv_heads, cfg['head_size'], dtype=cfg['dtype'])
    block_tables_this_rank = torch.randint(0, NUM_BLOCKS_PER_RANK, (cfg['num_seqs'], NUM_BLOCKS_PER_RANK), dtype=torch.int32)
    
    return {
        "query": query,
        "key_cache_this_rank": key_cache_this_rank,
        "value_cache_this_rank": value_cache_this_rank,
        "block_tables_this_rank": block_tables_this_rank
    }

# ==============================================================================
# Test Cases
# ==============================================================================

@register_test("correctness")
def test_correctness(args) -> None:
    """
    Tests the correctness of the FUSED_FULL implementation against the Torch reference.
    """
    torch.manual_seed(42)
    cfg = TestConfigs.CORRECTNESS
    torch.set_default_device("cuda")

    # Keep parameter setup within the test function
    num_heads = cfg['num_heads']
    head_size = cfg['head_size']
    num_seqs = cfg['num_seqs']
    num_query_heads = num_heads
    num_kv_heads = num_query_heads // 8
    scale = head_size**-0.5
    NUM_BLOCKS_PER_RANK = cfg['kv_len'] + 1
    NUM_BLOCKS = NUM_BLOCKS_PER_RANK * args.num_ranks
    
    # Use the data prep function to get tensors
    tensor_data = prepare_correctness_data(cfg, args, num_query_heads, num_kv_heads, NUM_BLOCKS)
    query = tensor_data['query']
    key_value_cache = tensor_data['key_value_cache']

    # Keep operator setup and other logic here
    key_cache = key_value_cache[:, 0, :, :, :].contiguous()
    value_cache = key_value_cache[:, 1, :, :, :].contiguous()
    key_cache_this_rank = key_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) * NUM_BLOCKS_PER_RANK].contiguous()
    value_cache_this_rank = value_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) * NUM_BLOCKS_PER_RANK].contiguous()

    block_tables_this_rank = torch.arange(NUM_BLOCKS_PER_RANK, dtype=torch.int32).unsqueeze(0)
    all_block_tables_numpy = iris._mpi_helpers.mpi_allgather(block_tables_this_rank.cpu().numpy())
    block_tables = torch.from_numpy(all_block_tables_numpy).view(args.num_ranks, num_seqs, -1)
    ref_block_tables = torch.cat([block_tables[i] + i * NUM_BLOCKS_PER_RANK for i in range(args.num_ranks)], dim=-1).squeeze(0)

    common_params = {
        "num_q_heads": num_query_heads, "num_kv_heads": num_kv_heads, "q_head_dim": head_size,
        "v_head_dim": head_size, "page_size": cfg['block_size'], "scale": scale, "soft_cap": cfg['soft_cap'],
        "max_allowed_batch": num_seqs
    }
    ths_op = SpGQAFlashDecodeAttentionIrisFusedFull(
        args.iris_instance, args.rank, args.rank // args.local_num_ranks,
        args.num_ranks, args.num_ranks // args.local_num_ranks, **common_params
    )
    
    args.iris_instance.barrier()
    if hasattr(ths_op, 'clear_flags'):
        ths_op.clear_flags()
    args.iris_instance.barrier()
    
    kv_lens_per_rank = [cfg['kv_len']]
    global_kv_lens = [kv_lens_per_rank[0] * args.num_ranks]
    kv_lens_tensor = torch.tensor(kv_lens_per_rank, dtype=torch.int32, device=query.device)
    global_kv_lens_tensor = torch.cat([kv_lens_tensor.view(1, -1) for _ in range(args.num_ranks)], dim=0)

    output = ths_op(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)
    torch.cuda.synchronize()
    
    ref_output = ref_paged_attn(
        query=query.clone(), key_cache=key_cache, value_cache=value_cache,
        query_lens=[1] * num_seqs, kv_lens_per_rank=global_kv_lens,
        block_tables=ref_block_tables.unsqueeze(0), scale=scale, soft_cap=cfg['soft_cap']
    )
    args.iris_instance.barrier()

    error = None
    try:
        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
    except AssertionError as e:
        error = e

    print_correctness_report(args.rank, output, ref_output, error)
    args.iris_instance.barrier()


@register_test("perf")
def test_performance(args, logger: PerfLogger):
    """Benchmarks the FUSED_FULL implementation across various large sequence lengths."""
    torch.manual_seed(42)
    logger.log_header("Performance Benchmark")
    
    cfg = TestConfigs.PERF
    torch.set_default_device("cuda")
    
    full_results = []
    impl_name = "FUSED IRIS"
    
    for kv_len_per_rank in cfg['kv_len_configs']:
        logger.log(f"\n----- Benchmarking @ KV Length per Rank: {kv_len_per_rank} -----")
        
        # Keep parameter setup within the test loop
        num_query_heads, num_kv_heads = cfg['num_heads'], cfg['num_heads'] // 8
        scale = cfg['head_size']**-0.5
        NUM_BLOCKS_PER_RANK = kv_len_per_rank + 1
        
        # Use the data prep function to get local tensors
        tensor_data = prepare_perf_data(cfg, num_query_heads, num_kv_heads, NUM_BLOCKS_PER_RANK)
        query = tensor_data['query']
        key_cache_this_rank = tensor_data['key_cache_this_rank']
        value_cache_this_rank = tensor_data['value_cache_this_rank']
        block_tables_this_rank = tensor_data['block_tables_this_rank']

        # Keep operator setup and other logic here
        kv_lens_tensor = torch.tensor([kv_len_per_rank], dtype=torch.int32)
        global_kv_lens_tensor = torch.cat([kv_lens_tensor.view(1, -1) for _ in range(args.num_ranks)], dim=0)

        common_params = {
            "num_q_heads": num_query_heads, "num_kv_heads": num_kv_heads, "q_head_dim": cfg['head_size'],
            "v_head_dim": cfg['head_size'], "page_size": cfg['block_size'], "scale": scale, 
            "soft_cap": cfg['soft_cap'], "max_allowed_batch": cfg['num_seqs']
        }
        
        node_id = args.rank // args.local_num_ranks
        num_nodes = args.num_ranks // args.local_num_ranks
        ths_op = SpGQAFlashDecodeAttentionIrisFusedFull(
            args.iris_instance, args.rank, node_id, args.num_ranks, num_nodes, **common_params
        )

        def func_to_benchmark():
            return ths_op(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)

        logger.log(f"--> Benchmarking {impl_name}...")
        time_ms = iris.do_bench(
            fn=func_to_benchmark, preamble_fn=getattr(ths_op, 'clear_flags', None),
            barrier_fn=args.iris_instance.barrier, n_warmup=cfg['n_warmup'], n_repeat=cfg['n_repeat'], return_mode="mean"
        )
        args.iris_instance.barrier()
        
        if args.rank == 0:
            logger.log(f"    Done. Average time: {time_ms:.3f} ms")
            full_results.append({"kv_len": kv_len_per_rank, impl_name: time_ms})

    logger.log_perf_summary(full_results, args.num_ranks)

# ==============================================================================
# Main Execution Block
# ==============================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Run correctness or performance tests for FUSED_FULL attention.")
    parser.add_argument("--list", action="store_true", help="List available tests.")
    parser.add_argument("--case", type=str, choices=list(ALL_TESTS.keys()), help="Specify which test case to run.")
    return parser.parse_args()

def help():
    print(f"\nAvailable choices: {list(ALL_TESTS.keys())}")
    print(f"Run a test with: python {os.path.abspath(__file__)} --case [test_name]\n")

if __name__ == "__main__":
    try:
        _iris = iris.iris()
    except Exception as e:
        print(f"FATAL: Failed to initialize iris: {e}", file=sys.stderr)
        sys.exit(1)

    args = get_args()
    args.rank = _iris.get_rank()
    args.num_ranks = _iris.get_num_ranks()
    args.local_num_ranks = _iris.get_num_ranks()
    args.iris_instance = _iris
    
    if args.list:
        if args.rank == 0: help()
        sys.exit(0)
    
    if not args.case:
        dist_print("No test case specified. Defaulting to 'correctness'.", args.rank, is_error=True)
        if args.rank == 0: help()
        args.case = "correctness"

    if args.case in ALL_TESTS:
        test_func = ALL_TESTS[args.case]
        try:
            if args.case == "correctness":
                test_func(args)
            else:
                logger = PerfLogger(args.rank)
                test_func(args, logger)
        except Exception as e:
            dist_print(f"An uncaught error occurred during test '{args.case}': {e}", args.rank, is_error=True)
            import traceback
            traceback.print_exc()
    else:
        dist_print(f"Case '{args.case}' not found.", args.rank, is_error=True)
        if args.rank == 0: help()

    _iris.barrier()
    dist_print("Run complete.", args.rank)