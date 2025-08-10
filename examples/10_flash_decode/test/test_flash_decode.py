import argparse
import datetime
import os
import sys
from typing import List, Optional
import json

import numpy as np
import torch
import iris
import time

from utils.utils import dist_print
from layers.fd_layer_iris_ag_kernel import SpGQAFlashDecodeAttentionIrisAG
from layers.fd_layer_mpi import SpGQAFlashDecodeAttentionMPI
from layers.fd_layer_iris_no_wait import SpGQAFlashDecodeAttentionIrisAGNoWait
from layers.fd_layer_iris_fused_nonoptimized import SpGQAFlashDecodeAttentionIrisFused
from layers.fd_layer_iris_fused_full import SpGQAFlashDecodeAttentionIrisFusedFull
from layers.fd_layer_iris_fused_persistent import SpGQAFlashDecodeAttentionIrisFusedPersistent

CORRECTNESS_IMPL_TO_TEST = "FUSED_FULL"
PERF_IMPLS_TO_TEST = ["FUSED_FULL", "FUSED_PERSISTENT"]

ALL_TESTS = {}

TP_GROUP = None

def register_test(name):

    def wrapper(func):
        assert name not in ALL_TESTS
        ALL_TESTS[name] = func

    return wrapper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", type=str, choices=list(ALL_TESTS.keys()))
    parser.add_argument("--shape_id", type=str, default="")
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()
    return args


def help():
    print(f"""
Available choices: {list(ALL_TESTS.keys())}.
run: python {os.path.abspath(__file__)} --case XXX
""")


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens_per_rank: List[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    """
    Reference Torch Implementation For Flash Decode
    """
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens_per_rank[i]
        q = query[start_idx:start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = torch.triu(empty_mask,
                                             diagonal=kv_len - (query_len + sliding_window) + 1).bool().logical_not()
            mask |= sliding_window_mask
        if soft_cap > 0.0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def get_op_instance(impl_name, args, common_params):
    if impl_name == "FUSED_PERSISTENT":
        return SpGQAFlashDecodeAttentionIrisFusedPersistent(args.iris_instance, args.rank, args.rank // args.local_num_ranks, args.num_ranks, args.num_ranks // args.local_num_ranks, **common_params)
    if impl_name == "FUSED":
        return SpGQAFlashDecodeAttentionIrisFused(args.iris_instance, args.rank, args.rank // args.local_num_ranks, args.num_ranks, args.num_ranks // args.local_num_ranks, **common_params)
    elif impl_name == "FUSED_FULL":
        return SpGQAFlashDecodeAttentionIrisFusedFull(args.iris_instance, args.rank, args.rank // args.local_num_ranks, args.num_ranks, args.num_ranks // args.local_num_ranks, **common_params)
    elif impl_name == "NOWAIT":
        return SpGQAFlashDecodeAttentionIrisAGNoWait(args.iris_instance, args.rank, args.rank // args.local_num_ranks, args.num_ranks, args.num_ranks // args.local_num_ranks, **common_params)
    elif impl_name == "STANDARD":
        return SpGQAFlashDecodeAttentionIrisAG(args.iris_instance, args.rank, args.rank // args.local_num_ranks, args.num_ranks, args.num_ranks // args.local_num_ranks, **common_params)
    elif impl_name == "MPI":
        return SpGQAFlashDecodeAttentionMPI(args.iris_instance, args.rank, args.rank // args.local_num_ranks, args.num_ranks, args.num_ranks // args.local_num_ranks, **common_params)
    else:
        raise ValueError(f"Unknown implementation choice: {impl_name}")


@register_test("correctness")
def test_triton_decode_with_paged_kv(args) -> None:
    """
    Test the correctness of the implementation compared to Torch Reference
    Set CORRECTNESS_IMPL_TO_TEST to select which implementation to test. default=FUSED_FULL
    """
    kv_lens_per_rank = [128 * 12]
    num_heads = 96
    head_size = 128
    block_size = 1
    dtype = torch.float16
    soft_cap = 0
    torch.set_default_device("cuda")
    
    iris_instance = args.iris_instance
    num_seqs = len(kv_lens_per_rank)
    num_query_heads = num_heads
    num_kv_heads = num_query_heads // 8
    scale = head_size**-0.5
    NUM_BLOCKS_PER_RANK = 128 * 12 + 1
    NUM_BLOCKS = NUM_BLOCKS_PER_RANK * args.num_ranks

    common_params = {
        "num_q_heads": num_query_heads, "num_kv_heads": num_kv_heads, "q_head_dim": head_size,
        "v_head_dim": head_size, "page_size": block_size, "scale": scale, "soft_cap": soft_cap,
        "max_allowed_batch": num_seqs
    }
    ths_op = get_op_instance(CORRECTNESS_IMPL_TO_TEST, args, common_params)
    
    for i in range(1):
        iris_instance.barrier()
        if hasattr(ths_op, 'clear_flags'):
            ths_op.clear_flags()
        elif hasattr(ths_op, 'iris_ag_layer') and hasattr(ths_op.iris_ag_layer, 'clear_flags'):
            ths_op.iris_ag_layer.clear_flags()
        iris_instance.barrier()
        
        dist_print(f"\n<<<<<<<<<< Correctness Test (Impl: {CORRECTNESS_IMPL_TO_TEST}): Iteration {i+1} >>>>>>>>>>", allowed_ranks=[0])
        
        # Data preparation
        if args.rank == 0:
            query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype) / 10
            key_value_cache = torch.randn(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype) / 10
        else:
            query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
            key_value_cache = torch.empty(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype)
        
        query = torch.from_numpy(iris_instance.broadcast_tensor(query.cpu().numpy(), source_rank=0)).to(query.device)
        key_value_cache = torch.from_numpy(iris_instance.broadcast_tensor(key_value_cache.cpu().numpy(), source_rank=0)).to(key_value_cache.device)

        key_cache = key_value_cache[:, 0, :, :, :].contiguous()
        value_cache = key_value_cache[:, 1, :, :, :].contiguous()
        key_cache_this_rank = key_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) * NUM_BLOCKS_PER_RANK].contiguous()
        value_cache_this_rank = value_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) * NUM_BLOCKS_PER_RANK].contiguous()

        block_tables_this_rank = torch.arange(NUM_BLOCKS_PER_RANK, dtype=torch.int32).unsqueeze(0)
        all_block_tables_numpy = iris._mpi_helpers.mpi_allgather(block_tables_this_rank.cpu().numpy())
        block_tables = torch.from_numpy(all_block_tables_numpy).view(args.num_ranks, num_seqs, -1)
        ref_block_tables = torch.cat([block_tables[i] + i * NUM_BLOCKS_PER_RANK for i in range(args.num_ranks)], dim=-1).squeeze(0)

        global_kv_lens = [kv_lens_per_rank[0] * args.num_ranks]
        kv_lens_tensor = torch.tensor(kv_lens_per_rank, dtype=torch.int32, device=query.device)
        global_kv_lens_tensor = torch.cat([kv_lens_tensor.view(1, -1) for _ in range(args.num_ranks)], dim=0)

        # Run chosen op
        output = ths_op(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)
        torch.cuda.synchronize()
        
        # Run reference op
        ref_output = ref_paged_attn(
            query=query.clone(), key_cache=key_cache, value_cache=value_cache,
            query_lens=[1] * num_seqs, kv_lens_per_rank=global_kv_lens,
            block_tables=ref_block_tables.unsqueeze(0), scale=scale, soft_cap=soft_cap
        )
        iris_instance.barrier()

        # Validation and detailed print
        if args.rank == 0:
            print(f"--- Detailed Validation on Rank {args.rank} ---")
            header = f"{'Index':<8} | {'Computed':<15} | {'Reference':<15} | {'Abs. Diff':<15}"
            print("--- Comparison of First 16 Values (Head 0) ---")
            print(header)
            print("-" * len(header))
            comp_slice = output[0, 0, :16].cpu()
            ref_slice = ref_output[0, 0, :16].cpu()
            diff_slice = torch.abs(comp_slice - ref_slice)
            for j in range(len(comp_slice)):
                print(f"{j:<8} | {comp_slice[j]:<15.6f} | {ref_slice[j]:<15.6f} | {diff_slice[j]:<15.6f}")
            print("-" * 50)
        
        try:
            torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
            max_diff = torch.max(torch.abs(output - ref_output))
            print(f"✅ TEST PASSED for Rank {args.rank}")
            print(f"   Max absolute difference: {max_diff:.6f} (within tolerance)")
        except AssertionError as e:
            print(f"❌ TEST FAILED for Rank {args.rank}:\n{e}")
        iris_instance.barrier()

@register_test("perf")
def perf_decode_iris(args):
    """
    Benchmarks the selected implementations across various, extremely large sequence lengths.
    """
    kv_len_configs = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    # kv_len_configs = [131072]
    full_results = []

    for kv_len_per_rank in kv_len_configs:
        num_heads, head_size, block_size, dtype, soft_cap, num_seqs = 96, 128, 1, torch.float16, 0, 1
        torch.set_default_device("cuda")
        num_query_heads, num_kv_heads = num_heads, num_heads // 8
        scale = head_size**-0.5
        NUM_BLOCKS_PER_RANK = kv_len_per_rank + 1
        
        dist_print(f"\n----- Benchmarking @ KV Length per Rank: {kv_len_per_rank} -----", allowed_ranks=[0])
        
        # Prepare tensors once per KV length
        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
        key_cache_this_rank = torch.randn(NUM_BLOCKS_PER_RANK, block_size, num_kv_heads, head_size, dtype=dtype)
        value_cache_this_rank = torch.randn(NUM_BLOCKS_PER_RANK, block_size, num_kv_heads, head_size, dtype=dtype)
        block_tables_this_rank = torch.randint(0, NUM_BLOCKS_PER_RANK, (num_seqs, NUM_BLOCKS_PER_RANK), dtype=torch.int32)
        kv_lens_tensor = torch.tensor([kv_len_per_rank], dtype=torch.int32)
        global_kv_lens_tensor = torch.cat([kv_lens_tensor.view(1, -1) for _ in range(args.num_ranks)], dim=0)

        kv_results = {"kv_len": kv_len_per_rank}

        for impl_name in PERF_IMPLS_TO_TEST:
            dist_print(f"--> Benchmarking: {impl_name}...", allowed_ranks=[0])
            common_params = {
                "num_q_heads": num_query_heads, "num_kv_heads": num_kv_heads, "q_head_dim": head_size,
                "v_head_dim": head_size, "page_size": block_size, "scale": scale, "soft_cap": soft_cap,
                "max_allowed_batch": num_seqs
            }
            ths_op = get_op_instance(impl_name, args, common_params)

            def func_to_benchmark():
                return ths_op(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)

            preamble_fn = None
            if hasattr(ths_op, 'clear_flags'):
                preamble_fn = ths_op.clear_flags
            elif hasattr(ths_op, 'iris_ag_layer') and hasattr(ths_op.iris_ag_layer, 'clear_flags'):
                preamble_fn = ths_op.iris_ag_layer.clear_flags

            time_ms = iris.do_bench(
                fn=func_to_benchmark,
                preamble_fn=preamble_fn,
                barrier_fn=args.iris_instance.barrier,
                n_warmup=5,
                n_repeat=20,
                return_mode="mean",
            )
            args.iris_instance.barrier()
            
            if args.rank == 0:
                dist_print(f"         Done. Average time: {time_ms:.3f} ms", allowed_ranks=[0])
                kv_results[impl_name] = time_ms
        
        full_results.append(kv_results)

    if args.rank == 0:
        print("\n\n--- Final Performance Summary ---")
        print(f"Implementations: {PERF_IMPLS_TO_TEST} | Num GPUs: {args.num_ranks}")
        header = f"{'KV Len/GPU':<15} | " + " | ".join([f"{(name + ' (ms)'):<18}" for name in PERF_IMPLS_TO_TEST])          
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for r in full_results:
            row = f"{r['kv_len']:<15} | "
            row += " | ".join([f"{r.get(name, 'N/A'):<18.3f}" for name in PERF_IMPLS_TO_TEST])
            print(row)
        print("-" * len(header))


@register_test("compare")
def perf_decode_comparison(args) -> None:
    """
    Benchmarks and compares implementations across various model shapes and sequence lengths.
    """
    # Define model shapes to test
    model_configs = [
    {"batch_size": 1, "num_q_heads": 96, "head_size": 128},
    {"batch_size": 4, "num_q_heads": 96, "head_size": 128},
    {"batch_size": 8, "num_q_heads": 96, "head_size": 128},
    {"batch_size": 1, "num_q_heads": 384, "head_size": 128},
    {"batch_size": 1, "num_q_heads": 96, "head_size": 32},
    ]
    
    # Dynamically set local KV lengths based on GPU count for consistent Global KV lengths
    # This can be adjusted to avoid OOM issues
    if args.num_ranks == 1:
        kv_len_configs = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    elif args.num_ranks == 2:
        kv_len_configs = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    elif args.num_ranks == 4:
        kv_len_configs = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    elif args.num_ranks == 8:
        kv_len_configs = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    else:
        print(f"Warning: Using default kv_len_configs for {args.num_ranks} GPUs.")
        kv_len_configs = [8192, 16384, 32768, 65536, 131072, 262144]
        
    warmup_iters = 5
    benchmark_iters = 20
    
    torch.set_default_device("cuda")

    all_benchmark_data_for_json = []

    for model_config in model_configs:
        results_summary = []
        
        batch_size = model_config["batch_size"]
        num_query_heads = model_config["num_q_heads"]
        head_size = model_config["head_size"]
        
        num_kv_heads = num_query_heads // 8
        scale = head_size**-0.5
        block_size = 1
        dtype = torch.float16
        soft_cap = 0
        bytes_per_token = num_kv_heads * head_size * 2 * 2 

        dist_print(f"\n\n{'#'*80}", allowed_ranks=[0])
        dist_print(f"### New Benchmark Suite ###", allowed_ranks=[0])
        dist_print(f"# Config: Batch={batch_size}, Q-Heads={num_query_heads}, Head-Dim={head_size}", allowed_ranks=[0])
        dist_print(f"{'#'*80}", allowed_ranks=[0])

        for kv_len_per_rank in kv_len_configs:
            NUM_BLOCKS_PER_RANK = kv_len_per_rank + 1
            
            kv_cache_size_bytes = kv_len_per_rank * bytes_per_token * batch_size
            kv_cache_size_gb = kv_cache_size_bytes / (1024 * 1024 * 1024)
            
            dist_print(f"\n{'='*20} Comparing @ KV Cache/GPU: {kv_cache_size_gb:.2f} GB (Len: {kv_len_per_rank}) {'='*20}", allowed_ranks=[0])

            query = torch.randn(batch_size, num_query_heads, head_size, dtype=dtype)
            key_cache_this_rank = torch.randn(NUM_BLOCKS_PER_RANK * batch_size, block_size, num_kv_heads, head_size, dtype=dtype)
            value_cache_this_rank = torch.randn(NUM_BLOCKS_PER_RANK * batch_size, block_size, num_kv_heads, head_size, dtype=dtype)
            block_tables_this_rank = torch.randint(0, NUM_BLOCKS_PER_RANK, (batch_size, NUM_BLOCKS_PER_RANK), dtype=torch.int32)
            kv_lens_tensor = torch.tensor([kv_len_per_rank] * batch_size, dtype=torch.int32)
            global_kv_lens_tensor = torch.cat([kv_lens_tensor.view(1, -1) for _ in range(args.num_ranks)], dim=0)
            
            args.iris_instance.barrier()
            torch.cuda.synchronize()

            common_op_params = {
                "num_q_heads": num_query_heads, "num_kv_heads": num_kv_heads, "q_head_dim": head_size,
                "v_head_dim": head_size, "page_size": block_size, "scale": scale, "soft_cap": soft_cap,
                "max_allowed_batch": batch_size
            }

            dist_print("--> Benchmarking: Fused Iris Version...", allowed_ranks=[0])
            ths_op_fused = get_op_instance("FUSED", args, common_op_params)
            fn_to_benchmark_fused = lambda: ths_op_fused(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)
            time_fused_ms = iris.do_bench(fn=fn_to_benchmark_fused, preamble_fn=ths_op_fused.clear_flags, barrier_fn=args.iris_instance.barrier, n_warmup=warmup_iters, n_repeat=benchmark_iters, return_mode="mean")
            dist_print(f"         Done. Average time: {time_fused_ms:.3f} ms", allowed_ranks=[0])
            args.iris_instance.barrier()

            dist_print("\n--> Benchmarking: Fused Full Iris Version...", allowed_ranks=[0])
            ths_op_fused_full = get_op_instance("FUSED_FULL", args, common_op_params)
            fn_to_benchmark_fused_full = lambda: ths_op_fused_full(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)
            time_fused_full_ms = iris.do_bench(fn=fn_to_benchmark_fused_full, preamble_fn=ths_op_fused_full.clear_flags, barrier_fn=args.iris_instance.barrier, n_warmup=warmup_iters, n_repeat=benchmark_iters, return_mode="mean")
            dist_print(f"         Done. Average time: {time_fused_full_ms:.3f} ms", allowed_ranks=[0])
            args.iris_instance.barrier()

            dist_print("\n--> Benchmarking: No Wait Iris Version...", allowed_ranks=[0])
            ths_op_no_wait = get_op_instance("NOWAIT", args, common_op_params)
            fn_to_benchmark_iris_ag_no_wait = lambda: ths_op_no_wait(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)
            time_no_wait_ms = iris.do_bench(fn=fn_to_benchmark_iris_ag_no_wait, preamble_fn=ths_op_no_wait.iris_ag_layer.clear_flags, barrier_fn=args.iris_instance.barrier, n_warmup=warmup_iters, n_repeat=benchmark_iters, return_mode="mean")
            dist_print(f"         Done. Average time: {time_no_wait_ms:.3f} ms", allowed_ranks=[0])
            args.iris_instance.barrier()

            dist_print("\n--> Benchmarking: Standard Iris AG Version...", allowed_ranks=[0])
            ths_op_iris_ag = get_op_instance("STANDARD", args, common_op_params)
            fn_to_benchmark_iris_ag = lambda: ths_op_iris_ag(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)
            time_iris_ag_ms = iris.do_bench(fn=fn_to_benchmark_iris_ag, preamble_fn=ths_op_iris_ag.iris_ag_layer.clear_flags, barrier_fn=args.iris_instance.barrier, n_warmup=warmup_iters, n_repeat=benchmark_iters, return_mode="mean")
            dist_print(f"         Done. Average time: {time_iris_ag_ms:.3f} ms", allowed_ranks=[0])
            args.iris_instance.barrier()

            dist_print("\n--> Benchmarking: MPI Baseline...", allowed_ranks=[0])
            ths_op_mpi = get_op_instance("MPI", args, common_op_params)
            fn_to_benchmark_mpi = lambda: ths_op_mpi(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)
            time_mpi_ms = iris.do_bench(fn=fn_to_benchmark_mpi, barrier_fn=args.iris_instance.barrier, n_warmup=warmup_iters, n_repeat=benchmark_iters, return_mode="mean")
            dist_print(f"         Done. Average time: {time_mpi_ms:.3f} ms", allowed_ranks=[0])
            args.iris_instance.barrier()
        
            if args.rank == 0:
                results_summary.append({
                    "kv_len": kv_len_per_rank, "cache_gb": kv_cache_size_gb,
                    "fused_ms": time_fused_ms, "fused_full_ms": time_fused_full_ms,
                    "no_wait_ms": time_no_wait_ms, "iris_ag_ms": time_iris_ag_ms,
                    "mpi_ms": time_mpi_ms,
                })
        
        args.iris_instance.barrier()
        if args.rank == 0 and results_summary:
            # TODO Add console and text log printing logic
            
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

    if args.rank == 0 and all_benchmark_data_for_json:
        json_filename = "compare_results.json"
        
        existing_data = []
        if os.path.exists(json_filename):
            try:
                with open(json_filename, 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    print(f"Warning: Existing {json_filename} is not a list. Overwriting.")
                    existing_data = []
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not read or parse {json_filename}. Overwriting.")
                existing_data = []

        existing_data.extend(all_benchmark_data_for_json)

        dist_print(f"\nAppending structured results to {json_filename}...", allowed_ranks=[0])
        try:
            # Write the combined list back to the file
            with open(json_filename, 'w') as f:
                json.dump(existing_data, f, indent=4)
            dist_print(f"Successfully appended to {json_filename}", allowed_ranks=[0])
        except IOError as e:
            dist_print(f"Error writing to {json_filename}: {e}", allowed_ranks=[0], is_error=True)


if __name__ == "__main__":
    _iris = iris.iris() 

    args = get_args()
    args.rank = _iris.get_rank()
    args.local_num_ranks = _iris.get_num_ranks()
    args.num_ranks = _iris.get_num_ranks()
    args.iris_instance = _iris
    
    if args.list:
        help()
        sys.exit()
    if args.case:
        if args.case in ALL_TESTS:
            func = ALL_TESTS[args.case]
            func(args)
        else:
            print(f"Error: Case '{args.case}' not found in ALL_TESTS.")

    _iris.barrier()