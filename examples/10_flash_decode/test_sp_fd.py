# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

import argparse
import datetime
import os
import sys
from typing import List, Optional

import numpy as np
import torch
import iris

from utils import (perf_func, dist_print, group_profile)
from sp_flash_decode_layer_iris import SpGQAFlashDecodeAttentionIrisAG
from sp_flash_decode_layer import SpGQAFlashDecodeAttention
from sp_flash_decode_layer_mpi import SpGQAFlashDecodeAttentionMPI
from sp_flash_decode_layer_iris_fused import SpGQAFlashDecodeAttentionIrisAGFused


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


@register_test("correctness")
def test_triton_decode_with_paged_kv(args) -> None:
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
    assert num_query_heads % num_kv_heads == 0
    scale = head_size**-0.5
    NUM_BLOCKS_PER_RANK = 128 * 12 + 1
    NUM_BLOCKS = NUM_BLOCKS_PER_RANK * args.num_ranks

    ths_op = SpGQAFlashDecodeAttentionIrisAGFused(
        iris_instance, args.rank, args.rank // args.local_num_ranks, args.num_ranks,
        args.num_ranks // args.local_num_ranks, num_query_heads, num_kv_heads, head_size,
        head_size, page_size=block_size, scale=scale, soft_cap=soft_cap,
        max_allowed_batch=1, thrink_buffer_threshold=500, stages=20
    )

    for i in range(4):
        dist_print(f"\n<<<<<<<<<< Correctness Test: Iteration {i+1} >>>>>>>>>>", allowed_ranks=[0])
        if args.rank == 0:
            query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype) / 10
            key_value_cache = torch.randn(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype) / 10
        else:
            query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
            key_value_cache = torch.empty(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype)
        
        query = torch.from_numpy(iris_instance.broadcast_tensor(query, source_rank=0)).to(query.device)
        key_value_cache = torch.from_numpy(iris_instance.broadcast_tensor(key_value_cache, source_rank=0)).to(key_value_cache.device)

        key_cache = key_value_cache[:, 0, :, :, :].contiguous()
        value_cache = key_value_cache[:, 1, :, :, :].contiguous()
        key_cache_this_rank = key_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) * NUM_BLOCKS_PER_RANK].contiguous()
        value_cache_this_rank = value_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) * NUM_BLOCKS_PER_RANK].contiguous()

        block_tables_this_rank = torch.arange(NUM_BLOCKS_PER_RANK, dtype=torch.int32).unsqueeze(0)
        all_block_tables_numpy = iris._mpi_helpers.mpi_allgather(block_tables_this_rank.cpu().numpy())
        block_tables = torch.from_numpy(all_block_tables_numpy).view(args.num_ranks, num_seqs, -1)
        
        ref_block_tables = torch.cat([
             block_tables[i] + i * NUM_BLOCKS_PER_RANK for i in range(args.num_ranks)
        ], dim=-1).squeeze(0)

        global_kv_lens = [kv_lens_per_rank[0] * args.num_ranks]
        kv_lens_tensor = torch.tensor(kv_lens_per_rank, dtype=torch.int32, device=query.device)
        global_kv_lens_tensor = torch.cat([kv_lens_tensor.view(1, -1) for _ in range(args.num_ranks)], dim=0)

        output = ths_op(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank, iteration_id=i)
        ref_output = ref_paged_attn(
            query=query.clone(), key_cache=key_cache, value_cache=value_cache,
            query_lens=[1] * num_seqs, kv_lens_per_rank=global_kv_lens,
            block_tables=ref_block_tables.unsqueeze(0), scale=scale, soft_cap=soft_cap
        )
        
        iris_instance.barrier()

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

      
        atol = 1e-3
        rtol = 1e-2
        
        try:
        
            torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)
            max_diff = torch.max(torch.abs(output - ref_output))
            print(f"✅ TEST PASSED for Rank {args.rank}")
            print(f"   Max absolute difference: {max_diff:.6f} (within tolerance)")
        except AssertionError as e:
            print(f"❌ TEST FAILED for Rank {args.rank}:\n{e}")

        iris_instance.barrier()

@register_test("perf")
def perf_decode_iris(args):
    """
    Benchmarks the SpGQAFlashDecodeAttentionIrisAG layer across various, extremely large sequence lengths.
    """
    kv_len_configs = [131072, 262144, 524288, 1048576]

    for kv_len_per_rank in kv_len_configs:
        num_heads = 96
        head_size = 128
        block_size = 1
        dtype = torch.float16
        soft_cap = 0
        num_seqs = 1 

        torch.set_default_device("cuda")

        num_query_heads = num_heads
        num_kv_heads = num_query_heads // 8
        assert num_query_heads % num_kv_heads == 0, "Number of query heads must be divisible by KV heads."
        scale = head_size**-0.5

        NUM_BLOCKS_PER_RANK = kv_len_per_rank + 1
        
        dist_print(f"\n----- Benchmarking (rank {args.rank}) | KV Length per Rank: {kv_len_per_rank} -----", allowed_ranks="all")

        ths_op = SpGQAFlashDecodeAttentionIrisAG(
            args.iris_instance, args.rank, args.rank // args.local_num_ranks, args.num_ranks,
            args.num_ranks // args.local_num_ranks, num_query_heads, num_kv_heads, head_size,
            head_size, page_size=block_size, scale=scale, soft_cap=soft_cap,
            max_allowed_batch=num_seqs, thrink_buffer_threshold=500, stages=20
        )
        
        # ths_op = SpGQAFlashDecodeAttention(
        #     args.rank, args.rank // args.local_num_ranks, args.num_ranks,
        #     args.num_ranks // args.local_num_ranks, num_query_heads, num_kv_heads, head_size,
        #     head_size, page_size=block_size, scale=scale, soft_cap=soft_cap,
        #     max_allowed_batch=num_seqs, thrink_buffer_threshold=500, stages=20
        # )
        
        # ths_op = SpGQAFlashDecodeAttentionMPI(
        #     args.iris_instance, args.rank, args.rank // args.local_num_ranks, args.num_ranks,
        #     args.num_ranks // args.local_num_ranks, num_query_heads, num_kv_heads, head_size,
        #     head_size, page_size=block_size, scale=scale, soft_cap=soft_cap,
        #     max_allowed_batch=num_seqs, thrink_buffer_threshold=500, stages=20
        # )

        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
        key_cache_this_rank = torch.randn(NUM_BLOCKS_PER_RANK, block_size, num_kv_heads, head_size, dtype=dtype)
        value_cache_this_rank = torch.randn(NUM_BLOCKS_PER_RANK, block_size, num_kv_heads, head_size, dtype=dtype)

        max_num_blocks_per_seq_per_rank = NUM_BLOCKS_PER_RANK
        block_tables_this_rank = torch.randint(
            0, NUM_BLOCKS_PER_RANK, (num_seqs, max_num_blocks_per_seq_per_rank), dtype=torch.int32
        )
        
        kv_lens_this_rank_list = [kv_len_per_rank]
        kv_lens_tensor = torch.tensor(kv_lens_this_rank_list, dtype=torch.int32)
        global_kv_lens_tensor = torch.cat([kv_lens_tensor.view(1, -1) for _ in range(args.num_ranks)], dim=0)

        args.iris_instance.barrier()
        torch.cuda.synchronize()

        def func_to_benchmark():
            return ths_op(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)

        _, time_ms = perf_func(func=func_to_benchmark, iters=100, warmup_iters=20)

        args.iris_instance.barrier()
        
        dist_print(f"✅ Result: KV len={kv_len_per_rank}, Avg GPU time={time_ms:.3f} ms", allowed_ranks=[0])

@register_test("compare")
def perf_decode_comparison(args):
    """
    Benchmarks and compares SpGQAFlashDecodeAttentionIrisAG against a standard
    SpGQAFlashDecodeAttention implementation at extreme scales.
    """
    kv_len_configs = [131072, 262144, 524288, 1048576]
    
    results_summary = []

    for kv_len_per_rank in kv_len_configs:
        num_heads = 96
        head_size = 128
        block_size = 1
        dtype = torch.float16
        soft_cap = 0
        num_seqs = 1
        torch.set_default_device("cuda")

        num_query_heads = num_heads
        num_kv_heads = num_query_heads // 8
        assert num_query_heads % num_kv_heads == 0
        scale = head_size**-0.5

        NUM_BLOCKS_PER_RANK = kv_len_per_rank + 1
        
        dist_print(f"\n----- Comparing @ KV Len per Rank: {kv_len_per_rank} -----", allowed_ranks=[0])

        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
        key_cache_this_rank = torch.randn(NUM_BLOCKS_PER_RANK, block_size, num_kv_heads, head_size, dtype=dtype)
        value_cache_this_rank = torch.randn(NUM_BLOCKS_PER_RANK, block_size, num_kv_heads, head_size, dtype=dtype)
        max_num_blocks_per_seq_per_rank = NUM_BLOCKS_PER_RANK
        block_tables_this_rank = torch.randint(0, NUM_BLOCKS_PER_RANK, (num_seqs, max_num_blocks_per_seq_per_rank), dtype=torch.int32)
        kv_lens_tensor = torch.tensor([kv_len_per_rank], dtype=torch.int32)
        global_kv_lens_tensor = torch.cat([kv_lens_tensor.view(1, -1) for _ in range(args.num_ranks)], dim=0)
        
        args.iris_instance.barrier()
        torch.cuda.synchronize()



        dist_print("Benchmarking: SpGQAFlashDecodeAttentionMPI...", allowed_ranks=[0])
        ths_op_std = SpGQAFlashDecodeAttentionMPI(args.iris_instance,
            args.rank, args.rank // args.local_num_ranks, args.num_ranks,
            args.num_ranks // args.local_num_ranks, num_query_heads, num_kv_heads, head_size,
            head_size, page_size=block_size, scale=scale, soft_cap=soft_cap,
            max_allowed_batch=num_seqs, thrink_buffer_threshold=500, stages=20
        )
        def func_to_benchmark_std():
            return ths_op_std(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)
        _, time_std = perf_func(func=func_to_benchmark_std, iters=100, warmup_iters=20)
        args.iris_instance.barrier()
        
        dist_print("Benchmarking: SpGQAFlashDecodeAttentionIrisAG...", allowed_ranks=[0])
        ths_op_iris = SpGQAFlashDecodeAttentionIrisAG(
            args.iris_instance, args.rank, args.rank // args.local_num_ranks, args.num_ranks,
            args.num_ranks // args.local_num_ranks, num_query_heads, num_kv_heads, head_size,
            head_size, page_size=block_size, scale=scale, soft_cap=soft_cap,
            max_allowed_batch=num_seqs, thrink_buffer_threshold=500, stages=20
        )
        def func_to_benchmark_iris():
            return ths_op_iris(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)
        _, time_iris = perf_func(func=func_to_benchmark_iris, iters=100, warmup_iters=20)
        args.iris_instance.barrier()

        if args.rank == 0:
            speedup = time_std / time_iris if time_iris > 0 and time_std > 0 else 0.0
            results_summary.append({
                "kv_len": kv_len_per_rank,
                "iris_ms": time_iris,
                "std_ms": time_std,
                "speedup": speedup
            })
            print(f"Result (KV len={kv_len_per_rank}):")
            print(f"   - IrisAG:   {time_iris:.3f} ms")
            print(f"   - Standard: {time_std:.3f} ms")
            print(f"   - Relative Perf (Standard / IrisAG): {speedup:.2f}x")

    if args.rank == 0 and results_summary:
        print("\n\n--- Final Performance Comparison Summary ---")
        print("-" * 75)
        print(f"{'KV Len/GPU':<15} | {'IrisAG (ms)':<15} | {'Standard (ms)':<15} | {'Relative Perf':<15}")
        print("-" * 75)
        for r in results_summary:
            print(f"{r['kv_len']:<15} | {r['iris_ms']:<15.3f} | {r['std_ms']:<15.3f} | {r['speedup']:.2f}x")
        print("-" * 75)
        
@register_test("compare2")
def perf_decode_comparison(args) -> None:
    """
    Benchmarks and compares the fused Iris attention, standard Iris All-Gather,
    and the baseline MPI implementation.
    """
    kv_len_configs = [131072, 262144, 524288, 1048576]
    results_summary = []
    
    warmup_iters = 2
    benchmark_iters = 2

    for kv_len_per_rank in kv_len_configs:
        num_heads = 96
        head_size = 128
        block_size = 1
        dtype = torch.float16
        soft_cap = 0
        num_seqs = 1
        torch.set_default_device("cuda")

        num_query_heads = num_heads
        num_kv_heads = num_query_heads // 8
        scale = head_size**-0.5
        NUM_BLOCKS_PER_RANK = kv_len_per_rank + 1
        
        dist_print(f"\n{'='*20} Comparing @ KV Len per Rank: {kv_len_per_rank} {'='*20}", allowed_ranks=[0])

        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
        key_cache_this_rank = torch.randn(NUM_BLOCKS_PER_RANK, block_size, num_kv_heads, head_size, dtype=dtype)
        value_cache_this_rank = torch.randn(NUM_BLOCKS_PER_RANK, block_size, num_kv_heads, head_size, dtype=dtype)
        block_tables_this_rank = torch.randint(0, NUM_BLOCKS_PER_RANK, (num_seqs, NUM_BLOCKS_PER_RANK), dtype=torch.int32)
        kv_lens_tensor = torch.tensor([kv_len_per_rank], dtype=torch.int32)
        global_kv_lens_tensor = torch.cat([kv_lens_tensor.view(1, -1) for _ in range(args.num_ranks)], dim=0)
        
        args.iris_instance.barrier()
        torch.cuda.synchronize()

        # 1. BENCHMARK FUSED IRIS IMPLEMENTATION
      
        dist_print("--> Benchmarking: Fused Iris Version (SpGQAFlashDecodeAttentionIrisAGFused)...", allowed_ranks=[0])
        ths_op_fused = SpGQAFlashDecodeAttentionIrisAGFused(
            args.iris_instance, args.rank, args.rank // args.local_num_ranks, args.num_ranks,
            args.num_ranks // args.local_num_ranks, num_query_heads, num_kv_heads, head_size,
            head_size, page_size=block_size, scale=scale, soft_cap=soft_cap,
            max_allowed_batch=num_seqs
        )

        dist_print(f"    Running {warmup_iters} warmup iterations...", allowed_ranks=[0])
        for i in range(warmup_iters):
            ths_op_fused(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank, iteration_id=i)

        dist_print(f"    Running {benchmark_iters} benchmark iterations...", allowed_ranks=[0])
        torch.cuda.synchronize()
        args.iris_instance.barrier()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for i in range(warmup_iters, warmup_iters + benchmark_iters):
            ths_op_fused(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank, iteration_id=i)
        end_event.record()
        
        torch.cuda.synchronize()
        time_fused_ms = start_event.elapsed_time(end_event) / benchmark_iters
        dist_print(f"    Done. Average time: {time_fused_ms:.3f} ms", allowed_ranks=[0])
        
        args.iris_instance.barrier()

        # 2. BENCHMARK STANDARD IRIS ALL-GATHER IMPLEMENTATION
        
        dist_print("\n--> Benchmarking: Standard Iris AG Version (SpGQAFlashDecodeAttentionIrisAG)...", allowed_ranks=[0])
        ths_op_iris_ag = SpGQAFlashDecodeAttentionIrisAG(
            args.iris_instance, args.rank, args.rank // args.local_num_ranks, args.num_ranks,
            args.num_ranks // args.local_num_ranks, num_query_heads, num_kv_heads, head_size,
            head_size, page_size=block_size, scale=scale, soft_cap=soft_cap,
            max_allowed_batch=num_seqs
        )
        
        def func_to_benchmark_iris_ag():
            return ths_op_iris_ag(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)
        
        _, time_iris_ag_ms = perf_func(func=func_to_benchmark_iris_ag, iters=benchmark_iters, warmup_iters=warmup_iters)
        dist_print(f"    Done. Average time: {time_iris_ag_ms:.3f} ms", allowed_ranks=[0])
        
        args.iris_instance.barrier()

        
        # 3. BENCHMARK MPI BASELINE IMPLEMENTATION

        dist_print("\n--> Benchmarking: MPI Baseline (SpGQAFlashDecodeAttentionMPI)...", allowed_ranks=[0])
        ths_op_mpi = SpGQAFlashDecodeAttentionMPI(
            args.iris_instance, args.rank, args.rank // args.local_num_ranks, args.num_ranks,
            args.num_ranks // args.local_num_ranks, num_query_heads, num_kv_heads, head_size,
            head_size, page_size=block_size, scale=scale, soft_cap=soft_cap,
            max_allowed_batch=num_seqs
        )
        
        def func_to_benchmark_mpi():
            return ths_op_mpi(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor, block_tables_this_rank)
        
        _, time_mpi_ms = perf_func(func=func_to_benchmark_mpi, iters=benchmark_iters, warmup_iters=warmup_iters)
        dist_print(f"    Done. Average time: {time_mpi_ms:.3f} ms", allowed_ranks=[0])

        args.iris_instance.barrier()

      
        if args.rank == 0:
            results_summary.append({
                "kv_len": kv_len_per_rank,
                "fused_ms": time_fused_ms,
                "iris_ag_ms": time_iris_ag_ms,
                "mpi_ms": time_mpi_ms,
            })
            print(f"--- Comparison Result ---")
            print(f"  - Fused Iris    : {time_fused_ms:.3f} ms")
            print(f"  - Std. Iris AG  : {time_iris_ag_ms:.3f} ms")
            print(f"  - MPI Baseline  : {time_mpi_ms:.3f} ms")

    if args.rank == 0 and results_summary:
        print("\n\n--- Final Performance Comparison Summary ---")
        print("-" * 95)
        header = f"{'KV Len/GPU':<15} | {'Fused Iris (ms)':<20} | {'Std. Iris AG (ms)':<20} | {'MPI Baseline (ms)':<20} | {'Speedup vs MPI':<15}"
        print(header)
        print("-" * 95)
        for r in results_summary:
            speedup = r['mpi_ms'] / r['fused_ms'] if r['fused_ms'] > 0 else 0.0
            print(f"{r['kv_len']:<15} | {r['fused_ms']:<20.3f} | {r['iris_ag_ms']:<20.3f} | {r['mpi_ms']:<20.3f} | {speedup:<15.2f}x")
        print("-" * 95)

if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)

    _iris_global = iris.iris() 

    args = get_args()
    args.rank = _iris_global.get_rank()
    args.num_ranks = _iris_global.get_num_ranks()
    args.local_rank = LOCAL_RANK
    args.local_num_ranks = LOCAL_WORLD_SIZE
    args.iris_instance = _iris_global
    
    if args.list:
        help()
        sys.exit()
    func = ALL_TESTS[args.case]
    func(args)

    _iris_global.barrier()