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

from sp_flash_decode_layer import SpGQAFlashDecodeAttention


ALL_TESTS = {}


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

    _iris = iris.iris() 

    torch.set_default_device("cuda")

    num_seqs = len(kv_lens_per_rank)
    num_query_heads = num_heads
    num_kv_heads = num_query_heads // 8
    assert num_query_heads % num_kv_heads == 0
    scale = head_size**-0.5

    NUM_BLOCKS_PER_RANK = 128 * 12 + 1
    NUM_BLOCKS = NUM_BLOCKS_PER_RANK * args.num_ranks

    ths_op = SpGQAFlashDecodeAttention(args.rank, args.rank // args.local_num_ranks, args.num_ranks,
                                       args.num_ranks // args.local_num_ranks, num_query_heads, num_kv_heads, head_size,
                                       head_size, page_size=block_size, scale=scale, soft_cap=soft_cap,
                                       max_allowed_batch=1, thrink_buffer_threshold=500, stages=20)

    for _ in range(2):
        if args.rank == 0:
            query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype) / 10
            key_value_cache = torch.randn(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype) / 10
        else:
            query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
            key_value_cache = torch.empty(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype)
        
        query = torch.from_numpy(_iris.broadcast_tensor(query, source_rank=0)).to(query.device)
        key_value_cache = torch.from_numpy(_iris.broadcast_tensor(key_value_cache, source_rank=0)).to(key_value_cache.device)

        # torch.manual_seed(3 + args.rank) 
        # print("SEED:", 3 + args.rank)

        key_cache = key_value_cache[:, 0, :, :, :].contiguous()
        value_cache = key_value_cache[:, 1, :, :, :].contiguous()
        key_cache_this_rank = key_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) *
                                        NUM_BLOCKS_PER_RANK].contiguous()
        value_cache_this_rank = value_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) *
                                             NUM_BLOCKS_PER_RANK].contiguous()

        max_num_blocks_per_seq_per_rank = NUM_BLOCKS_PER_RANK
        block_tables_list = [
            torch.randint(0, NUM_BLOCKS_PER_RANK, (num_seqs, max_num_blocks_per_seq_per_rank), dtype=torch.int32)
            for i in range(args.num_ranks)
        ]
        block_tables_list_shift = [
            torch.zeros((num_seqs, max_num_blocks_per_seq_per_rank)).to(torch.int32) + i * NUM_BLOCKS_PER_RANK
            for i in range(args.num_ranks)
        ]
        block_tables_shift = torch.cat(block_tables_list_shift, dim=-1)
        block_tables_this_rank = block_tables_list[args.rank]

        gathered_block_tables_numpy = iris._mpi_helpers.mpi_allgather(block_tables_this_rank.cpu().numpy().flatten())
        
        # print(f"Rank {args.rank}: Sending block_tables_this_rank shape: {block_tables_this_rank.shape}, numel: {block_tables_this_rank.numel()}")
        # print(f"Rank {args.rank}: Sending block_tables_this_rank flattened numpy size: {block_tables_this_rank.cpu().numpy().flatten().size}")
        
        # print(f"Rank {args.rank}: After mpi_allgather, gathered_block_tables_numpy shape: {gathered_block_tables_numpy.shape}")
        # print(f"Rank {args.rank}: After mpi_allgather, gathered_block_tables_numpy flattened size: {gathered_block_tables_numpy.size}")


        gathered_block_tables_list = []
        original_shape = block_tables_this_rank.shape
        elements_per_rank = block_tables_this_rank.numel()
        
        actual_gathered_ranks_count = gathered_block_tables_numpy.shape[0]

        for i in range(actual_gathered_ranks_count):
            rank_data_numpy_1d = gathered_block_tables_numpy[i, :] 
            
            # print(f"Rank {args.rank}: Iter {i}, rank_data_numpy_1d size: {rank_data_numpy_1d.size}, expected: {elements_per_rank}")

            gathered_block_tables_list.append(torch.from_numpy(rank_data_numpy_1d).to(block_tables_this_rank.device).view(original_shape))
        
        block_tables = torch.cat(gathered_block_tables_list, dim=-1) + block_tables_shift


        global_kv_lens = [i * args.num_ranks for i in kv_lens_per_rank]
        kv_lens_tensor = torch.tensor(kv_lens_per_rank, dtype=torch.int32, device=query.device)
        global_kv_lens_tensor = torch.cat([kv_lens_tensor.view(1, -1) for _ in range(args.num_ranks)], dim=0)

        output = ths_op(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor,
                        block_tables_this_rank)

        new_query_for_ref = torch.empty_like(query).copy_(query)

        ref_output = ref_paged_attn(query=new_query_for_ref, key_cache=key_cache, value_cache=value_cache,
                                     query_lens=[1] * num_seqs, kv_lens_per_rank=global_kv_lens,
                                     block_tables=block_tables, scale=scale, soft_cap=soft_cap)
        
        print(f"\n--- Rank {args.rank} Output ---")
        print(f"Shape: {output.shape}")
        print("A small slice of the computed result (first sequence, first head, first 10 values):")
        print(output[0, 0, :10].cpu())
        print("-" * 45)

        if args.rank == 0: 
            print("\n--- Golden Reference Output ---")
            print(f"Shape: {ref_output.shape}")
            print("A small slice of the reference result (first sequence, first head, first 10 values):")
            print(ref_output[0, 0, :10].cpu())
            print("-" * 45)

        try:
            torch.testing.assert_close(output, ref_output, atol=0.05, rtol=1e-2)
            max_val_out = torch.max(torch.abs(output))
            max_val_ref = torch.max(torch.abs(ref_output))
            print(f"Max Val OUTPUT {max_val_out}, Max Val REF {max_val_ref}")
            max_diff = torch.max(torch.abs(output - ref_output))
            print(f"TEST PASSED for Rank {args.rank}: Max absolute difference: {max_diff}")
        except AssertionError as e:
            max_diff = torch.max(torch.abs(output - ref_output))
            print(f"TEST FAILED for Rank {args.rank}: {e}. Max absolute difference: {max_diff}")

    _iris.barrier()


@register_test("perf")
def perf_decode(args):
    _iris = iris.iris() 

    for kv_len_per_rank in [2**i for i in range(10, 18)]:
        kv_lens_per_rank = [kv_len_per_rank]
        num_heads = 96
        head_size = 128
        block_size = 1
        dtype = torch.float16
        soft_cap = 0

        torch.set_default_device("cuda")

        num_seqs = len(kv_lens_per_rank)
        num_query_heads = num_heads
        num_kv_heads = num_query_heads // 8
        assert num_query_heads % num_kv_heads == 0
        scale = head_size**-0.5

        NUM_BLOCKS_PER_RANK = kv_len_per_rank[0] + 1
        NUM_BLOCKS = NUM_BLOCKS_PER_RANK * args.num_ranks 

        if args.rank == 0:
            query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
            key_value_cache = torch.randn(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype)
        else:
            query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
            key_value_cache = torch.empty(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype)
        
        query = torch.from_numpy(_iris.broadcast_tensor(query, source_rank=0)).to(query.device)
        key_value_cache = torch.from_numpy(_iris.broadcast_tensor(key_value_cache, source_rank=0)).to(key_value_cache.device)

        torch.manual_seed(3 + args.rank) 
        
        key_cache = key_value_cache[:, 0, :, :, :].contiguous()
        value_cache = key_value_cache[:, 1, :, :, :].contiguous()
        key_cache_this_rank = key_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) *
                                        NUM_BLOCKS_PER_RANK].contiguous()
        value_cache_this_rank = value_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) *
                                             NUM_BLOCKS_PER_RANK].contiguous()

        max_num_blocks_per_seq_per_rank = NUM_BLOCKS_PER_RANK
        block_tables_list = [
            torch.randint(0, NUM_BLOCKS_PER_RANK, (num_seqs, max_num_blocks_per_seq_per_rank), dtype=torch.int32)
            for i in range(args.num_ranks)
        ]
        block_tables_this_rank = block_tables_list[args.rank]

        kv_lens_tensor = torch.tensor(kv_lens_per_rank, dtype=torch.int32, device=query.device)
        global_kv_lens_tensor = torch.cat([kv_lens_tensor.view(1, -1) for _ in range(args.num_ranks)], dim=0)

        ths_op = SpGQAFlashDecodeAttention(args.rank, args.rank // args.local_num_ranks, args.num_ranks,
                                           args.num_ranks // args.local_num_ranks, num_query_heads, num_kv_heads,
                                           head_size, head_size, page_size=block_size, scale=scale, soft_cap=soft_cap,
                                           max_allowed_batch=1, thrink_buffer_threshold=500)
        torch.cuda.synchronize()
        _iris.barrier()

        def func():
            return ths_op(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor,
                            block_tables_this_rank)

        _iris.barrier()

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
    
    if args.list:
        help()
        sys.exit()
    func = ALL_TESTS[args.case]
    func(args)

    _iris_global.barrier()