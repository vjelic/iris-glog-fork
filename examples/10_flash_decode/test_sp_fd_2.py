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
import argparse
import datetime
import os
import sys
from typing import List, Optional

import numpy as np
import torch
# from cuda import cuda, cudart
# import nvshmem.core

# from triton_dist.layers.nvidia import SpGQAFlashDecodeAttention
# from triton_dist.utils import (dist_print, group_profile, init_nvshmem_by_torch_process_group,
#                                  nvshmem_barrier_all_on_stream, perf_func, sleep_async)

# Import Iris for distributed operations
import iris

# Import the class definition from the local file
from sp_flash_decode_layer_2 import SpGQAFlashDecodeAttention

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


# Removed CUDA_CHECK as it's not directly related to Triton or distributed setup
# def CUDA_CHECK(err):
#     if isinstance(err, cuda.CUresult):
#         if err != cuda.CUresult.CUDA_SUCCESS:
#             raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
#     elif isinstance(err, cudart.cudaError_t):
#         if err != cudart.cudaError_t.cudaSuccess:
#             raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
#     else:
#         raise RuntimeError(f"Unknown error type: {err}")


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

    # Initialize Iris instance for broadcast/barrier
    _iris = iris.iris() # Initialize Iris for this process

    torch.set_default_device("cuda")

    num_seqs = len(kv_lens_per_rank)
    num_query_heads = num_heads
    num_kv_heads = num_query_heads // 8
    assert num_query_heads % num_kv_heads == 0
    scale = head_size**-0.5

    NUM_BLOCKS_PER_RANK = 128 * 12 + 1
    NUM_BLOCKS = NUM_BLOCKS_PER_RANK * args.num_ranks  # Large enough to test overflow in index calculation.

    ths_op = SpGQAFlashDecodeAttention(args.rank, args.rank // args.local_num_ranks, args.num_ranks,
                                       args.num_ranks // args.local_num_ranks, num_query_heads, num_kv_heads, head_size,
                                       head_size, page_size=block_size, scale=scale, soft_cap=soft_cap,
                                       max_allowed_batch=1, thrink_buffer_threshold=500, stages=20)

    for _ in range(2):
        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype) / 10
        # Use Iris broadcast for query synchronization
        query = torch.from_numpy(_iris.broadcast(0, query.cpu().numpy())).to(query.device)

        key_value_cache = torch.randn(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype) / 10
        # Use Iris broadcast for key_value_cache synchronization
        key_value_cache = torch.from_numpy(_iris.broadcast(0, key_value_cache.cpu().numpy())).to(key_value_cache.device)

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

        # Use Iris for all_gather equivalent.
        # Collect each rank's block_tables_this_rank
        gathered_block_tables_numpy = iris._mpi_helpers.mpi_allgather(block_tables_this_rank.cpu().numpy())
        # Reshape the gathered data back into a list of tensors for concatenation
        gathered_block_tables_list = []
        element_size = block_tables_this_rank.element_size()
        elements_per_rank = block_tables_this_rank.numel()
        
        for i in range(args.num_ranks):
            start_byte = i * elements_per_rank * element_size
            end_byte = start_byte + elements_per_rank * element_size
            # Create a view of the numpy array for each rank's data
            rank_data_numpy = gathered_block_tables_numpy.view(dtype=block_tables_this_rank.numpy().dtype)[i * elements_per_rank : (i + 1) * elements_per_rank]
            gathered_block_tables_list.append(torch.from_numpy(rank_data_numpy).to(block_tables_this_rank.device).view(block_tables_this_rank.shape))
        
        block_tables = torch.cat(gathered_block_tables_list, dim=-1) + block_tables_shift


        global_kv_lens = [i * args.num_ranks for i in kv_lens_per_rank]
        kv_lens_tensor = torch.tensor(kv_lens_per_rank, dtype=torch.int32, device=query.device)
        global_kv_lens_tensor = torch.cat([kv_lens_tensor.view(1, -1) for _ in range(args.num_ranks)], dim=0)

        query = torch.randn_like(query)
        # Use Iris broadcast for query synchronization
        query = torch.from_numpy(_iris.broadcast(0, query.cpu().numpy())).to(query.device)

        output = ths_op(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor,
                        block_tables_this_rank)

        new_query = torch.empty_like(query).copy_(query)

        ref_output = ref_paged_attn(query=new_query, key_cache=key_cache, value_cache=value_cache,
                                     query_lens=[1] * num_seqs, kv_lens_per_rank=global_kv_lens,
                                     block_tables=block_tables, scale=scale, soft_cap=soft_cap)
        

        # --- Print the Result ---
        print("\n--- Golden Reference Output ---")
        print(f"Shape: {ref_output.shape}")
        print(
            "A small slice of the result (first sequence, first head, first 10 values):"
        )
        print(ref_output[0, 0, :10].cpu())
        print("-" * 45)

        # Assertion using torch.testing.assert_close with clear pass/fail output
        try:
            torch.testing.assert_close(output, ref_output, atol=0.05, rtol=1e-2)
            print("TEST PASSED")
        except AssertionError as e:
            max_diff = torch.max(torch.abs(output - ref_output))
            print(f"TEST FAILED: {e}. Max absolute difference: {max_diff}")


@register_test("perf")
def perf_decode(args):
    _iris = iris.iris() # Initialize Iris for this process

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

        NUM_BLOCKS_PER_RANK = kv_lens_per_rank[0] + 1
        NUM_BLOCKS = NUM_BLOCKS_PER_RANK * args.num_ranks  # Large enough to test overflow in index calculation.

        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
        # Use Iris broadcast for query synchronization
        query = torch.from_numpy(_iris.broadcast(source_rank=0, value=query.cpu().numpy())).to(query.device)

        key_value_cache = torch.randn(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype) / 10
        key_value_cache = torch.from_numpy(_iris.broadcast(source_rank=0, value=key_value_cache.cpu().numpy())).to(key_value_cache.device)

        query = torch.randn_like(query)
        query = torch.from_numpy(_iris.broadcast(source_rank=0, value=query.cpu().numpy())).to(query.device)
        
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
        # Use Iris barrier for synchronization
        _iris.barrier()

        def func():
            return ths_op(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor,
                            block_tables_this_rank)

        # Removed perf_func and group_profile, as they are not directly provided by Iris
        # If performance profiling is needed, integrate existing Triton/PyTorch profiling tools
        # or implement a simple timing mechanism.
        # perf_func(func, iters=100, warmup_iters=20)

        # Use Iris barrier for synchronization
        _iris.barrier()
        # dist_print(f"rank: {args.rank} KV len={kv_lens_per_rank[0]} Performance is {time_ms} ms", allowed_ranks="all",
        #            need_sync=True)


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)

    # Initialize Iris at the very beginning of each process
    # This replaces torch.distributed.init_process_group and nvshmem.core.init_nvshmem_by_torch_process_group
    _iris_global = iris.iris() # Global Iris instance for main function utilities

    # Removed all torch.distributed and nvshmem.core related initialization
    # torch.distributed.init_process_group(
    #     backend="nccl",
    #     world_size=WORLD_SIZE,
    #     rank=RANK,
    #     timeout=datetime.timedelta(seconds=1800),
    # )
    # assert torch.distributed.is_initialized()
    # TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    # torch.distributed.barrier(TP_GROUP)

    # Keep torch random seed settings if desired
    # torch.use_deterministic_algorithms(False, warn_only=True)
    # torch.set_printoptions(precision=2)
    # torch.manual_seed(3 + RANK)
    # torch.cuda.manual_seed_all(3 + RANK)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    # torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    # np.random.seed(3 + RANK)

    # current_stream = torch.cuda.current_stream()
    # torch.cuda.synchronize()
    # init_nvshmem_by_torch_process_group(TP_GROUP)

    args = get_args()
    # Update args to use Iris rank and num_ranks
    args.rank = _iris_global.get_rank()
    args.num_ranks = _iris_global.get_num_ranks()
    args.local_rank = LOCAL_RANK # Assuming LOCAL_RANK is still useful for device selection
    args.local_num_ranks = LOCAL_WORLD_SIZE
    
    if args.list:
        help()
        sys.exit()
    func = ALL_TESTS[args.case]
    func(args)

    # Finalize Iris (this will call MPI_Finalize)
    # nvshmem.core.finalize()
    # torch.distributed.destroy_process_group()