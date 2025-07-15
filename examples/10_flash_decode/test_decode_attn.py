################################################################################
#
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
from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch
# import nvshmem.core

# from triton_dist.kernels.nvidia import (gqa_fwd_batch_decode, gqa_fwd_batch_decode_aot, gqa_fwd_batch_decode_persistent,
#                                         gqa_fwd_batch_decode_persistent_aot)
# from triton_dist.utils import dist_print, perf_func, init_nvshmem_by_torch_process_group

from decode_kernels import (gqa_fwd_batch_decode, gqa_fwd_batch_decode_intra_rank)

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
    kv_lens: List[int],
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
        kv_len = kv_lens[i]
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


NUM_BLOCKS = 32000  # Large enough to test overflow in index calculation.


def test_triton_decode_with_paged_kv(
    kv_lens: List[int],
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
) -> None:
    torch.set_default_device("cuda")
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)

    key_value_cache = torch.randn(NUM_BLOCKS + 1, 2, block_size, num_kv_heads, head_size, dtype=dtype)
    key_cache = key_value_cache[:, 0, :, :, :].contiguous()
    value_cache = key_value_cache[:, 1, :, :, :].contiguous()
    workspace = torch.zeros([num_seqs * num_query_heads * 32], dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)

    # output = gqa_fwd_batch_decode(query, key_cache, value_cache, workspace, [1] * num_seqs,
    #                               torch.tensor(kv_lens, dtype=torch.int32, device=query.device), block_tables, scale,
    #                               soft_cap)

    output = gqa_fwd_batch_decode_intra_rank(query, key_cache, value_cache, workspace, [1] * num_seqs,
                                  torch.tensor(kv_lens, dtype=torch.int32, device=query.device), block_tables, scale,
                                  soft_cap)


    ref_output = ref_paged_attn(query=query, key_cache=key_cache, value_cache=value_cache, query_lens=[1] * num_seqs,
                                kv_lens=kv_lens, block_tables=block_tables, scale=scale, soft_cap=soft_cap)
    
    # --- Print the Output ---
    print("\n--- Actual Output ---")
    print(f"Shape: {output.shape}")
    print(
        "A small slice of the result (first sequence, first head, first 10 values):"
    )
    print(output[0, 0, :10].cpu())
    print("-" * 45)

     # --- Print the Result ---
    print("\n--- Golden Reference Output ---")
    print(f"Shape: {ref_output.shape}")
    print(
        "A small slice of the ref result (first sequence, first head, first 10 values):"
    )
    print(ref_output[0, 0, :10].cpu())
    print("-" * 45)

    max_diff = torch.max(torch.abs(output - ref_output))
    try:
        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
        print(f"TEST PASSED. Max absolute difference: {max_diff:.6f}")
    except AssertionError as e:
        print(f"TEST FAILED: {e}. Max absolute difference: {max_diff:.6f}")




if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    # torch.cuda.set_device(LOCAL_RANK)
    # torch.distributed.init_process_group(
    #     backend="nccl",
    #     world_size=WORLD_SIZE,
    #     rank=RANK,
    #     timeout=datetime.timedelta(seconds=1800),
    # )
    # assert torch.distributed.is_initialized()
    # TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    # torch.distributed.barrier(TP_GROUP)

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
    # args.default_group = TP_GROUP
    args.rank = RANK
    args.num_ranks = WORLD_SIZE
    if args.list:
        help()
        sys.exit()
    # func = ALL_TESTS[args.case]
    # func(args)

    kv_lens_param = [1320, 18, 463]
    num_heads_param = (16, 16)
    head_size_param = 128
    dtype_param = torch.float16
    block_size_param = 16
    soft_cap_param = 0.0

    # Manually set seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Call the function with the defined parameters
    test_triton_decode_with_paged_kv(
        kv_lens=kv_lens_param,
        num_heads=num_heads_param,
        head_size=head_size_param,
        dtype=dtype_param,
        block_size=block_size_param,
        soft_cap=soft_cap_param,
    )

    # nvshmem.core.finalize()
    # torch.distributed.destroy_process_group()