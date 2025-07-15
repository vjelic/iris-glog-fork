################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# This code has been modified to run a single-process reference implementation.
# All distributed and custom operator logic has been removed.
#
################################################################################

import argparse
import datetime
import os
import sys
from typing import List, Optional

import numpy as np
import torch

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



def calculate_and_print_reference():
    """
    This function sets up data and computes the golden reference result on one GPU.
    """
    # --- Configuration ---
    kv_lens_per_rank = [128 * 12]
    num_heads = 96
    head_size = 128
    block_size = 1
    dtype = torch.float16
    soft_cap = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # FIXED: Set the default device for all subsequent tensor creations.
    # This ensures torch.ones inside ref_paged_attn creates a tensor on the GPU.
    torch.set_default_device(device)

    # Use a fixed seed for reproducible results
    torch.manual_seed(0)
    np.random.seed(0)

    num_seqs = len(kv_lens_per_rank)
    num_query_heads = num_heads
    num_kv_heads = num_query_heads // 8
    scale = head_size**-0.5

    # We simulate a single-rank scenario
    num_ranks = 1
    NUM_BLOCKS_PER_RANK = 128 * 12 + 1
    NUM_BLOCKS = NUM_BLOCKS_PER_RANK * num_ranks

    # --- Data Generation (all local) ---
    # These tensors will now be created on the default device set above.
    query = torch.randn(num_seqs,
                        num_query_heads,
                        head_size,
                        dtype=dtype) / 10

    key_value_cache = torch.randn(NUM_BLOCKS,
                                  2,
                                  block_size,
                                  num_kv_heads,
                                  head_size,
                                  dtype=dtype) / 10
                                  
    key_cache = key_value_cache[:, 0, :, :, :].contiguous()
    value_cache = key_value_cache[:, 1, :, :, :].contiguous()

    max_num_blocks_per_seq = NUM_BLOCKS_PER_RANK
    block_tables = torch.randint(0,
                                 NUM_BLOCKS_PER_RANK,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)

    # In a single-process run, the global length is just the local length
    global_kv_lens = kv_lens_per_rank

    # --- Execute the Reference Function ---
    ref_output = ref_paged_attn(query=query,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                query_lens=[1] * num_seqs,
                                kv_lens_per_rank=global_kv_lens,
                                block_tables=block_tables,
                                scale=scale,
                                soft_cap=soft_cap)

    # --- Print the Result ---
    print("\n--- Golden Reference Output ---")
    print(f"Shape: {ref_output.shape}")
    print(
        "A small slice of the result (first sequence, first head, first 10 values):"
    )
    print(ref_output[0, 0, :10].cpu())
    print("-" * 45)


if __name__ == "__main__":
    calculate_and_print_reference()