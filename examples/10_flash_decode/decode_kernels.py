import torch
import triton
import math
import os
import triton.language as tl
from triton.language.extra import libdevice
import iris

def gqa_fwd_batch_decode_intra_rank(q, k_cache, v_cache, workspace, q_lens, kv_lens, block_table, scale, soft_cap=0.0,
                                    output_split=None, output_combine=None, kv_split=-1):
    batch, q_heads, q_head_dim = q.shape
    _, page_size, kv_heads, k_head_dim = k_cache.shape
    assert page_size == v_cache.shape[1] and kv_heads == v_cache.shape[2] and k_head_dim == q_head_dim
    v_head_dim = v_cache.shape[-1]

    BLOCK_N = 64
    BLOCK_HEAD_DIM = 2**int(math.log2(q_head_dim))
    BLOCK_DPE = q_head_dim - BLOCK_HEAD_DIM
    BLOCK_DV = triton.next_power_of_2(v_head_dim)

    kv_group_num = q_heads // kv_heads
    assert q_heads % kv_heads == 0

    BLOCK_H = 16
    NUM_KV_SPLITS = 32 if kv_split == -1 else kv_split

    grid_split_kv = (batch, triton.cdiv(q_heads, min(BLOCK_H, kv_group_num)), NUM_KV_SPLITS)

    output_split = torch.empty([batch, q_heads, NUM_KV_SPLITS, v_head_dim +
                                1], dtype=q.dtype, device=q.device) if output_split is None else output_split
    output_combine = torch.empty([batch, q_heads, v_head_dim +
                                  1], dtype=q.dtype, device=q.device) if output_combine is None else output_combine

    kernel_gqa_fwd_batch_decode_split_kv[grid_split_kv](
        q,
        k_cache,
        v_cache,
        output_split,
        scale,
        block_table,
        kv_lens,
        # shape
        batch,
        # strides
        q.stride(0),
        q.stride(1),
        k_cache.stride(-3),
        k_cache.stride(-2),
        v_cache.stride(-3),
        v_cache.stride(-2),
        output_split.stride(0),
        output_split.stride(1),
        output_split.stride(2),
        block_table.stride(0),
        # constants
        kv_group_num,
        q_heads,
        BLOCK_HEAD_DIM,
        BLOCK_DPE,
        BLOCK_DV,
        BLOCK_N,
        BLOCK_H,
        NUM_KV_SPLITS,
        page_size,
        soft_cap,
        k_head_dim,
        v_head_dim,
        num_warps=4,
        num_stages=2,
    )

    kernel_intra_rank_gqa_fwd_batch_decode_combine_kv[(batch, q_heads)](
        output_split,
        output_combine,
        kv_lens,
        batch,
        q_heads,
        output_split.stride(0),
        output_split.stride(1),
        output_split.stride(2),
        output_combine.stride(0),
        output_combine.stride(1),
        NUM_KV_SPLITS,
        BLOCK_DV,
        v_head_dim,
        num_warps=4,
        num_stages=2,
    )

    return output_combine

def gqa_fwd_batch_decode_intra_rank_fused(
    q, k_cache, v_cache,
    
    gathered_buffer, signal_flags, iris_instance,

    q_lens, kv_lens, block_table, scale, soft_cap=0.0,
    output_split=None, kv_split=-1
):
    batch, q_heads, q_head_dim = q.shape
    _, page_size, kv_heads, k_head_dim = k_cache.shape
    v_head_dim = v_cache.shape[-1]
    rank = iris_instance.get_rank()
    num_ranks = iris_instance.get_num_ranks()

    BLOCK_N = 64
    BLOCK_HEAD_DIM = 2**int(math.log2(q_head_dim))
    BLOCK_DPE = q_head_dim - BLOCK_HEAD_DIM
    BLOCK_DV = triton.next_power_of_2(v_head_dim)
    kv_group_num = q_heads // kv_heads
    BLOCK_H = 16
    NUM_KV_SPLITS = 32 if kv_split == -1 else kv_split

    # Split-K calculation (same as before)
    grid_split_kv = (batch, triton.cdiv(q_heads, min(BLOCK_H, kv_group_num)), NUM_KV_SPLITS)
    if output_split is None:
        output_split = torch.empty(
            [batch, q_heads, NUM_KV_SPLITS, v_head_dim + 1], dtype=q.dtype, device=q.device
        )

    kernel_gqa_fwd_batch_decode_split_kv[grid_split_kv](
        q, k_cache, v_cache, output_split, scale, block_table, kv_lens,
        batch, q.stride(0), q.stride(1),
        k_cache.stride(-3), k_cache.stride(-2), v_cache.stride(-3), v_cache.stride(-2),
        output_split.stride(0), output_split.stride(1), output_split.stride(2),
        block_table.stride(0), kv_group_num, q_heads,
        BLOCK_HEAD_DIM, BLOCK_DPE, BLOCK_DV, BLOCK_N, BLOCK_H, NUM_KV_SPLITS,
        page_size, soft_cap, k_head_dim, v_head_dim,
        num_warps=4, num_stages=2
    )

    # Fused Intra-Rank Combine (same as before) but this time with Intra-Node Push instead of writing the data
    grid_combine_push = (batch, q_heads)
    kernel_intra_rank_gqa_fwd_batch_decode_combine_kv_fused[grid_combine_push](
        output_split,
        kv_lens,
        gathered_buffer,
        signal_flags,
        iris_instance.get_heap_bases(),
        output_split.stride(0), output_split.stride(1), output_split.stride(2),
        gathered_buffer.stride(0), gathered_buffer.stride(1), gathered_buffer.stride(2),
        rank,
        num_ranks,
        NUM_KV_SPLITS,
        BLOCK_DV,
        v_head_dim,
    )

@triton.jit
def kernel_intra_rank_gqa_fwd_batch_decode_combine_kv_fused(
    # Input
    Mid_O,
    B_Seqlen,

    gathered_output_ptr,
    signal_flags_ptr,
    heap_bases_ptr,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    stride_gathered_rank, stride_gathered_bs, stride_gathered_h,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # Same as normal kernel
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v_base = cur_batch * stride_mid_ob + cur_head * stride_mid_oh
    
    for split_kv_id in range(0, NUM_KV_SPLITS):
        split_kv_start = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS) * split_kv_id
        if split_kv_start < cur_batch_seq_len:
            offs_v = offs_v_base + split_kv_id * stride_mid_os + offs_d
            offs_logic = offs_v_base + split_kv_id * stride_mid_os + Lv
            
            tv = tl.load(Mid_O + offs_v, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O + offs_logic)
            
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = libdevice.fast_expf(e_max - n_e_max)
            exp_logic = libdevice.fast_expf(tlogic - n_e_max)
            
            acc = acc * old_scale + exp_logic * tv
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    final_v = acc / e_sum
    final_logic = e_max + tl.log(e_sum)
    final_v = tl.where(e_sum == 0.0, 0.0, final_v)

    # We no longer write to local memory, but we directly write to all other GPUs
    base_write_ptr = (gathered_output_ptr +
                      my_rank * stride_gathered_rank +
                      cur_batch * stride_gathered_bs +
                      cur_head * stride_gathered_h)
    
    for dest_rank_id in range(0, world_size):
        # Write output and lv
        iris.store(base_write_ptr + offs_d, final_v, my_rank, dest_rank_id, heap_bases_ptr, mask=mask_d)
        iris.store(base_write_ptr + Lv, final_logic, my_rank, dest_rank_id, heap_bases_ptr)
        
        # Signal the dest rank (we add 1, but here is for each cur_batch/head
        flag_index = dest_rank_id * world_size + my_rank
        iris.atomic_add(signal_flags_ptr + flag_index, 1, my_rank, dest_rank_id, heap_bases_ptr, sem="release", scope="sys")

def gqa_fwd_batch_decode(q, k_cache, v_cache, workspace, q_lens, kv_lens, block_table, scale, soft_cap=0.0,
                         output_split=None, output_combine=None, kv_split=-1):
    batch, q_heads, q_head_dim = q.shape
    _, page_size, kv_heads, k_head_dim = k_cache.shape
    assert page_size == v_cache.shape[1] and kv_heads == v_cache.shape[2] and k_head_dim == q_head_dim
    v_head_dim = v_cache.shape[-1]

    BLOCK_N = 64
    BLOCK_HEAD_DIM = 2**int(math.log2(q_head_dim))
    BLOCK_DPE = q_head_dim - BLOCK_HEAD_DIM
    BLOCK_DV = triton.next_power_of_2(v_head_dim)

    kv_group_num = q_heads // kv_heads
    assert q_heads % kv_heads == 0

    BLOCK_H = 16
    NUM_KV_SPLITS = 32 if kv_split == -1 else kv_split

    grid_split_kv = (batch, triton.cdiv(q_heads, min(BLOCK_H, kv_group_num)), NUM_KV_SPLITS)

    output_split = torch.empty([batch, q_heads, NUM_KV_SPLITS, v_head_dim +
                                1], dtype=torch.float16, device=q.device) if output_split is None else output_split
    output_combine = torch.empty([batch, q_heads, v_head_dim], dtype=torch.float16,
                                 device=q.device) if output_combine is None else output_combine

    kernel_gqa_fwd_batch_decode_split_kv[grid_split_kv](
        q,
        k_cache,
        v_cache,
        output_split,
        scale,
        block_table,
        kv_lens,
        # shape
        batch,
        # strides
        q.stride(0),
        q.stride(1),
        k_cache.stride(-3),
        k_cache.stride(-2),
        v_cache.stride(-3),
        v_cache.stride(-2),
        output_split.stride(0),
        output_split.stride(1),
        output_split.stride(2),
        block_table.stride(0),
        # constants
        kv_group_num,
        q_heads,
        BLOCK_HEAD_DIM,
        BLOCK_DPE,
        BLOCK_DV,
        BLOCK_N,
        BLOCK_H,
        NUM_KV_SPLITS,
        page_size,
        soft_cap,
        k_head_dim,
        v_head_dim,
        num_warps=4,
        num_stages=2,
    )

    kernel_gqa_fwd_batch_decode_combine_kv[(batch, q_heads)](
        output_split,
        output_combine,
        kv_lens,
        batch,
        q_heads,
        output_split.stride(0),
        output_split.stride(1),
        output_split.stride(2),
        output_combine.stride(0),
        output_combine.stride(1),
        NUM_KV_SPLITS,
        BLOCK_DV,
        v_head_dim,
        num_warps=4,
        num_stages=2,
    )

    return output_combine


@triton.jit
def kernel_gqa_fwd_batch_decode_split_kv(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    output_ptr,
    sm_scale,
    block_table_ptr,
    kv_length_ptr,
    # shape
    batch,
    # strides
    stride_q_bs,
    stride_q_h,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_o_bs,
    stride_o_h,
    stride_o_split,
    stride_table_bs,
    # constants
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    soft_cap: tl.constexpr,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kv_hid = hid // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if kv_group_num > BLOCK_H:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num

    cur_head = hid * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (hid + 1) * VALID_BLOCK_H) & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_HEAD_DIM)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < K_DIM
    mask_dv = offs_dv < V_DIM
    cur_kv_seq_len = tl.load(kv_length_ptr + bid)

    offs_q = bid * stride_q_bs + cur_head[:, None] * stride_q_h + offs_d[None, :]
    q = tl.load(q_ptr + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_HEAD_DIM + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < K_DIM
        offs_qpe = bid * stride_q_bs + cur_head[:, None] * stride_q_h + offs_dpe[:, None]
        qpe = tl.load(q_ptr + offs_qpe, mask=mask_h[:, None] & mask_dpe[None, :], other=0.0)

    kv_len_per_split = tl.cdiv(cur_kv_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_kv_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_page_number = tl.load(block_table_ptr + bid * stride_table_bs + offs_n // PAGE_SIZE, mask=offs_n
                                 < split_kv_end, other=0)
        kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
        offs_cache_k = kv_loc[None, :] * stride_k_cache_bs + kv_hid * stride_k_cache_h + offs_d[:, None]
        k = tl.load(k_cache_ptr + offs_cache_k, mask=(offs_n[None, :] < split_kv_end) & mask_d[:, None], other=0.0)
        qk = tl.dot(q, k.to(q.dtype))

        if BLOCK_DPE > 0:
            offs_cache_kpe = kv_loc[None, :] * stride_k_cache_bs + kv_hid * stride_k_cache_h + offs_dpe[:, None]
            kpe = tl.load(k_cache_ptr + offs_cache_kpe, mask=(offs_n[None, :] < split_kv_end)
                          & mask_dpe[:, None], other=0.0)
            qk += tl.dot(qpe, kpe.to(qpe.dtype))

        qk *= sm_scale

        if soft_cap > 0:
            qk = soft_cap * tanh(qk / soft_cap)

        qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf"))

        offs_cache_v = kv_loc[:, None] * stride_v_cache_bs + kv_hid * stride_v_cache_h + offs_dv[None, :]
        v = tl.load(v_cache_ptr + offs_cache_v, mask=(offs_n[:, None] < split_kv_end) & mask_dv[None, :], other=0.0)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = libdevice.fast_expf(e_max - n_e_max)
        p = libdevice.fast_expf(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    offs_out = bid * stride_o_bs + cur_head[:, None] * stride_o_h + split_kv_id * stride_o_split + offs_dv[None, :]
    tl.store(output_ptr + offs_out, acc / e_sum[:, None], mask=mask_h[:, None] & mask_dv[None, :])

    offs_log = bid * stride_o_bs + cur_head * stride_o_h + split_kv_id * stride_o_split + V_DIM
    tl.store(output_ptr + offs_log, e_max + tl.log(e_sum), mask=mask_h)

@triton.jit
def kernel_gqa_fwd_batch_decode_combine_kv(
    Mid_O,
    o,
    B_Seqlen,
    batch,
    q_heads,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = libdevice.fast_expf(e_max - n_e_max)
            acc *= old_scale
            exp_logic = libdevice.fast_expf(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )

@triton.jit
def kernel_intra_rank_gqa_fwd_batch_decode_combine_kv(
    Mid_O,
    o,
    B_Seqlen,
    batch,
    q_heads,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = libdevice.fast_expf(e_max - n_e_max)
            acc *= old_scale
            exp_logic = libdevice.fast_expf(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )
    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + Lv,
        e_max + tl.log(e_sum),
    )

@triton.jit
def kernel_inter_rank_gqa_fwd_batch_decode_combine_kv(
    Mid_O,
    o,
    B_Seqlens,
    batch,
    q_heads,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len_ptr = B_Seqlens + cur_batch

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        effective_kv_len = tl.load(cur_batch_seq_len_ptr + split_kv_id * batch)

        if effective_kv_len > 0:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = libdevice.fast_expf(e_max - n_e_max)
            acc *= old_scale
            exp_logic = libdevice.fast_expf(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )

@triton.jit
def kernel_fused_wait_and_combine_fused(
    All_Ranks_Mid_O,
    o,
    B_Seqlens,
    signal_flags_ptr,
    batch,
    q_heads,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    my_rank: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    total_arrivals_needed = batch * q_heads

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    cur_batch_seq_len_ptr = B_Seqlens + cur_batch

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    
    for source_rank_id in range(0, NUM_KV_SPLITS):
        
        flag_index = my_rank * NUM_KV_SPLITS + source_rank_id
        # while tl.load(signal_flags_ptr + flag_index, cache_modifier=".ca") <= iteration_id:
        #     pass
        while tl.atomic_cas(signal_flags_ptr + flag_index, 0, 0, sem="acquire") < total_arrivals_needed:
            pass
        
        # tl.atomic_xchg(signal_flags_ptr + flag_index, 0, sem="release")

        effective_kv_len = tl.load(cur_batch_seq_len_ptr + source_rank_id * batch)
        
        if effective_kv_len > 0:
            base_ptr = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + source_rank_id * stride_mid_os
            offs_v = base_ptr + offs_d
            offs_logic = base_ptr + Lv

            tv = tl.load(All_Ranks_Mid_O + offs_v, mask=mask_d, other=0.0)
            tlogic = tl.load(All_Ranks_Mid_O + offs_logic)
            
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = libdevice.fast_expf(e_max - n_e_max)
            acc *= old_scale
            
            exp_logic = libdevice.fast_expf(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    final_out = acc / e_sum
    final_out = tl.where(e_sum == 0, 0.0, final_out)
    
    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        final_out,
        mask=mask_d,
    )
    
    
@triton.jit
def kernel_fused_wait_and_combine(
    All_Ranks_Mid_O,
    o,
    B_Seqlens,
    signal_flags_ptr,
    batch,
    q_heads,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    my_rank: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    total_arrivals_needed = batch * q_heads

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    cur_batch_seq_len_ptr = B_Seqlens + cur_batch

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    
    for source_rank_id in range(0, NUM_KV_SPLITS):
        
        flag_index = my_rank * NUM_KV_SPLITS + source_rank_id
        # while tl.load(signal_flags_ptr + flag_index, cache_modifier=".ca") <= iteration_id:
        #     pass
        while tl.atomic_cas(signal_flags_ptr + flag_index, 0, 0, sem="acquire") == 0:
            pass
        
        # tl.atomic_xchg(signal_flags_ptr + flag_index, 0, sem="release")

        effective_kv_len = tl.load(cur_batch_seq_len_ptr + source_rank_id * batch)
        
        if effective_kv_len > 0:
            base_ptr = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + source_rank_id * stride_mid_os
            offs_v = base_ptr + offs_d
            offs_logic = base_ptr + Lv

            tv = tl.load(All_Ranks_Mid_O + offs_v, mask=mask_d, other=0.0)
            tlogic = tl.load(All_Ranks_Mid_O + offs_logic)
            
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = libdevice.fast_expf(e_max - n_e_max)
            acc *= old_scale
            
            exp_logic = libdevice.fast_expf(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    final_out = acc / e_sum
    final_out = tl.where(e_sum == 0, 0.0, final_out)
    
    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        final_out,
        mask=mask_d,
    )