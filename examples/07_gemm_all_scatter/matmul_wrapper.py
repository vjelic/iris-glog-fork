# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

# from streamk_kernel import streamk_gemm
from gemm_all_scatter import persistent_gemm_all_scatter
from examples.common.utils import is_triton_interpret_set

gemm_kernel = persistent_gemm_all_scatter


class matmul(torch.autograd.Function):
    _debug = False
    _registers = None
    _spills = None

    @staticmethod
    def set_debug(debug: bool):
        matmul._debug = debug

    @staticmethod
    def get_matmul_registers():
        if matmul._debug:
            return matmul._registers
        else:
            raise RuntimeError("Debug mode is not enabled. Call set_debug(True) first.")

    @staticmethod
    def get_matmul_spills():
        if matmul._debug:
            return matmul._spills
        else:
            raise RuntimeError("Debug mode is not enabled. Call set_debug(True) first.")

    @staticmethod
    def _call(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        c_global: torch.Tensor,
        bias: torch.Tensor,
        rank: int,
        world_size: int,
        num_sms: int,
        BLK_M: int,
        BLK_N: int,
        BLK_K: int,
        gsize_m: int,
        heap_bases_ptr: torch.Tensor = None,
        arch: str = "gfx942",
        COLLECT_TIMESTAMPS: bool = False,
        mm_begin_timestamp: torch.Tensor = None,
        mm_end_timestamp: torch.Tensor = None,
    ):
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape

        num_xcds = 1
        if arch == "gfx942" or arch == "gfx950":
            num_xcds = 8

        # TODO: Use arch-specific values.
        num_stages = 2
        num_warps = 8
        waves_per_eu = 0
        mfma = 16
        kpack = 1

        total_blocks_M = triton.cdiv(M, BLK_M)
        total_blocks_N = triton.cdiv(N, BLK_N)
        iters_per_tile = triton.cdiv(K, BLK_K)
        total_tiles = total_blocks_M * total_blocks_N
        even_k = K % BLK_K == 0
        use_bias = False

        # compute grid (work to do per SM on the first wave)
        stride_bias = bias.stride(0) if use_bias else 0
        kk = gemm_kernel[(num_sms,)](
            a,
            b,
            c,
            c_global,
            bias,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            c_global.stride(0),
            c_global.stride(1),
            stride_bias,
            BLOCK_SIZE_M=BLK_M,
            BLOCK_SIZE_N=BLK_N,
            BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m,
            NUM_SMS=num_sms,
            NUM_XCDS=num_xcds,
            BIAS=use_bias,
            EVEN_K=even_k,
            num_stages=num_stages,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=mfma,
            kpack=kpack,
            heap_bases=heap_bases_ptr,
            cur_rank=rank,
            world_size=world_size,
            COLLECT_TIMESTAMPS=COLLECT_TIMESTAMPS,
            mm_begin_timestamp_ptr=mm_begin_timestamp,
            mm_end_timestamp_ptr=mm_end_timestamp,
        )

        if matmul._debug and not is_triton_interpret_set():
            matmul._registers = kk.n_regs
            matmul._spills = kk.n_spills

        return c

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        c_global: torch.Tensor,
        bias: torch.Tensor,
        rank: int,
        world_size: int,
        num_sms: int,
        BLK_M: int,
        BLK_N: int,
        BLK_K: int,
        gsize_m: int,
        heap_bases_ptr: torch.Tensor = None,
        arch: str = "gfx942",
        COLLECT_TIMESTAMPS: bool = False,
        mm_begin_timestamp: torch.Tensor = None,
        mm_end_timestamp: torch.Tensor = None,
    ):
        matmul._call(
            a=a,
            b=b,
            c=c,
            c_global=c_global,
            bias=bias,
            rank=rank,
            world_size=world_size,
            num_sms=num_sms,
            BLK_M=BLK_M,
            BLK_N=BLK_N,
            BLK_K=BLK_K,
            gsize_m=gsize_m,
            heap_bases_ptr=heap_bases_ptr,
            arch=arch,
            COLLECT_TIMESTAMPS=COLLECT_TIMESTAMPS,
            mm_begin_timestamp=mm_begin_timestamp,
            mm_end_timestamp=mm_end_timestamp,
        )
        return c
