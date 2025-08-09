# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import iris
import torch.distributed as dist

# This file implements the pipelined model with the user-specified
# (source, dest, tile) synchronization flag system.

# --- KERNEL 1: The Producer (MODIFIED) ---
@triton.jit
def mock_producer_kernel(
    incoming_data_buffer,
    tile_sync_flags,
    heap_bases,
    M, N,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    
    for tile_id in range(pid, total_tiles, NUM_SMS):
        dummy_tile = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), cur_rank + 1.0, dtype=tl.float16)
        
        rm = tl.arange(0, BLOCK_SIZE_M)
        rn = tl.arange(0, BLOCK_SIZE_N)
        # Data layout for incoming tiles remains (source, tile)
        tile_offset = (cur_rank * total_tiles + tile_id) * BLOCK_SIZE_M * BLOCK_SIZE_N
        dest_ptr_tensor = incoming_data_buffer + tile_offset + rm[:, None] * BLOCK_SIZE_N + rn[None, :]

        # --- MODIFIED: Signal is now INSIDE the broadcast loop ---
        for remote_rank in range(world_size):
            # 1. Send the data to the specific remote_rank
            iris.store(dest_ptr_tensor, dummy_tile, cur_rank, remote_rank, heap_bases)
            
            # 2. Immediately signal that THIS specific transfer is done.
            #    The flag index is now 3D: [source][dest][tile]
            flag_offset = cur_rank * (world_size * total_tiles) + \
                          remote_rank * total_tiles + \
                          tile_id
            flag_address = tile_sync_flags + flag_offset
            tl.atomic_xchg(flag_address, 1, sem="release", scope="sys")


# --- KERNEL 2: The Consumer (MODIFIED) ---
@triton.jit
def pipelined_consumer_gemm(
    A, B, C,
    incoming_data_buffer,
    tile_sync_flags,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr, EVEN_K: tl.constexpr,
    heap_bases: tl.tensor, cur_rank: tl.constexpr, world_size: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUM_XCDS > 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        # ... Full Local GEMM logic (unchanged) ...
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        rk_init = tl.arange(0, BLOCK_SIZE_K)
        A_BASE = A + rm[:, None] * stride_am + rk_init[None, :] * stride_ak
        B_BASE = B + rk_init[:, None] * stride_bk + rn[None, :] * stride_bn
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K: loop_k -= 1
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk
        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        for source_rank in range(world_size):
            # --- MODIFIED: Calculate the 3D flag address for the data we need ---
            # We are consumer `cur_rank`, waiting for data from `source_rank`
            flag_offset = source_rank * (world_size * total_tiles) + \
                          cur_rank * total_tiles + \
                          tile_id
            flag_address = tile_sync_flags + flag_offset
            
            while tl.atomic_cas(flag_address, 1, 1, sem="acquire", scope="sys") == 0:
                pass
            
            # Data loading logic is unchanged
            rm_tile = tl.arange(0, BLOCK_SIZE_M)
            rn_tile = tl.arange(0, BLOCK_SIZE_N)
            tile_offset = (source_rank * total_tiles + tile_id) * BLOCK_SIZE_M * BLOCK_SIZE_N
            incoming_ptr = incoming_data_buffer + tile_offset + rm_tile[:, None] * BLOCK_SIZE_N + rn_tile[None, :]
            incoming_mask = (rm[:, None] < M) & (rn[None, :] < N)
            incoming_tile = tl.load(incoming_ptr, mask=incoming_mask, other=0.0)
            acc += incoming_tile.to(acc_dtype)
            
        c = acc.to(C.type.element_ty)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_ptr, c, mask=c_mask)


# --- Host class to run the kernels (MODIFIED) ---
class PipelinedGemm(torch.nn.Module):
    def __init__(self, iris_instance, TP):
        super().__init__()
        self.iris = iris_instance
        self.rank = iris_instance.get_rank()
        self.world_size = TP
        self.BLOCK_SIZE_M = 128
        self.BLOCK_SIZE_N = 128
        self.BLOCK_SIZE_K = 128
        self.GROUP_SIZE_M = 1
        self.NUM_SMS = 288
        self.NUM_XCDS = 1
        self.incoming_data_buffer = None
        self.tile_sync_flags = None
        
    def forward(self, A, B):
        M, K = A.shape
        _K, N = B.shape
        assert K == _K
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        
        total_tiles = (M // self.BLOCK_SIZE_M) * (N // self.BLOCK_SIZE_N)
        if total_tiles == 0: total_tiles = 1
        
        buffer_shape = (self.world_size * total_tiles * self.BLOCK_SIZE_M * self.BLOCK_SIZE_N,)
        if self.incoming_data_buffer is None or self.incoming_data_buffer.numel() != buffer_shape[0]:
            self.incoming_data_buffer = self.iris.empty(buffer_shape, dtype=A.dtype)
        
        # --- MODIFIED: Allocate the larger 3D flag buffer ---
        sync_flag_shape = (self.world_size * self.world_size * total_tiles,)
        if self.tile_sync_flags is None or self.tile_sync_flags.numel() != sync_flag_shape[0]:
            self.tile_sync_flags = self.iris.zeros(sync_flag_shape, dtype=torch.int32)

        self.tile_sync_flags.zero_()
        self.iris.barrier()

        # Split GPU resources to prevent potential deadlocks
        producer_sms = self.NUM_SMS // 2
        consumer_sms = self.NUM_SMS - producer_sms
        producer_grid = (producer_sms,)
        consumer_grid = (consumer_sms,)

        mock_producer_kernel[producer_grid](
            self.incoming_data_buffer, self.tile_sync_flags, self.iris.get_heap_bases(),
            M, N,
            self.rank, self.world_size,
            self.BLOCK_SIZE_M, self.BLOCK_SIZE_N, producer_sms
        )

        pipelined_consumer_gemm[consumer_grid](
            A, B, C,
            self.incoming_data_buffer, self.tile_sync_flags,
            M, N, K,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
            self.BLOCK_SIZE_M, self.BLOCK_SIZE_N, self.BLOCK_SIZE_K,
            self.GROUP_SIZE_M, consumer_sms, self.NUM_XCDS,
            (K % self.BLOCK_SIZE_K == 0),
            self.iris.get_heap_bases(), self.rank, self.world_size,
        )
        return C