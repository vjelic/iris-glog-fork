# SPDX-License-Identifier: MIT
# Copyright (c) 2025
# Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import iris

# ============================================================
# Utils
# ============================================================

def ceil_div(x, y):  # host-side ceil-div
    return (x + y - 1) // y


# ============================================================
# Stage 0: Mock previous GEMM to produce act_local (B, D_local)
#   act_local = A_prev @ W_prev_local
#   A_prev: (B, K_prev), W_prev_local: (K_prev, D_local)
# ============================================================

@triton.jit
def _tile_coords(tile_id, M, N, BM: tl.constexpr, BN: tl.constexpr, GM: tl.constexpr):
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    num_in_group = GM * num_n
    group_id = tile_id // num_in_group
    first_m = group_id * GM
    group_m = tl.minimum(num_m - first_m, GM)
    t = tile_id % num_in_group
    pid_m = first_m + (t % group_m)
    pid_n = t // group_m
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    return rm, rn

@triton.jit
def mock_prev_gemm_kernel(
    A_prev, W_prev, act_local,
    M, N, K,
    sa_m, sa_k, sw_k, sw_n, so_m, so_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GM: tl.constexpr, NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_total = tl.cdiv(M, BM) * tl.cdiv(N, BN)

    for tile in range(pid, tiles_total, NUM_SMS):
        rm, rn = _tile_coords(tile, M, N, BM, BN, GM)
        rk = tl.arange(0, BK)

        acc = tl.zeros((BM, BN), dtype=tl.float32)

        # main K
        loops = tl.cdiv(K, BK)
        for i in range(loops):
            k = i * BK + rk
            Ap = A_prev + rm[:, None] * sa_m + k[None, :] * sa_k
            Wp = W_prev + k[:, None] * sw_k + rn[None, :] * sw_n
            a = tl.load(Ap, mask=(rm[:, None] < M) & (k[None, :] < K), other=0.0)
            b = tl.load(Wp, mask=(k[:, None] < K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(a, b)

        # store (cast once)
        Cptr = act_local + rm[:, None] * so_m + rn[None, :] * so_n
        tl.store(Cptr, acc.to(act_local.type.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


# ============================================================
# Stage 1: Explicit All-Gather (IRIS)
#   Each source rank writes its local shard into every dest rank's act_global
#   layout: act_global[B, D], with source shard at cols [s*D_local : (s+1)*D_local)
#   Signaling: flags[dest, source] = 1 (system-scoped)
# ============================================================

@triton.jit
def ag_push_local_shard(
    local_act_ptr,        # (B, D_local) fp16
    dest_act_global_ptr,  # base ptr to *dest's* (B, D) fp16
    flags_ptr,            # (world, world) int32
    heap_bases,
    B, D_local, D,
    sa_m, sa_n, sd_m, sd_n,
    my_rank: tl.constexpr, world_size: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    dest_rank = tl.program_id(0)  # one program per destination rank

    # We tile over (B, D_local)
    tiles_m = tl.cdiv(B, BM)
    tiles_n = tl.cdiv(D_local, BN)
    tiles_total = tiles_m * tiles_n

    # Flattened tile id for this program (serial copy)
    for tid in range(0, tiles_total):
        pid_m = tid % tiles_m
        pid_n = tid // tiles_m

        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)

        # Source tile
        s_ptr = local_act_ptr + rm[:, None] * sa_m + rn[None, :] * sa_n
        s_mask = (rm[:, None] < B) & (rn[None, :] < D_local)
        tile = tl.load(s_ptr, mask=s_mask, other=0.0)

        # Dest column start for this source
        col0 = my_rank * D_local
        d_ptr = dest_act_global_ptr + rm[:, None] * sd_m + (col0 + rn)[None, :] * sd_n

        # Cross-GPU store of this tile into dest_rank memory
        iris.store(d_ptr, tile, my_rank, dest_rank, heap_bases, mask=s_mask)

    # Signal (dest, source) flag on dest GPU
    flag = flags_ptr + dest_rank * world_size + my_rank
    iris.atomic_xchg(flag, 1, my_rank, dest_rank, heap_bases, sem="release", scope="sys")


@triton.jit
def ag_wait_all_sources(flags_ptr, my_rank: tl.constexpr, world_size: tl.constexpr):
    # Single-threaded spin: wait until every source has set flags[my_rank, src] = 1
    for src in range(0, world_size):
        f = flags_ptr + my_rank * world_size + src
        while tl.atomic_cas(f, 0, 0, sem="acquire", scope="sys") == 0:
            pass


# ============================================================
# Stage 2: Post-AG GEMM  act_global(B, D) @ W_up_local(D, F_local) -> out(B, F_local)
# ============================================================

@triton.jit
def post_ag_gemm_kernel(
    A, Bm, C,
    M, N, K,
    sa_m, sa_k, sb_k, sb_n, sc_m, sc_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GM: tl.constexpr, NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_total = tl.cdiv(M, BM) * tl.cdiv(N, BN)

    for tile in range(pid, tiles_total, NUM_SMS):
        rm, rn = _tile_coords(tile, M, N, BM, BN, GM)
        rk = tl.arange(0, BK)

        acc = tl.zeros((BM, BN), dtype=tl.float32)

        loops = tl.cdiv(K, BK)
        for i in range(loops):
            k = i * BK + rk
            Ap = A + rm[:, None] * sa_m + k[None, :] * sa_k
            Bp = Bm + k[:, None] * sb_k + rn[None, :] * sb_n
            a = tl.load(Ap, mask=(rm[:, None] < M) & (k[None, :] < K), other=0.0)
            b = tl.load(Bp, mask=(k[:, None] < K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(a, b)

        Cptr = C + rm[:, None] * sc_m + rn[None, :] * sc_n
        tl.store(Cptr, acc.to(C.type.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


# ============================================================
# Host module: AGThenGemm
# ============================================================

class AGThenGemm(torch.nn.Module):
    """
    Stage-0: act_local = A_prev @ W_prev_local      (B, K_prev) @ (K_prev, D_local) -> (B, D_local)
    Stage-1: explicit all_gather via IRIS to act_global (B, D)
             (wait for all sources to finish before proceeding)
    Stage-2: out = act_global @ W_up_local          (B, D) @ (D, F_local) -> (B, F_local)
    """

    def __init__(self, iris_instance, B, D, F, TP, K_prev,
                 dtype=torch.float16,
                 bm=128, bn=128, bk=128, group_m=1, num_sms=288):
        super().__init__()
        self.iris = iris_instance
        self.rank = iris_instance.get_rank()
        self.world = iris_instance.get_num_ranks()
        assert self.world == TP, f"IRIS world {self.world} != TP {TP}"

        self.B = B
        self.D = D
        self.F = F
        self.TP = TP
        self.D_local = D // TP
        self.F_local = F // TP
        self.K_prev = K_prev
        self.dtype = dtype

        # Tiling
        self.BM = bm
        self.BN = bn
        self.BK = bk
        self.GM = group_m
        self.NUM_SMS = num_sms

        # Flags buffer (symmetric heap)
        self.flags = self.iris.zeros((self.world, self.world), dtype=torch.int32)  # [dest, source]

    def clear_flags(self):
        self.flags.zero_()

    # ----- Public forward -----
    def forward(self, A_prev, W_prev_local, W_up_local):
        """
        A_prev:        (B, K_prev)
        W_prev_local:  (K_prev, D_local)     -- shard that produces act_local
        W_up_local:    (D, F_local)          -- shard used after AG
        Returns out:   (B, F_local)
        """
        B, K_prev = A_prev.shape
        assert B == self.B and K_prev == self.K_prev
        assert W_prev_local.shape == (K_prev, self.D_local)
        assert W_up_local.shape == (self.D, self.F_local)

        # Allocate in IRIS heap for cross-GPU access
        act_local  = self.iris.empty((B, self.D_local), dtype=self.dtype)
        act_global = self.iris.empty((B, self.D),       dtype=self.dtype)
        out = torch.empty((B, self.F_local), device=A_prev.device, dtype=self.dtype)

        # -------- Stage 0: previous GEMM -> act_local --------
        mock_prev_gemm_kernel[(self.NUM_SMS,)](
            A_prev, W_prev_local, act_local,
            B, self.D_local, self.K_prev,
            A_prev.stride(0), A_prev.stride(1),
            W_prev_local.stride(0), W_prev_local.stride(1),
            act_local.stride(0), act_local.stride(1),
            BM=self.BM, BN=self.BN, BK=self.BK,
            GM=self.GM, NUM_SMS=self.NUM_SMS
        )

        # Ensure all ranks finish Stage 0
        self.iris.barrier()
        self.clear_flags()

        # -------- Stage 1: explicit All-Gather into act_global --------
        # Each src pushes its local shard into every dest's act_global
        ag_grid = (self.world,)  # one program per destination rank
        ag_push_local_shard[ag_grid](
            act_local,
            act_global,               # base pointer (this rank will be "dest" when dest_rank == my_rank)
            self.flags,
            self.iris.get_heap_bases(),
            B, self.D_local, self.D,
            act_local.stride(0), act_local.stride(1),
            act_global.stride(0), act_global.stride(1),
            my_rank=self.rank, world_size=self.world,
            BM=128, BN=128  # copy tile sizes (independent from GEMM tiles)
        )

        # Wait (on device) until all sources have written to *this* rank
        ag_wait_all_sources[(1,)](self.flags, self.rank, self.world)

        # (Optional) rank barrier for good measure
        self.iris.barrier()

        # -------- Stage 2: post-AG GEMM -> out --------
        post_ag_gemm_kernel[(self.NUM_SMS,)](
            act_global, W_up_local, out,
            B, self.F_local, self.D,
            act_global.stride(0), act_global.stride(1),
            W_up_local.stride(0), W_up_local.stride(1),
            out.stride(0), out.stride(1),
            BM=self.BM, BN=self.BN, BK=self.BK,
            GM=self.GM, NUM_SMS=self.NUM_SMS
        )

        return out
