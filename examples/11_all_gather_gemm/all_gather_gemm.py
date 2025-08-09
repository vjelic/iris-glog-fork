import torch
import triton
import triton.language as tl
import iris

@triton.jit
def tile_id_to_index_range(
    tile_id,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    tile_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (tile_in_group % group_size_m)
    pid_n = tile_in_group // group_size_m

    rm_start = pid_m * BLOCK_SIZE_M
    rn_start = pid_n * BLOCK_SIZE_N

    # clamp to the maximum valid index (M-1, N-1)
    max_m = M - 1
    max_n = N - 1

    # generate indices
    rm = rm_start + tl.arange(0, BLOCK_SIZE_M)
    rn = rn_start + tl.arange(0, BLOCK_SIZE_N)

    rm = tl.minimum(rm, max_m)
    rn = tl.minimum(rn, max_n)

    # rm_mod = rm % M
    # rm = tl.max_contiguous(tl.multiple_of(rm_mod, BLOCK_SIZE_M), BLOCK_SIZE_M)

    return rm, rn, rm_start, rn_start

@triton.jit
def fused_ag_gemm_kernel_no_wait(
    gathered_act_ptr,
    W_ptr,
    C_ptr,
    signal_flags_ptr,
    M, K, N, # Using M, K, N naming convention
    stride_gathered_rank_bytes: tl.constexpr,
    stride_w_k, stride_w_n,
    stride_c_m, stride_c_n,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    K_local: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K_local: tl.constexpr, # K dimension processed per inner loop
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles_total = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    
    # --- MODIFICATION ---
    # The blocking wait loop for the All-Gather has been removed as requested.
    # WARNING: This will cause a race condition and produce incorrect results,
    # as the GEMM will start before data has arrived in `gathered_act_ptr`.
    #
    # current_arrivals = 0
    # while current_arrivals < world_size:
    #     current_arrivals = tl.atomic_cas(signal_flags_ptr + my_rank, -1, -1, sem="acquire", scope="sys")

    # Persistent loop over output tiles
    for tile_idx in range(pid, num_tiles_total, NUM_SMS):
        # Get 2D coordinates for the output tile C
        rm, rn, _, _ = tile_id_to_index_range(tile_idx, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
        
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Loop over data from each source_rank
        for source_rank in range(world_size):
            act_shard_base_ptr = gathered_act_ptr + source_rank * stride_gathered_rank_bytes
            act_shard_ptr_typed = tl.cast(act_shard_base_ptr, tl.pointer_type(C_ptr.dtype.element_ty))
            
            W_slice_ptr = W_ptr + (source_rank * K_local) * stride_w_k

            # Loop over the K_local dimension
            for k_offset in range(0, K_local, BLOCK_SIZE_K_local):
                rk_local = k_offset + tl.arange(0, BLOCK_SIZE_K_local)
                
                # Load activation shard (Matrix A)
                A_ptr = act_shard_ptr_typed + rm[:, None] * K_local + rk_local[None, :]
                A_mask = (rm[:, None] < M) & (rk_local[None, :] < K_local)
                a = tl.load(A_ptr, mask=A_mask, other=0.0)

                # Load weight slice (Matrix B)
                B_ptr = W_slice_ptr + rk_local[:, None] * stride_w_k + rn[None, :] * stride_w_n
                B_mask = (rk_local[:, None] < K_local) & (rn[None, :] < N)
                b = tl.load(B_ptr, mask=B_mask, other=0.0)
                
                acc += tl.dot(a, b)

        # Store the final computed tile
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_tile_ptr = C_ptr + rm[:, None] * stride_c_m + rn[None, :] * stride_c_n
        tl.store(C_tile_ptr, acc.to(C_ptr.dtype.element_ty), mask=c_mask)

@triton.jit
def push_kernel_bytes(
    local_staging_ptr,
    gathered_buffer_ptr,
    signal_flags_ptr,
    heap_bases_ptr,
    bytes_to_send: tl.constexpr,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    stride_gathered_rank_bytes: tl.constexpr,
    BLOCK_SIZE_BYTES: tl.constexpr,
):
    """
    Pushes this rank's local data from a staging buffer to all other ranks.
    """
    dest_rank = tl.program_id(0)
    local_ptr = tl.cast(local_staging_ptr, tl.pointer_type(tl.int8))
    gathered_ptr = tl.cast(gathered_buffer_ptr, tl.pointer_type(tl.int8))
    write_offset = my_rank * stride_gathered_rank_bytes
    
    for B_offset in range(0, bytes_to_send, BLOCK_SIZE_BYTES):
        b_offsets = B_offset + tl.arange(0, BLOCK_SIZE_BYTES)
        b_mask = b_offsets < bytes_to_send
        data_block = tl.load(local_ptr + b_offsets, mask=b_mask, other=0)
        dest_ptr = gathered_ptr + write_offset + b_offsets
        iris.store(dest_ptr, data_block, my_rank, dest_rank, heap_bases_ptr, mask=b_mask)

    iris.atomic_add(signal_flags_ptr + dest_rank, 1, my_rank, dest_rank, heap_bases_ptr, sem="release", scope="sys")

@triton.jit()
def fused_ag_gemm_from_template(
    # --- Signature adapted from scatter kernel ---
    A_gathered, B, C, bias_ptr, signal_flags_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bias,
    stride_gathered_rank_bytes: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K_local: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K_local: tl.constexpr,
    K_local: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    COLLECT_TIMESTAMPS: tl.constexpr = False,
    mm_begin_timestamp_ptr: tl.tensor = None,
    mm_end_timestamp_ptr: tl.tensor = None,
):
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    current_arrivals = 0
    while current_arrivals < world_size:
        current_arrivals = tl.atomic_cas(signal_flags_ptr + cur_rank, -1, -1, sem="acquire", scope="sys")

    for tile_id in range(pid, total_tiles, NUM_SMS):
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_min(mm_begin_timestamp_ptr + tile_id, timestamp)
        
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
        
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for source_rank in range(world_size):
            act_shard_base_ptr = A_gathered + source_rank * stride_gathered_rank_bytes
            act_shard_ptr_typed = tl.cast(act_shard_base_ptr, tl.pointer_type(C.type.element_ty))
            W_slice_ptr = B + (source_rank * K_local) * stride_bk

            loop_k = tl.cdiv(K_local, BLOCK_SIZE_K_local)
            if not EVEN_K_local:
                loop_k -= 1
            
            for k_offset_idx in range(0, loop_k):
                k_offset = k_offset_idx * BLOCK_SIZE_K_local
                rk_local = k_offset + tl.arange(0, BLOCK_SIZE_K_local)
                A_ptr = act_shard_ptr_typed + rm[:, None] * stride_am + rk_local[None, :] * stride_ak
                B_ptr = W_slice_ptr + rk_local[:, None] * stride_bk + rn[None, :] * stride_bn

                # --- FIX IS HERE: Provide a tuple of length 2 for the 2D pointers ---
                a = tl.load(tl.multiple_of(A_ptr, (1, 16)))
                b = tl.load(tl.multiple_of(B_ptr, (16, 1)))
                acc += tl.dot(a, b)
            
            if not EVEN_K_local:
                k_offset = loop_k * BLOCK_SIZE_K_local
                rk_local = k_offset + tl.arange(0, BLOCK_SIZE_K_local)
                A_ptr = act_shard_ptr_typed + rm[:, None] * stride_am + rk_local[None, :] * stride_ak
                B_ptr = W_slice_ptr + rk_local[:, None] * stride_bk + rn[None, :] * stride_bn
                a = tl.load(A_ptr, mask=rk_local[None, :] < K_local, other=0.0)
                b = tl.load(B_ptr, mask=rk_local[:, None] < K_local, other=0.0)
                acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)

        if BIAS:
            bias_val = tl.load(bias_ptr + rn, mask=rn < N)
            c = c + bias_val

        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_ptr, c, mask=c_mask)

        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(mm_end_timestamp_ptr + tile_id, timestamp)


class FusedAGGemm(torch.nn.Module):
    # This class definition is correct and does not need to be changed
    def __init__(self, iris_instance, M, K, N_global, TP, dtype=torch.float16):
        super().__init__()
        self.iris = iris_instance
        self.rank = iris_instance.get_rank()
        self.world_size = iris_instance.get_num_ranks()
        assert K % TP == 0 and N_global % TP == 0
        self.M, self.K, self.N_global = M, K, N_global
        self.K_local = K // TP
        self.N_local = N_global // TP
        self.dtype = dtype
        bytes_per_shard = M * self.K_local * torch.tensor([], dtype=dtype).element_size()
        self.gathered_buffer = self.iris.empty((TP, bytes_per_shard), dtype=torch.int8)
        self.local_staging_buffer = self.iris.empty((bytes_per_shard,), dtype=torch.int8)
        self.signal_flags = self.iris.zeros((TP,), dtype=torch.int32)
        self.BLOCK_SIZE_M = 16
        self.BLOCK_SIZE_N = 64
        self.BLOCK_SIZE_K_local = 64
        self.GROUP_SIZE_M = 8
        self.num_sms = 304
        self.num_xcds = 8
        self.PUSH_BLOCK_BYTES = 32768

    def clear_flags(self):
        self.signal_flags.zero_()

    def forward(self, local_act, local_W, bias=None):
        bytes_to_send = local_act.nbytes
        self.local_staging_buffer[:bytes_to_send].copy_(local_act.flatten().view(torch.int8))
        self.iris.barrier()
        output = torch.empty((self.M, self.N_local), device=local_act.device, dtype=self.dtype)
        
        push_grid = (self.world_size,)
        push_kernel_bytes[push_grid](
            self.local_staging_buffer, self.gathered_buffer, self.signal_flags, self.iris.get_heap_bases(),
            bytes_to_send, my_rank=self.rank, world_size=self.world_size,
            stride_gathered_rank_bytes=self.gathered_buffer.stride(0),
            BLOCK_SIZE_BYTES=self.PUSH_BLOCK_BYTES,
        )

        gemm_grid = (self.num_sms, )
        is_even_k_local = (self.K_local % self.BLOCK_SIZE_K_local) == 0

        fused_ag_gemm_from_template[gemm_grid](
            A_gathered=self.gathered_buffer,
            B=local_W.contiguous(),
            C=output,
            bias_ptr=bias,
            signal_flags_ptr=self.signal_flags,
            M=self.M, N=self.N_local, K=self.K,
            stride_am=self.K_local, stride_ak=1,
            stride_bk=local_W.stride(0), stride_bn=local_W.stride(1),
            stride_cm=output.stride(0), stride_cn=output.stride(1),
            stride_bias=bias.stride(0) if bias is not None else 0,
            stride_gathered_rank_bytes=self.gathered_buffer.stride(0),
            BLOCK_SIZE_M=self.BLOCK_SIZE_M,
            BLOCK_SIZE_N=self.BLOCK_SIZE_N,
            BLOCK_SIZE_K_local=self.BLOCK_SIZE_K_local,
            GROUP_SIZE_M=self.GROUP_SIZE_M,
            NUM_SMS=self.num_sms,
            NUM_XCDS=self.num_xcds,
            BIAS=(bias is not None),
            EVEN_K_local=is_even_k_local,
            K_local=self.K_local,
            heap_bases=self.iris.get_heap_bases(),
            cur_rank=self.rank,
            world_size=self.world_size,
            COLLECT_TIMESTAMPS=False,
            mm_begin_timestamp_ptr=None,
            mm_end_timestamp_ptr=None,
        )
        
        return output