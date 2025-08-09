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
    """
    Converts a 1D tile_id to a 2D index range for a matrix.
   
    """
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
    
    rm = rm_start + tl.arange(0, BLOCK_SIZE_M)
    rn = rn_start + tl.arange(0, BLOCK_SIZE_N)

    return rm, rn, rm_start, rn_start

@triton.jit
def push_kernel_bytes(
    local_staging_ptr,
    gathered_buffer_ptr,
    signal_flags_ptr,
    heap_bases_ptr,
    bytes_to_send,
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
    
    offset = 0
    while offset < bytes_to_send:
        b_offsets = offset + tl.arange(0, BLOCK_SIZE_BYTES)
        b_mask = b_offsets < bytes_to_send
        data_block = tl.load(local_ptr + b_offsets, mask=b_mask, other=0)
        dest_ptr = gathered_ptr + write_offset + b_offsets
        iris.store(dest_ptr, data_block, my_rank, dest_rank, heap_bases_ptr, mask=b_mask)
        offset += BLOCK_SIZE_BYTES

    # MODIFIED: Signal a specific flag at [destination, source]
    flag_ptr = signal_flags_ptr + dest_rank * world_size + my_rank
    iris.atomic_xchg(flag_ptr, 1, my_rank, dest_rank, heap_bases_ptr, sem="release")

@triton.jit
def fused_ag_gemm_kernel(
    gathered_act_ptr,
    W_ptr,
    C_ptr,
    signal_flags_ptr,
    M, K, N,
    stride_gathered_rank_bytes: tl.constexpr,
    stride_w_k, stride_w_n,
    stride_c_m, stride_c_n,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    K_local: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K_local: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles_total = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    
    # MODIFIED: The blocking wait at the start of the kernel is removed.

    # Persistent loop over output tiles
    for tile_idx in range(pid, num_tiles_total, NUM_SMS):
        rm, rn, _, _ = tile_id_to_index_range(tile_idx, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
        
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Loop over data from each source_rank
        for source_rank in range(world_size):
            # MODIFIED: Wait logic is now INSIDE the loop.
            # It waits for the specific shard from `source_rank` before proceeding.
            flag_ptr = signal_flags_ptr + my_rank * world_size + source_rank
            while tl.atomic_cas(flag_ptr, 0, 0, sem="acquire") == 0:
                pass # Spin-wait for this specific shard's data to be ready

            act_shard_base_ptr = gathered_act_ptr + source_rank * stride_gathered_rank_bytes
            act_shard_ptr_typed = tl.cast(act_shard_base_ptr, tl.pointer_type(C_ptr.dtype.element_ty))
            W_slice_ptr = W_ptr + (source_rank * K_local) * stride_w_k

            # Loop over the K_local dimension
            for k_offset in range(0, K_local, BLOCK_SIZE_K_local):
                rk_local = k_offset + tl.arange(0, BLOCK_SIZE_K_local)
                
                A_ptr = act_shard_ptr_typed + rm[:, None] * K_local + rk_local[None, :]
                A_mask = (rm[:, None] < M) & (rk_local[None, :] < K_local)
                a = tl.load(A_ptr, mask=A_mask, other=0.0)

                B_ptr = W_slice_ptr + rk_local[:, None] * stride_w_k + rn[None, :] * stride_w_n
                B_mask = (rk_local[:, None] < K_local) & (rn[None, :] < N)
                b = tl.load(B_ptr, mask=B_mask, other=0.0)
                
                acc += tl.dot(a, b)

        # Store the final computed tile
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_tile_ptr = C_ptr + rm[:, None] * stride_c_m + rn[None, :] * stride_c_n
        tl.store(C_tile_ptr, acc.to(C_ptr.dtype.element_ty), mask=c_mask)

class FusedAGGemmFused(torch.nn.Module):
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
        
        # MODIFIED: Signal flags are now 2D for fine-grained signaling.
        self.signal_flags = self.iris.zeros((TP, TP), dtype=torch.int32)
        
        self.BLOCK_SIZE_M = 16
        self.BLOCK_SIZE_N = 64
        self.BLOCK_SIZE_K_local = 64
        self.GROUP_SIZE_M = 8
        self.num_warps = 4
        self.num_stages = 3
        
    def clear_flags(self):
        self.signal_flags.zero_()

    def forward(self, local_act, local_W):
        bytes_to_send = local_act.nbytes
        self.local_staging_buffer[:bytes_to_send].copy_(local_act.flatten().view(torch.int8))
        self.iris.barrier()
        
        push_grid = (self.world_size,)
        push_kernel_bytes[push_grid](
            self.local_staging_buffer, self.gathered_buffer, self.signal_flags, self.iris.get_heap_bases(),
            bytes_to_send, self.rank, self.world_size,
            stride_gathered_rank_bytes=self.gathered_buffer.stride(0),
            BLOCK_SIZE_BYTES=32768,
        )

        output = torch.empty((self.M, self.N_local), device=local_act.device, dtype=self.dtype)
        num_sms = 304
        gemm_grid = (num_sms, )

        fused_ag_gemm_kernel[gemm_grid](
            self.gathered_buffer, local_W.contiguous(), output, self.signal_flags,
            self.M, self.K, self.N_local,
            stride_gathered_rank_bytes=self.gathered_buffer.stride(0),
            stride_w_k=local_W.stride(0), stride_w_n=local_W.stride(1),
            stride_c_m=output.stride(0), stride_c_n=output.stride(1),
            my_rank=self.rank, world_size=self.world_size, K_local=self.K_local,
            BLOCK_SIZE_M=self.BLOCK_SIZE_M,
            BLOCK_SIZE_N=self.BLOCK_SIZE_N,
            BLOCK_SIZE_K_local=self.BLOCK_SIZE_K_local,
            GROUP_SIZE_M=self.GROUP_SIZE_M,
            NUM_SMS=num_sms,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )
        
        return output