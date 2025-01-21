import triton
import triton.language as tl

from mpi_helpers import (
    init_mpi,
    mpi_allgather,
    world_barrier,
)
from hip import (
    set_device,
    get_device,
    count_devices,
    malloc_fine_grained,
    get_ipc_handle,
    open_ipc_handle,
)
import numpy as np
import math
import torch
import ctypes

LOGGING = True


class pyrocSHMEM:
    def __init__(self, heap_size = 1 << 30):
        comm, cur_rank, num_ranks = init_mpi()
        num_gpus = count_devices()

        gpu_id = cur_rank % num_gpus
        set_device(gpu_id)


        self.comm = comm
        self.num_ranks = num_ranks
        self.cur_rank = cur_rank
        self.gpu_id = gpu_id
        self.heap_size = heap_size
        self.heap_offset = 0
        self.alignment = 1024
        self.device = f"cuda:{gpu_id}"
        self.memory_pool = torch.empty(heap_size, device=self.device,
                                       dtype=torch.int8)

        heap_base = self.memory_pool.data_ptr()
        heap_base_ptr = ctypes.c_void_p(heap_base)

        self.log(f"heap_base: {heap_base}")
        self.log(f"type(heap_base): {type(heap_base)}")

        heap_bases = np.zeros(num_ranks, dtype=np.uint64)
        heap_bases[cur_rank] = heap_base
        ipc_handles = np.zeros((num_ranks, 64), dtype=np.uint8)
        ipc_handle = get_ipc_handle(heap_base_ptr, cur_rank)

        world_barrier()

        all_ipc_handles = mpi_allgather(np.frombuffer(ipc_handle, dtype=np.uint8))
        all_heap_bases = mpi_allgather(np.array([heap_bases[cur_rank]], dtype=np.uint64))

        world_barrier()

        ipc_heap_bases = np.zeros(num_ranks, dtype=np.uintp)
        for rank in range(num_ranks):
            if rank != cur_rank:
                handle = open_ipc_handle(all_ipc_handles[rank], cur_rank)
                ipc_heap_bases[rank] = int(handle)
            else:
                ipc_heap_bases[rank] = heap_bases[rank]

        self.log(f"Rank {rank}: Collected IPC heap bases:")
        for i in range(num_ranks):
            self.log(f"  GPU {i}: Heap base {hex(int(ipc_heap_bases[i]))}")

        world_barrier()
        self.heap_bases = torch.from_numpy(ipc_heap_bases).to(
                            device=self.device,
                            dtype=torch.uint64)

        world_barrier()



    def log(self, message):
        if LOGGING:
            print(f"[pyrocSHMEM] [{self.cur_rank}/{self.num_ranks}] {message}")

    def allocate(self, num_elements, dtype):
        self.log(f"allocate: num_elements = {num_elements}, dtype = {dtype}")

        element_size = torch.tensor([], dtype=dtype).element_size()
        size_in_bytes = num_elements * element_size
        aligned_size = math.ceil(size_in_bytes / self.alignment) * self.alignment

        if self.heap_offset + aligned_size > self.heap_size:
            raise MemoryError("Heap out of memory")

        start = self.heap_offset
        self.heap_offset += aligned_size

        sub_buffer = self.memory_pool[start : start + size_in_bytes].view(dtype)
        return sub_buffer.reshape((num_elements,))

    def zeros_like(self, tensor):
        dtype = tensor.dtype
        num_elements = tensor.numel()
        new_tensor = self.allocate(num_elements, dtype)
        new_tensor.zero_()
        return new_tensor

    def arange(self, size, dtype=torch.int):
        self.log(f"arange: size = {size}, dtype = {dtype}")
        tensor = self.allocate(num_elements=size, dtype=dtype)
        tensor[:] = torch.arange(size, device="cuda", dtype=dtype)
        return tensor

    def zeros(self, size, dtype=torch.int):
        self.log(f"zeros: size = {size}, dtype = {dtype}")
        tensor = self.allocate(num_elements=size, dtype=dtype)
        tensor.zero_()
        return tensor

    def ones(self, size, dtype=torch.int):
        self.log(f"ones: size = {size}, dtype = {dtype}")
        tensor = self.allocate(num_elements=size, dtype=dtype)
        tensor.fill_(1)
        return tensor

    def full(self, size, fill_value, dtype=torch.int):
        self.log(f"full: size = {size}, fill_value = {fill_value}, dtype = {dtype}")
        tensor = self.allocate(num_elements=size, dtype=dtype)
        tensor.fill_(fill_value)
        return tensor

    def uniform(self, size, low=0.0, high=1.0, dtype=torch.float):
        self.log(f"uniform: size = {size}, low = {low}, high = {high}, dtype = {dtype}")
        tensor = self.allocate(num_elements=size, dtype=dtype)
        tensor.uniform_(low, high)
        return tensor

    def empty(self, size, dtype=torch.float):
        self.log(f"empty: size = {size}, dtype = {dtype}")
        return self.allocate(num_elements=size, dtype=dtype)

    def randint(self, size, low, high, dtype=torch.int):
        self.log(f"randint: size = {size}, low = {low}, high = {high}, dtype = {dtype}")
        tensor = self.allocate(num_elements=size, dtype=dtype)
        tensor[:] = torch.randint(low, high, (size,), device="cuda", dtype=dtype)
        return tensor

    def linspace(self, start, end, steps, dtype=torch.float):
        self.log(f"linspace: start = {start}, end = {end}, steps = {steps}, dtype = {dtype}")
        tensor = self.allocate(num_elements=steps, dtype=dtype)
        torch.linspace(start, end, steps, out=tensor)
        return tensor

    def deallocate(self, pointer):
        pass

    def get_heap_bases(self):
        return self.heap_bases

    def barrier(self):
        world_barrier()
        # Not sure if we have to sync here...
        torch.cuda.synchronize()

    def get_device(self):
        return self.memory_pool.device

    def get_rank(self):
        return self.cur_rank

    def get_num_ranks(self):
        return self.num_ranks

@triton.jit
def translate(src_ptr, cur_rank, target_rank, heap_bases, dtype_size, debug = False):
    src_base = tl.load(heap_bases + cur_rank)
    dst_base = tl.load(heap_bases + target_rank)
    # Cast dst_base to pointer type
    dst_base = tl.cast(dst_base, src_ptr.dtype)
    # convert to int to compute difference
    src_ptr_int = tl.cast(src_ptr, tl.uint64)
    # Find the offset from current rank heap
    offset = (src_ptr_int - src_base) // tl.cast(dtype_size, tl.uint64)
    # Find the offset into the destination heap
    dst_ptr = dst_base + offset

    pid = tl.program_id(axis=0)
    if debug and pid == 0:
        print("src_ptr", src_ptr)
        print("src_base", src_base)
        print("dst_base", dst_base)
        print("offset", offset)
        print("dst_ptr", dst_ptr)

    return dst_ptr


@triton.jit
def get(src_ptr, cur_rank, target_rank, heap_bases, mask=None):

    dtype_size = 4
    dst_ptr = translate(src_ptr, cur_rank,
                        target_rank, heap_bases,
                        dtype_size)

    result = tl.load(dst_ptr, mask=mask)
    return result

@triton.jit
def put(src_ptr, data, cur_rank, target_rank, heap_bases, mask=None):
    dtype_size = 4
    dst_ptr = translate(src_ptr, cur_rank,
                        target_rank, heap_bases,
                        dtype_size,
                        False)
    tl.store(dst_ptr, data, mask=mask)

@triton.jit
def atomic_add(src_ptr, data, cur_rank, target_rank,
               heap_bases, mask=None, sem=None, scope=None):
    dtype_size = 4
    dst_ptr = translate(src_ptr, cur_rank,
                        target_rank, heap_bases,
                        dtype_size,
                        False)
    tl.atomic_add(dst_ptr, data, mask=mask, sem=sem, scope=scope)

@triton.jit
def atomic_sub(src_ptr, data, cur_rank, target_rank,
               heap_bases, mask=None, sem=None, scope=None):
    dtype_size = 4
    dst_ptr = translate(src_ptr, cur_rank,
                        target_rank, heap_bases,
                        dtype_size,
                        False)
    tl.atomic_sub(dst_ptr, data, mask=mask, sem=sem, scope=scope)

@triton.jit
def atomic_cas(src_ptr, compare, value, cur_rank, target_rank,
               heap_bases, sem=None, scope=None):
    dtype_size = 4
    dst_ptr = translate(src_ptr, cur_rank,
                        target_rank, heap_bases,
                        dtype_size,
                        False)
    return tl.atomic_cas(dst_ptr, compare, value, sem=sem, scope=scope)

@triton.jit
def atomic_xchg(src_ptr, value, cur_rank, target_rank,
               heap_bases, mask=None, sem=None, scope=None):
    dtype_size = 4
    dst_ptr = translate(src_ptr, cur_rank,
                        target_rank, heap_bases,
                        dtype_size,
                        False)
    return tl.atomic_xchg(dst_ptr, value, mask=mask, sem=sem, scope=scope)

