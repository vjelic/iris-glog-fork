# __init__.py

import os
import torch


# Pipe allocations via finegrained allocator
current_dir = os.path.dirname(__file__)
finegrained_alloc_path = os.path.join(
    current_dir, "finegrained_alloc", "libfinegrained_allocator.so"
)
finegrained_allocator = torch.cuda.memory.CUDAPluggableAllocator(
    finegrained_alloc_path,
    "finegrained_hipMalloc",
    "finegrained_hipFree",
)
torch.cuda.memory.change_current_allocator(finegrained_allocator)


from .pyrocSHMEM import (
    pyrocSHMEM,
    translate,
    get,
    put,
    atomic_add,
    atomic_sub,
    atomic_cas,
    atomic_xchg,
)

__all__ = [
    "pyrocSHMEM",
    "translate",
    "get",
    "put",
    "atomic_add",
    "atomic_sub",
    "atomic_cas",
    "atomic_xchg,",
]
