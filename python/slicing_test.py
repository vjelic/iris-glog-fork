import torch

# Allocate a large memory pool on the GPU as int8
pool_size = 1024  # Size in bytes
memory_pool = torch.empty(pool_size, device="cuda", dtype=torch.int8)

# Simulate allocation: slice the memory pool
size = 10  # Number of elements of type torch.int
size_in_bytes = torch.tensor([], dtype=torch.int).element_size()
start = 0
end = size * size_in_bytes

# Get a sub-tensor view and typecast it to torch.int
sub_tensor = memory_pool[start:end].view(dtype=torch.int)

# Use the allocated sub-tensor
sub_tensor[:] = torch.arange(size, device="cuda", dtype=torch.int)

# Print the results
print("Sub-tensor:", sub_tensor)
print("Memory pool:", memory_pool)
print(f"Pointer of memory_pool: {memory_pool.data_ptr():#x}")
print(f"Pointer of sub_tensor: {sub_tensor.data_ptr():#x}")