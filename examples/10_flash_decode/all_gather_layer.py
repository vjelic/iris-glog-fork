import torch
import triton
import iris
from all_gather_push_kernel import iris_allgather_push_kernel

class IrisAllGatherLayer:
    def __init__(self, iris_instance: iris.Iris, max_buffer_size: int, dtype: torch.dtype):
        self.iris = iris_instance
        self.rank = self.iris.get_rank()
        self.num_ranks = self.iris.get_num_ranks()
        
        max_bytes_per_rank = max_buffer_size // self.num_ranks
        
        
        self.gathered_buffer = self.iris.empty(
            (self.num_ranks, max_bytes_per_rank), dtype=torch.int8
        )
        
        self.local_staging_buffer = self.iris.empty(
            (max_bytes_per_rank,), dtype=torch.int8
        )
        
        self.signal_flags = self.iris.zeros((self.num_ranks,), dtype=torch.int32)

    def forward(self, local_data: torch.Tensor):
        bytes_per_rank = local_data.nbytes
        
        staging_slice = self.local_staging_buffer[:bytes_per_rank]
        
        staging_slice.copy_(local_data.flatten().view(torch.int8))
        
        self.signal_flags.zero_()
        self.iris.barrier() 

        grid = lambda meta: (self.num_ranks,)
        
        stride_bytes = self.gathered_buffer.stride(0)

        iris_allgather_push_kernel[grid](
            self.local_staging_buffer,
            self.gathered_buffer,
            self.signal_flags,
            self.iris.get_heap_bases(),
            self.rank,
            self.num_ranks,
            bytes_per_rank,
            stride_output_rank=stride_bytes,
            BLOCK_SIZE_B=1024, 
        )
        
        list_of_shards = []
        for i in range(self.num_ranks):
            row_int8 = self.gathered_buffer[i]
            valid_data_int8 = row_int8[:bytes_per_rank]
            valid_data_typed = valid_data_int8.view(local_data.dtype)
            list_of_shards.append(valid_data_typed)

        return torch.cat(list_of_shards)