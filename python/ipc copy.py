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


def run_ipc():
    comm, rank, world_size = init_mpi()
    num_gpus = count_devices()

    gpu_id = rank % num_gpus
    set_device(gpu_id)

    print(f"Rank {rank} using device {gpu_id}")

    world_barrier()

    bytes_count = 1 << 30  # 1GB

    # Allocate fine-grained memory
    heap_base = malloc_fine_grained(bytes_count)

    if heap_base.value is None:
        raise RuntimeError(f"Rank {rank}: Memory allocation failed.")

    # Collect heap bases and IPC handles
    heap_bases = np.zeros(world_size, dtype=np.uint64)
    heap_bases[rank] = heap_base.value

    # Serialize IPC handles as 64-bit values
    ipc_handles = np.zeros(world_size, dtype=np.uint64)
    ipc_handle = get_ipc_handle(heap_base)

    # Not sure if the little is correct
    ipc_handle_bytes = bytes((b & 0xFF for b in ipc_handle[:8]))
    ipc_handles[rank] = int.from_bytes(ipc_handle_bytes, byteorder="little")
    print(f"Rank {rank}: Heap base {heap_bases[rank]}")
    print(f"Rank {rank}: IPC handle raw data {ipc_handles[rank]}")

    print(f"Rank {rank}: All IPC handles gathered: \n{ipc_handles}")
    print(f"Rank {rank}: All heap bases gathered: \n{heap_bases}")

    world_barrier()

    # Synchronize and exchange data
    all_ipc_handles = mpi_allgather(np.array([ipc_handles[rank]], dtype=np.uint64))
    all_heap_bases = mpi_allgather(np.array([heap_bases[rank]], dtype=np.uint64))

    world_barrier()

    # Verify exchanged data
    print(f"Rank {rank}: All IPC handles gathered: \n{all_ipc_handles}")
    print(f"Rank {rank}: All heap bases gathered: \n{all_heap_bases}")

    return

    # Open IPC memory handles
    ipc_heap_bases = np.zeros(world_size, dtype=np.uintp)
    for i in range(world_size):
        if i != rank:
            ipc_heap_bases[i] = int(open_ipc_handle(all_ipc_handles[i]))
        else:
            ipc_heap_bases[i] = heap_bases[i]

    # Print collected heap bases
    print(f"Rank {rank}: Collected IPC heap bases:")
    for i in range(world_size):
        print(f"  GPU {i}: Heap base {hex(ipc_heap_bases[i])}")

    for i in range(world_size):
        print(f"  GPU {i}: Heap base {hex(all_heap_bases[i])}")

    world_barrier()
    print(f"Rank {rank}: IPC test completed.")


if __name__ == "__main__":
    import sys

    run_ipc()
