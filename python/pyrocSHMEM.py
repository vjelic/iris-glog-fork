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

    print(f"Rank {rank} using device {gpu_id:#x}")

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
    ipc_handles = np.zeros((world_size, 64), dtype=np.uint8)
    ipc_handle = get_ipc_handle(heap_base, rank)

    world_barrier()

    # Synchronize and exchange data
    all_ipc_handles = mpi_allgather(np.frombuffer(ipc_handle, dtype=np.uint8))
    all_heap_bases = mpi_allgather(np.array([heap_bases[rank]], dtype=np.uint64))

    world_barrier()

    # Verify exchanged data
    # print(f"Rank {rank}: all_ipc_handles: {all_ipc_handles}")
    world_barrier()

    # print(
    #     f"Rank {rank}: All heap bases gathered: {[f'{x:#x}' for x in all_heap_bases]}"
    # )

    world_barrier()

    # Open IPC memory handles
    ipc_heap_bases = np.zeros(world_size, dtype=np.uintp)
    for i in range(world_size):
        if i != rank:
            # print(f"rank {rank} open handle -> {all_ipc_handles[i]}")
            handle = open_ipc_handle(all_ipc_handles[i], rank)
            ipc_heap_bases[i] = int(handle)
        else:
            ipc_heap_bases[i] = heap_bases[i]

    # Print collected heap bases
    print(f"Rank {rank}: Collected IPC heap bases:")
    for i in range(world_size):
        print(f"  GPU {i}: Heap base {hex(int(ipc_heap_bases[i]))}")

    for i in range(world_size):
        print(f"  GPU {i}: Heap base {hex(int(all_heap_bases[i]))}")

    world_barrier()

    print(type(ipc_heap_bases))
    print(type(ipc_heap_bases[0]))
    print(f"Rank {rank}: IPC test completed.")


if __name__ == "__main__":
    run_ipc()
