from hip import count_devices, set_device, get_device
from mpi_helpers import init_mpi, world_barrier


def test_mpi_hip():
    comm, rank, world_size = init_mpi()
    print(f"Rank {rank}/{world_size} initializing...")

    # Count devices on this rank
    num_devices = count_devices()
    print(f"Rank {rank}: Found {num_devices} GPU(s).")

    if num_devices > 0:
        # Set and verify the device
        gpu_id = rank % num_devices
        print(f"Rank {rank}: Setting device to GPU {gpu_id}...")
        set_device(gpu_id)
        current_device = get_device()
        print(f"Rank {rank}: Currently active GPU device is {current_device}.")
        assert current_device == gpu_id, f"Mismatch: GPU {current_device} != {gpu_id}"

    # Synchronize across all ranks
    print(f"Rank {rank}: Waiting at barrier...")
    world_barrier()
    print(f"Rank {rank}: Barrier synchronization complete.")

    # Final message
    print(f"Rank {rank}: MPI + HIP test completed successfully.")


if __name__ == "__main__":
    test_mpi_hip()
