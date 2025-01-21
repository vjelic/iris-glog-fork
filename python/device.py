from hip import count_devices, set_device, get_device


def test_count_set_get_devices():
    print("Counting available GPU devices...")
    num_devices = count_devices()
    print(f"Number of available GPU devices: {num_devices}")

    if num_devices == 0:
        print("No GPU devices available. Test cannot proceed.")
        return

    print("Testing set_device and get_device in a loop...")
    for gpu_id in range(num_devices):
        print(f"Setting device to GPU {gpu_id}...")
        set_device(gpu_id)
        current_device = get_device()
        assert current_device == gpu_id, f"Expected GPU {gpu_id}, got GPU {current_device}"
        print(f"Successfully set and verified device {gpu_id}.")

    print("All devices were successfully set and verified!")


if __name__ == "__main__":
    test_count_set_get_devices()