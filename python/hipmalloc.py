import ctypes


from hip import hip_try, hip_malloc, malloc_fine_grained, hip_free


def test_malloc_free():
    size = 1024 * 1024  # Allocate 1MB
    print(f"Allocating {size} bytes on the GPU...")
    ptr = hip_malloc(size)
    print(f"Memory allocated at address: {ptr}")

    print("Freeing GPU memory...")
    hip_free(ptr)
    print("Memory freed successfully!")


def test_malloc_fine_grained():
    size = 1024 * 1024  # 1MB
    print(f"Allocating {size} bytes of fine-grained memory on the GPU...")
    ptr = malloc_fine_grained(size)
    print(f"Fine-grained memory allocated at address: {ptr}")

    print("Freeing the fine-grained memory...")
    hip_free(ptr)
    print("Fine-grained memory freed successfully!")


# Run the test
if __name__ == "__main__":
    test_malloc_free()
    test_malloc_fine_grained()
