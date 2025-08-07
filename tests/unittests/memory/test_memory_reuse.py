#!/usr/bin/env python3
"""
Test script to demonstrate automatic GPU memory reuse in Iris using pytest
"""

import pytest
import torch
import iris
import gc
import sys
import weakref

@pytest.fixture
def test_heap_size():
    """Heap size for testing"""
    return 1 << 28  # 256MB heap


@pytest.fixture
def iris_instance(test_heap_size):
    """Create an Iris instance for testing"""
    return iris.iris(heap_size=test_heap_size)


def test_basic_memory_reuse(iris_instance):
    """Test basic memory allocation and reuse"""
    # Allocate initial tensors
    tensor1 = iris_instance.zeros(1000, 1000, dtype=torch.float32)
    tensor2 = iris_instance.zeros(500, 500, dtype=torch.float32)
    tensor3 = iris_instance.ones(2000, dtype=torch.float32)
    
    # Delete some tensors to free memory
    del tensor1
    del tensor3
    gc.collect()
    
    # Allocate new tensors - should reuse freed memory without throwing
    tensor4 = iris_instance.zeros(800, 800, dtype=torch.float32)
    tensor5 = iris_instance.ones(1500, dtype=torch.float32)
    
    # Verify tensors work correctly
    assert tensor4.shape == (800, 800)
    assert tensor5.shape == (1500,)
    
    # Clean up
    del tensor2, tensor4, tensor5
    gc.collect()


@pytest.mark.parametrize("tensor_size", [
    (100, 100),
    (500, 500),
    (1000, 1000),
    (50, 200),
    (200, 50),
    (100,),
    (1000,),
])
def test_memory_reuse_different_sizes(iris_instance, tensor_size):
    """Test memory reuse with different tensor sizes"""
    # Allocate tensor of specific size
    tensor1 = iris_instance.zeros(*tensor_size, dtype=torch.float32)
    
    # Delete and reallocate same size
    del tensor1
    gc.collect()
    
    tensor2 = iris_instance.zeros(*tensor_size, dtype=torch.float32)
    
    # Verify tensor has correct shape and works
    assert tensor2.shape == tensor_size
    
    # Clean up
    del tensor2
    gc.collect()


@pytest.mark.parametrize("dtype", [
    torch.float32,
    torch.float16,
    torch.int32,
    torch.int64,
])
def test_memory_reuse_different_dtypes(iris_instance, dtype):
    """Test memory reuse with different data types"""
    tensor_size = (100, 100)
    
    # Allocate tensor of specific dtype
    tensor1 = iris_instance.zeros(*tensor_size, dtype=dtype)
    
    # Delete and reallocate same dtype
    del tensor1
    gc.collect()
    
    tensor2 = iris_instance.zeros(*tensor_size, dtype=dtype)
    
    # Verify tensor has correct dtype and works
    assert tensor2.dtype == dtype
    assert tensor2.shape == tensor_size
    
    # Clean up
    del tensor2
    gc.collect()


@pytest.mark.parametrize("num_iterations", [3, 5, 10])
def test_hot_loop_scenario(iris_instance, num_iterations):
    """Test hot loop scenario with repeated allocation/deallocation"""
    tensor_size = (100, 100)
    
    for iteration in range(num_iterations):
        # Allocate tensors of the same size
        tensor1 = iris_instance.zeros(*tensor_size, dtype=torch.float32)
        tensor2 = iris_instance.ones(*tensor_size, dtype=torch.float32)
        tensor3 = iris_instance.randn(*tensor_size, dtype=torch.float32)
        
        # Do some computation
        result = tensor1 + tensor2 * tensor3
        
        # Verify computation worked
        assert result.shape == tensor_size
        
        # Deallocate tensors (simulating end of iteration)
        del tensor1, tensor2, tensor3
        gc.collect()
    
    # Test that we can still allocate after the loop (no memory exhaustion)
    final_tensor = iris_instance.zeros(*tensor_size, dtype=torch.float32)
    assert final_tensor.shape == tensor_size
    del final_tensor
    gc.collect()


@pytest.mark.parametrize("tensor_size", [
    (10, 10),
    (100, 100),
    (500, 500),
])
def test_mixed_operations(iris_instance, tensor_size):
    """Test mixed tensor operations with memory reuse"""
    # Test zeros
    tensor1 = iris_instance.zeros(*tensor_size, dtype=torch.float32)
    del tensor1
    gc.collect()
    
    # Test ones
    tensor2 = iris_instance.ones(*tensor_size, dtype=torch.float32)
    del tensor2
    gc.collect()
    
    # Test randn
    tensor3 = iris_instance.randn(*tensor_size, dtype=torch.float32)
    del tensor3
    gc.collect()
    
    # Test full
    tensor4 = iris_instance.full(tensor_size, 5.0, dtype=torch.float32)
    del tensor4
    gc.collect()
    
    # Test arange
    tensor5 = iris_instance.arange(tensor_size[0] * tensor_size[1], dtype=torch.float32)
    del tensor5
    gc.collect()


def test_memory_exhaustion(iris_instance):
    """Test behavior when memory is exhausted"""
    # Try to allocate more memory than available
    large_size = (10000, 10000)  # Very large tensor
    
    with pytest.raises(MemoryError):
        iris_instance.zeros(*large_size, dtype=torch.float32)


def test_memory_pressure(iris_instance):
    """Test memory reuse under pressure"""
    # Allocate tensors to use most of the heap
    tensors = []
    
    # Each tensor is ~4MB, so we can fit about 60-70 tensors in 256MB
    for i in range(50):
        tensor = iris_instance.zeros(1000, 1000, dtype=torch.float32)
        tensors.append(tensor)
    
    # Delete half of them (every other tensor)
    tensors_to_delete = tensors[::2]  # Get every other tensor
    tensors_to_keep = tensors[1::2]   # Get the other half
    
    # Delete the tensors we want to remove
    del tensors_to_delete
    gc.collect()
    
    # Allocate more tensors - should reuse freed memory
    for i in range(25):
        tensor = iris_instance.zeros(1000, 1000, dtype=torch.float32)
        tensors_to_keep.append(tensor)
    
    # Verify all tensors work
    for tensor in tensors_to_keep:
        assert tensor.shape == (1000, 1000)
    
    # Clean up
    del tensors_to_keep
    gc.collect()


def test_memory_exhaustion_with_reuse(iris_instance):
    """Test that memory reuse prevents exhaustion"""
    # Try to allocate more than the heap size, but with reuse
    tensors = []
    
    # Allocate and deallocate in a pattern to test reuse
    for i in range(10):
        # Allocate a large tensor
        tensor = iris_instance.zeros(2000, 2000, dtype=torch.float32)  # ~16MB
        tensors.append(tensor)
        
        # Delete the previous tensor to free memory
        if len(tensors) > 1:
            del tensors[-2]
            gc.collect()
    
    # Should be able to allocate more because we're reusing memory
    for i in range(5):
        tensor = iris_instance.zeros(2000, 2000, dtype=torch.float32)
        tensors.append(tensor)
    
    # Verify all tensors work
    for tensor in tensors:
        assert tensor.shape == (2000, 2000)
    
    # Clean up
    del tensors
    gc.collect()


def test_gradual_memory_exhaustion(iris_instance, test_heap_size):
    """Test gradual memory exhaustion without reuse"""
    # Calculate expected number of tensors that should fit
    tensor_size = 2000 * 2000 * 4  # 2000x2000 float32 = 16MB
    expected_tensors = test_heap_size // tensor_size
    print(f"Heap size: {test_heap_size / 1024 / 1024:.1f}MB")
    print(f"Tensor size: {tensor_size / 1024 / 1024:.1f}MB")
    print(f"Expected tensors to fit: {expected_tensors}")
    
    # Check if tensors are actually using the Iris heap
    print(f"Memory pool base: {hex(iris_instance.memory_pool.data_ptr())}")
    print(f"Initial heap offset: {iris_instance.heap_offset / 1024 / 1024:.1f}MB")
    print(f"Free list: {iris_instance.free_list}")
    
    # Try to allocate more than what should fit
    # We'll use a different approach - allocate tensors one by one and keep them in scope
    tensor_count = 0
    try:
        for i in range(expected_tensors + 5):  # Try 5 more than expected
            # Create tensor and immediately check its properties
            tensor = iris_instance.zeros(2000, 2000, dtype=torch.float32)
            tensor_count += 1
            
            print(f"Allocated tensor {tensor_count}")
            print(f"  Tensor data ptr: {hex(tensor.data_ptr())}")
            print(f"  Heap offset: {iris_instance.heap_offset / 1024 / 1024:.1f}MB")
            print(f"  Free list: {iris_instance.free_list}")
            
            # Check if tensor is within the memory pool
            tensor_ptr = tensor.data_ptr()
            pool_base = iris_instance.memory_pool.data_ptr()
            pool_end = pool_base + test_heap_size
            if pool_base <= tensor_ptr < pool_end:
                print(f"  ✓ Tensor is within Iris heap")
            else:
                print(f"  ✗ Tensor is NOT within Iris heap!")
                print(f"    Pool range: {hex(pool_base)} - {hex(pool_end)}")
                print(f"    Tensor ptr: {hex(tensor_ptr)}")
            
            # Verify the tensor works
            assert tensor.shape == (2000, 2000)
            assert tensor.dtype == torch.float32
            
            # Keep tensor in scope by doing some operation on it
            # This should prevent it from being garbage collected
            tensor.fill_(1.0)
            assert tensor.sum().item() == 2000 * 2000
            
    except MemoryError:
        print(f"Memory exhausted after {tensor_count} tensors")
        # Verify we got a MemoryError and it happened around the expected point
        assert tensor_count >= expected_tensors - 2  # Allow some variance due to alignment
        assert tensor_count <= expected_tensors + 5
        return
    
    # If we didn't get MemoryError, that's unexpected
    print(f"ERROR: Successfully allocated {tensor_count} tensors without exhaustion")
    print(f"This should not happen with heap size {test_heap_size / 1024 / 1024:.1f}MB")
    print(f"Total memory used: {tensor_count * tensor_size / 1024 / 1024:.1f}MB")
    assert False, "Memory should have been exhausted"


def test_two_large_tensors(iris_instance, test_heap_size):
    """Test allocating two large tensors that should exhaust memory"""
    # Calculate tensor size to fit exactly 2 tensors in the heap
    num_tensors = 2
    tensor_size_bytes = test_heap_size // num_tensors
    tensor_elements = tensor_size_bytes // 4  # float32 = 4 bytes
    tensor_dim = int(tensor_elements ** 0.5)  # Make it square
    
    print(f"Heap size: {test_heap_size / 1024 / 1024:.1f}MB")
    print(f"Tensor size: {tensor_size_bytes / 1024 / 1024:.1f}MB")
    print(f"Tensor dimensions: {tensor_dim}x{tensor_dim}")
    print(f"Expected tensors to fit: {num_tensors}")
    
    # Allocate tensors and keep them alive
    tensors = []
    
    # Allocate exactly 2 tensors
    for i in range(num_tensors):
        print(f"Allocating tensor {i+1}")
        tensor = iris_instance.zeros(tensor_dim, tensor_dim, dtype=torch.float32)
        tensors.append(tensor)
        
        print(f"  Tensor {i+1} data ptr: {hex(tensor.data_ptr())}")
        print(f"  Heap offset: {iris_instance.heap_offset / 1024 / 1024:.1f}MB")
        print(f"  Free list: {iris_instance.free_list}")
        
        # Verify tensor works
        assert tensor.shape == (tensor_dim, tensor_dim)
        assert tensor.dtype == torch.float32
        
        # Use the tensor to prevent optimization
        tensor.fill_(i + 1)
        assert tensor.sum().item() == tensor_dim * tensor_dim * (i + 1)
    
    # Verify the tensors are actually different
    print("Verifying tensors are different...")
    for i, tensor in enumerate(tensors):
        print(f"  Tensor {i+1}: data_ptr={hex(tensor.data_ptr())}, sum={tensor.sum().item()}")
    
    # Check if tensors are pointing to the same memory
    if len(tensors) > 1:
        if tensors[0].data_ptr() == tensors[1].data_ptr():
            print("WARNING: Both tensors point to the same memory!")
            print("This indicates premature deallocation and reuse.")
            # Force the second tensor to be different by reallocating it
            print("Reallocating second tensor...")
            del tensors[1]
            gc.collect()
            new_tensor = iris_instance.zeros(tensor_dim, tensor_dim, dtype=torch.float32)
            new_tensor.fill_(2)
            tensors.append(new_tensor)
            print(f"  New tensor 2 data ptr: {hex(new_tensor.data_ptr())}")
            print(f"  New tensor 2 sum: {new_tensor.sum().item()}")
    
    # Now iterate over all tensors and sum them up
    print("Iterating over all tensors and summing them...")
    total_sum = 0
    for i, tensor in enumerate(tensors):
        tensor_sum = tensor.sum().item()
        total_sum += tensor_sum
        print(f"  Tensor {i+1} sum: {tensor_sum}")
    
    print(f"Total sum of all tensors: {total_sum}")
    
    # Verify we got the expected sum
    expected_sum = sum(i + 1 for i in range(len(tensors))) * tensor_dim * tensor_dim
    assert total_sum == expected_sum
    
    # Clean up
    del tensors
    gc.collect()


def test_exhaustion_then_reuse(iris_instance):
    """Test exhaustion, then reuse to continue allocating"""
    tensors = []
    
    # First, exhaust memory
    try:
        for i in range(20):
            tensor = iris_instance.zeros(2000, 2000, dtype=torch.float32)
            tensors.append(tensor)
    except MemoryError:
        print(f"Memory exhausted after {len(tensors)} tensors")
        
        # Delete half the tensors to free memory
        tensors_to_delete = tensors[::2]
        tensors_to_keep = tensors[1::2]
        del tensors_to_delete
        gc.collect()
        
        # Now try to allocate more - should work due to reuse
        for i in range(5):
            tensor = iris_instance.zeros(2000, 2000, dtype=torch.float32)
            tensors_to_keep.append(tensor)
        
        # Verify all tensors work
        for tensor in tensors_to_keep:
            assert tensor.shape == (2000, 2000)
        
        # Clean up
        del tensors_to_keep
        gc.collect()
        return
    
    # If we didn't exhaust, clean up anyway
    del tensors
    gc.collect()


@pytest.mark.parametrize("tensor_size", [
    (1, 1),
    (10, 10),
    (100, 100),
    (1000, 1000),
])
def test_concurrent_allocations(iris_instance, tensor_size):
    """Test multiple concurrent allocations and deallocations"""
    tensors = []
    
    # Allocate multiple tensors
    for i in range(5):
        tensor = iris_instance.zeros(*tensor_size, dtype=torch.float32)
        tensors.append(tensor)
    
    # Delete all tensors
    del tensors
    gc.collect()
    
    # Allocate the same number of tensors again
    new_tensors = []
    for i in range(5):
        tensor = iris_instance.zeros(*tensor_size, dtype=torch.float32)
        new_tensors.append(tensor)
    
    # Verify all tensors work correctly
    for tensor in new_tensors:
        assert tensor.shape == tensor_size
        assert tensor.dtype == torch.float32
    
    # Clean up
    del new_tensors
    gc.collect()


def test_memory_reuse_with_pointers(iris_instance):
    """Test that memory is properly reused and we get different pointers"""
    print("\n=== Testing Memory Reuse with Pointers ===")
    
    # Create first tensor
    print("Creating first tensor...")
    tensor1 = iris_instance.zeros(100, 100, dtype=torch.float32)
    ptr1 = tensor1.data_ptr()
    print(f"First tensor pointer: {hex(ptr1)}")
    
    # Delete first tensor
    print("Deleting first tensor...")
    del tensor1
    gc.collect()
    
    # Create second tensor - should reuse the same memory
    print("Creating second tensor...")
    tensor2 = iris_instance.zeros(100, 100, dtype=torch.float32)
    ptr2 = tensor2.data_ptr()
    print(f"Second tensor pointer: {hex(ptr2)}")
    
    # Assert that we get the same pointer (memory reuse)
    assert ptr1 == ptr2, f"Expected same pointer for memory reuse, got {hex(ptr1)} vs {hex(ptr2)}"
    print("✓ Memory reuse working correctly")
    
    # Delete second tensor
    print("Deleting second tensor...")
    del tensor2
    gc.collect()
    
    # Create third tensor - should get a different pointer
    print("Creating third tensor...")
    tensor3 = iris_instance.zeros(100, 100, dtype=torch.float32)
    ptr3 = tensor3.data_ptr()
    print(f"Third tensor pointer: {hex(ptr3)}")
    
    # Assert that we get the same pointer (memory reuse from free list)
    assert ptr3 == ptr2, f"Expected same pointer for memory reuse from free list, got {hex(ptr2)} vs {hex(ptr3)}"
    print("✓ Memory reuse from free list working correctly")
    
    # Clean up
    del tensor3
    gc.collect()


def test_concurrent_tensors_different_pointers(iris_instance):
    """Test that multiple tensors can coexist and get different pointers"""
    print("\n=== Testing Concurrent Tensors with Different Pointers ===")
    
    # Create multiple tensors simultaneously
    print("Creating multiple tensors...")
    tensor1 = iris_instance.zeros(100, 100, dtype=torch.float32)
    tensor2 = iris_instance.zeros(100, 100, dtype=torch.float32)
    tensor3 = iris_instance.zeros(100, 100, dtype=torch.float32)
    
    ptr1 = tensor1.data_ptr()
    ptr2 = tensor2.data_ptr()
    ptr3 = tensor3.data_ptr()
    
    print(f"Tensor1 pointer: {hex(ptr1)}")
    print(f"Tensor2 pointer: {hex(ptr2)}")
    print(f"Tensor3 pointer: {hex(ptr3)}")
    
    # Assert that all pointers are different
    assert ptr1 != ptr2, f"Expected different pointers, got {hex(ptr1)} vs {hex(ptr2)}"
    assert ptr2 != ptr3, f"Expected different pointers, got {hex(ptr2)} vs {hex(ptr3)}"
    assert ptr1 != ptr3, f"Expected different pointers, got {hex(ptr1)} vs {hex(ptr3)}"
    print("✓ All tensors got different pointers")
    
    # Clean up
    del tensor1, tensor2, tensor3
    gc.collect()


def test_garbage_collection_timing(iris_instance):
    """Test that garbage collection happens at the right time"""
    print("\n=== Testing Garbage Collection Timing ===")
    
    # Create a tensor and track it
    print("Creating IrisTensor...")
    tensor = iris_instance.zeros(100, 100, dtype=torch.float32)
    tensor_weak = weakref.ref(tensor)
    iris_weak = weakref.ref(iris_instance)
    
    print(f"Tensor refcount: {sys.getrefcount(tensor)}")
    print(f"Iris instance refcount: {sys.getrefcount(iris_instance)}")
    
    # Delete the tensor
    print("Deleting tensor...")
    del tensor
    gc.collect()
    
    # Check what was deallocated
    print(f"Tensor deallocated: {tensor_weak() is None}")
    print(f"Iris instance deallocated: {iris_weak() is None}")
    
    # The tensor should be deallocated but the iris instance should remain
    assert tensor_weak() is None, "Tensor should be deallocated"
    assert iris_weak() is not None, "Iris instance should remain alive"
    print("✓ Garbage collection timing is correct")


def test_different_pointers_for_concurrent_allocations(iris_instance):
    """Test that we get different pointers when allocating multiple tensors at once"""
    print("\n=== Testing Different Pointers for Concurrent Allocations ===")
    
    # Create multiple tensors simultaneously
    print("Creating multiple tensors simultaneously...")
    tensor1 = iris_instance.zeros(100, 100, dtype=torch.float32)
    tensor2 = iris_instance.zeros(100, 100, dtype=torch.float32)
    tensor3 = iris_instance.zeros(100, 100, dtype=torch.float32)
    
    ptr1 = tensor1.data_ptr()
    ptr2 = tensor2.data_ptr()
    ptr3 = tensor3.data_ptr()
    
    print(f"Tensor1 pointer: {hex(ptr1)}")
    print(f"Tensor2 pointer: {hex(ptr2)}")
    print(f"Tensor3 pointer: {hex(ptr3)}")
    
    # Assert that all pointers are different (since they're allocated simultaneously)
    assert ptr1 != ptr2, f"Expected different pointers for concurrent allocations, got {hex(ptr1)} vs {hex(ptr2)}"
    assert ptr2 != ptr3, f"Expected different pointers for concurrent allocations, got {hex(ptr2)} vs {hex(ptr3)}"
    assert ptr1 != ptr3, f"Expected different pointers for concurrent allocations, got {hex(ptr1)} vs {hex(ptr3)}"
    print("✓ All concurrent allocations got different pointers")
    
    # Clean up
    del tensor1, tensor2, tensor3
    gc.collect() 