#!/usr/bin/env python3

import torch
import gc
import sys
import weakref

# Import our IrisTensor
import iris

def test_alternative_creation():
    """Test different ways of creating IrisTensor objects"""
    print("=== Testing Alternative Creation Methods ===")
    
    iris_instance = iris.iris(heap_size=1 << 28)
    
    # Method 1: Current method (as_subclass)
    print("\n--- Method 1: Current as_subclass method ---")
    tensor1 = iris_instance.zeros(100, 100, dtype=torch.float32)
    print(f"Tensor1 refcount: {sys.getrefcount(tensor1)}")
    print(f"Tensor1 data_ptr: {hex(tensor1.data_ptr())}")
    
    # Method 2: Try creating with explicit reference
    print("\n--- Method 2: With explicit reference ---")
    # Let's try to keep a reference to the original tensor
    original_tensor = iris_instance.memory_pool[0:40000].view(torch.float32).reshape((100, 100))
    print(f"Original tensor refcount: {sys.getrefcount(original_tensor)}")
    
    # Method 3: Test if the issue is with the tensor data itself
    print("\n--- Method 3: Test tensor data lifetime ---")
    # Create a regular torch tensor
    regular_tensor = torch.zeros(100, 100, dtype=torch.float32, device='cuda')
    print(f"Regular tensor refcount: {sys.getrefcount(regular_tensor)}")
    print(f"Regular tensor data_ptr: {hex(regular_tensor.data_ptr())}")
    
    # Store it in a list
    tensor_list = [regular_tensor]
    print(f"After storing in list: {sys.getrefcount(regular_tensor)}")
    
    # Remove from list
    tensor_list.clear()
    print(f"After removing from list: {sys.getrefcount(regular_tensor)}")
    
    # Clean up
    del tensor1, original_tensor, regular_tensor
    gc.collect()

def test_memory_pool_reference():
    """Test if the issue is with the memory pool reference"""
    print("\n=== Testing Memory Pool Reference ===")
    
    iris_instance = iris.iris(heap_size=1 << 28)
    
    # Check memory pool reference
    print(f"Memory pool refcount: {sys.getrefcount(iris_instance.memory_pool)}")
    
    # Create a slice of the memory pool
    memory_slice = iris_instance.memory_pool[0:40000]
    print(f"Memory slice refcount: {sys.getrefcount(memory_slice)}")
    
    # Create a view of the slice
    memory_view = memory_slice.view(torch.float32)
    print(f"Memory view refcount: {sys.getrefcount(memory_view)}")
    
    # Reshape the view
    memory_reshaped = memory_view.reshape((100, 100))
    print(f"Memory reshaped refcount: {sys.getrefcount(memory_reshaped)}")
    
    # Clean up
    del memory_slice, memory_view, memory_reshaped
    gc.collect()

def test_detailed_garbage_collection():
    """Test to understand when different parts of the tensor are deallocated"""
    print("\n=== Testing Detailed Garbage Collection ===")
    
    iris_instance = iris.iris(heap_size=1 << 28)
    
    # Create a tensor and track its components
    print("Creating IrisTensor...")
    tensor = iris_instance.zeros(100, 100, dtype=torch.float32)
    
    # Get weak references to track when objects are deallocated
    tensor_weak = weakref.ref(tensor)
    iris_weak = weakref.ref(iris_instance)
    
    print(f"Tensor refcount: {sys.getrefcount(tensor)}")
    print(f"Iris instance refcount: {sys.getrefcount(iris_instance)}")
    
    # Check if tensor has the _iris_tensor attribute
    if hasattr(tensor, '_original_tensor'):
        original_weak = weakref.ref(tensor._original_tensor)
        print(f"Original tensor refcount: {sys.getrefcount(tensor._original_tensor)}")
    else:
        original_weak = None
        print("No _original_tensor attribute found")
    
    # Delete the tensor
    print("\nDeleting tensor...")
    del tensor
    gc.collect()
    
    # Check what was deallocated
    print(f"Tensor deallocated: {tensor_weak() is None}")
    print(f"Iris instance deallocated: {iris_weak() is None}")
    if original_weak:
        print(f"Original tensor deallocated: {original_weak() is None}")

def test_memory_reuse():
    """Test that memory is properly reused and we get different pointers"""
    print("\n=== Testing Memory Reuse ===")
    
    iris_instance = iris.iris(heap_size=1 << 28)
    
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
    
    # Assert that we get a different pointer (new allocation)
    assert ptr3 != ptr2, f"Expected different pointer for new allocation, got {hex(ptr2)} vs {hex(ptr3)}"
    print("✓ New allocation working correctly")
    
    # Clean up
    del tensor3
    gc.collect()

def test_concurrent_tensors():
    """Test that multiple tensors can coexist and get different pointers"""
    print("\n=== Testing Concurrent Tensors ===")
    
    iris_instance = iris.iris(heap_size=1 << 28)
    
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

if __name__ == "__main__":
    test_alternative_creation()
    test_memory_pool_reference()
    test_detailed_garbage_collection()
    test_memory_reuse()
    test_concurrent_tensors() 