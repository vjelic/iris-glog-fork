#!/usr/bin/env python3

import torch
import gc
import sys

# Import our IrisTensor
import iris

def test_tensor_lifetimes():
    """Test to understand tensor lifetimes and reference counting"""
    print("=== Testing Tensor Lifetimes ===")
    
    # Create a simple Iris instance
    iris_instance = iris.iris(heap_size=1 << 28)  # 256MB
    
    print(f"Initial heap offset: {iris_instance.heap_offset}")
    print(f"Initial free list: {iris_instance.free_list}")
    
    # Test 1: Basic tensor allocation
    print("\n--- Test 1: Basic allocation ---")
    tensor1 = iris_instance.zeros(100, 100, dtype=torch.float32)
    print(f"After allocation - heap offset: {iris_instance.heap_offset}")
    print(f"After allocation - free list: {iris_instance.free_list}")
    print(f"Tensor1 data_ptr: {hex(tensor1.data_ptr())}")
    print(f"Tensor1 refcount: {sys.getrefcount(tensor1)}")
    
    # Test 2: Create another tensor
    print("\n--- Test 2: Second allocation ---")
    tensor2 = iris_instance.zeros(100, 100, dtype=torch.float32)
    print(f"After second allocation - heap offset: {iris_instance.heap_offset}")
    print(f"After second allocation - free list: {iris_instance.free_list}")
    print(f"Tensor2 data_ptr: {hex(tensor2.data_ptr())}")
    print(f"Tensor2 refcount: {sys.getrefcount(tensor2)}")
    
    # Test 3: Check if tensors are the same
    print("\n--- Test 3: Tensor comparison ---")
    print(f"Tensor1 data_ptr: {hex(tensor1.data_ptr())}")
    print(f"Tensor2 data_ptr: {hex(tensor2.data_ptr())}")
    print(f"Same data_ptr: {tensor1.data_ptr() == tensor2.data_ptr()}")
    
    # Test 4: Manual garbage collection
    print("\n--- Test 4: Manual garbage collection ---")
    print("Before gc - heap offset:", iris_instance.heap_offset)
    print("Before gc - free list:", iris_instance.free_list)
    
    # Delete one tensor and force GC
    del tensor1
    gc.collect()
    
    print("After gc - heap offset:", iris_instance.heap_offset)
    print("After gc - free list:", iris_instance.free_list)
    
    # Test 5: Allocate another tensor after GC
    print("\n--- Test 5: Allocation after GC ---")
    tensor3 = iris_instance.zeros(100, 100, dtype=torch.float32)
    print(f"After third allocation - heap offset: {iris_instance.heap_offset}")
    print(f"After third allocation - free list: {iris_instance.free_list}")
    print(f"Tensor3 data_ptr: {hex(tensor3.data_ptr())}")
    
    # Test 6: Check if reuse happened
    print("\n--- Test 6: Reuse check ---")
    print(f"Tensor2 data_ptr: {hex(tensor2.data_ptr())}")
    print(f"Tensor3 data_ptr: {hex(tensor3.data_ptr())}")
    print(f"Reuse happened: {tensor2.data_ptr() == tensor3.data_ptr()}")
    
    # Clean up
    del tensor2, tensor3
    gc.collect()

def test_reference_counting():
    """Test reference counting behavior"""
    print("\n=== Testing Reference Counting ===")
    
    iris_instance = iris.iris(heap_size=1 << 28)
    
    # Create a tensor
    tensor = iris_instance.zeros(100, 100, dtype=torch.float32)
    print(f"Initial refcount: {sys.getrefcount(tensor)}")
    
    # Store in a list
    tensor_list = [tensor]
    print(f"After storing in list: {sys.getrefcount(tensor)}")
    
    # Create another reference
    tensor_ref = tensor
    print(f"After creating ref: {sys.getrefcount(tensor)}")
    
    # Remove from list
    tensor_list.clear()
    print(f"After removing from list: {sys.getrefcount(tensor)}")
    
    # Remove the reference
    del tensor_ref
    print(f"After removing ref: {sys.getrefcount(tensor)}")
    
    # Final cleanup
    del tensor
    gc.collect()

if __name__ == "__main__":
    test_tensor_lifetimes()
    test_reference_counting() 