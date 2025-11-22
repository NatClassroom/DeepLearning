"""
Topic 4: Tensor Indexing
=========================

Indexing: tensor[i, j, k] for multi-dimensional access
Slicing: tensor[start:end:step] for ranges
Boolean indexing: tensor[mask] for conditional selection
"""

import torch
import numpy as np


def basic_indexing():
    """Basic indexing and slicing."""
    print("=" * 60)
    print("Basic Indexing")
    print("=" * 60)
    
    tensor_3d = torch.rand(3, 4, 5)
    print(f"\n3D tensor shape: {tensor_3d.shape}")
    print(f"First element: {tensor_3d[0, 0, 0]}")
    print(f"First 2D slice shape: {tensor_3d[0].shape}")
    print(f"Last slice shape: {tensor_3d[-1].shape}")
    
    # 2D matrix for slicing examples
    matrix = torch.arange(12).reshape(3, 4)
    print(f"\nMatrix:\n{matrix}")
    
    print(f"\nFirst row: {matrix[0, :]}")
    print(f"First column: {matrix[:, 0]}")
    print(f"\nFirst two rows:\n{matrix[:2, :]}")
    print(f"\nLast two columns:\n{matrix[:, -2:]}")
    print(f"\nEvery other row:\n{matrix[::2, :]}")
    print(f"\nEvery other column:\n{matrix[:, ::2]}")


def advanced_indexing():
    """Advanced indexing with lists and tensors."""
    print("\n" + "=" * 60)
    print("Advanced Indexing")
    print("=" * 60)
    
    data = torch.arange(20).reshape(4, 5)
    print(f"\nData:\n{data}")
    
    # Index with lists
    print(f"\nRows [0, 2]:\n{data[[0, 2]]}")
    print(f"\nColumns [1, 3]:\n{data[:, [1, 3]]}")
    
    # Index with tensor
    indices = torch.tensor([0, 2])
    print(f"\nUsing tensor indices {indices}:\n{data[indices]}")


def boolean_indexing():
    """Boolean indexing for conditional selection."""
    print("\n" + "=" * 60)
    print("Boolean Indexing")
    print("=" * 60)
    
    values = torch.tensor([1, 5, 3, 8, 2, 7, 4, 6])
    print(f"\nValues: {values}")
    
    # Create boolean mask
    mask = values > 4
    print(f"Mask (values > 4): {mask}")
    print(f"Filtered: {values[mask]}")
    
    # Multiple conditions
    mask_range = (values > 3) & (values < 7)
    print(f"\nMask (3 < values < 7): {mask_range}")
    print(f"Filtered: {values[mask_range]}")
    
    # 2D boolean indexing
    matrix = torch.rand(3, 4)
    print(f"\nMatrix:\n{matrix}")
    mask_2d = matrix > 0.5
    print(f"\nValues > 0.5: {matrix[mask_2d]}")


def modifying_tensors():
    """Modify tensors in-place."""
    print("\n" + "=" * 60)
    print("Modifying Tensors")
    print("=" * 60)
    
    original = torch.tensor([1, 2, 3, 4, 5])
    print(f"\nOriginal: {original}")
    
    # Modify single element
    original[0] = 10
    print(f"After original[0] = 10: {original}")
    
    # In-place operations
    original += 1
    print(f"After original += 1: {original}")
    
    # In-place methods (with _ suffix)
    original.add_(5)
    print(f"After original.add_(5): {original}")
    
    # Modify slices
    matrix = torch.arange(12).reshape(3, 4)
    print(f"\nMatrix:\n{matrix}")
    matrix[0, :] = 0  # Set first row to zeros
    print(f"After matrix[0, :] = 0:\n{matrix}")
    
    # Modify with boolean indexing
    matrix[matrix > 8] = -1
    print(f"After matrix[matrix > 8] = -1:\n{matrix}")


def main():
    """Run Topic 4: Tensor Indexing."""
    basic_indexing()
    advanced_indexing()
    boolean_indexing()
    modifying_tensors()
    
    print("\n" + "=" * 60)
    print("✓ You can now index and modify tensors!")
    print("=" * 60)


if __name__ == "__main__":
    main()
