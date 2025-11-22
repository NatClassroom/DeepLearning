"""
Topic 5: Tensor Manipulation
=============================

Reshape: .reshape() or .view() to change shape
Transpose: .T, .transpose(), or .permute() to reorder dimensions
Concatenate: torch.cat() to join along existing dimension
Stack: torch.stack() to create new dimension
"""

import torch
import numpy as np


def reshaping():
    """Reshape tensors to different shapes."""
    print("=" * 60)
    print("Reshaping Tensors")
    print("=" * 60)

    original = torch.arange(12)
    print(f"\nOriginal: {original} (shape: {original.shape})")

    # Reshape to different shapes
    reshaped_3x4 = original.reshape(3, 4)
    print(f"\nReshaped (3, 4):\n{reshaped_3x4}")

    reshaped_2x6 = original.reshape(2, 6)
    print(f"\nReshaped (2, 6):\n{reshaped_2x6}")

    # Using view (faster, requires contiguous memory)
    viewed = original.view(3, 4)
    print(f"\nUsing view (3, 4):\n{viewed}")

    # Auto-calculate dimension with -1
    auto = original.reshape(3, -1)
    print(f"\nReshape with -1 (auto):\n{auto} (shape: {auto.shape})")

    # Flatten
    flattened = reshaped_3x4.flatten()
    print(f"\nFlattened: {flattened}")


def transposing():
    """Transpose and permute tensors."""
    print("\n" + "=" * 60)
    print("Transposing Tensors")
    print("=" * 60)

    matrix = torch.arange(12).reshape(3, 4)
    print(f"\nOriginal:\n{matrix} (shape: {matrix.shape})")

    # Transpose
    transposed = matrix.T
    print(f"\nTransposed (.T):\n{transposed} (shape: {transposed.shape})")

    # Alternative
    transposed_alt = matrix.transpose(0, 1)
    print(f"\nUsing transpose(0, 1):\n{transposed_alt}")

    # Permute for higher dimensions
    tensor_3d = torch.rand(2, 3, 4)
    print(f"\n3D tensor: shape {tensor_3d.shape}")
    permuted = tensor_3d.permute(2, 1, 0)
    print(f"After permute(2, 1, 0): shape {permuted.shape}")


def concatenating_stacking():
    """Concatenate and stack tensors."""
    print("\n" + "=" * 60)
    print("Concatenating and Stacking")
    print("=" * 60)

    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])

    print(f"\nA:\n{a} (shape: {a.shape})")
    print(f"\nB:\n{b} (shape: {b.shape})")

    # Concatenate along dimension 0 (vertically)
    cat_0 = torch.cat([a, b], dim=0)
    print(f"\nConcatenate dim=0:\n{cat_0} (shape: {cat_0.shape})")

    # Concatenate along dimension 1 (horizontally)
    cat_1 = torch.cat([a, b], dim=1)
    print(f"\nConcatenate dim=1:\n{cat_1} (shape: {cat_1.shape})")

    # Stack (creates new dimension)
    stacked = torch.stack([a, b], dim=0)
    print(f"\nStack dim=0:\n{stacked} (shape: {stacked.shape})")

    stacked_1 = torch.stack([a, b], dim=1)
    print(f"\nStack dim=1:\n{stacked_1} (shape: {stacked_1.shape})")


def splitting():
    """Split tensors into chunks."""
    print("\n" + "=" * 60)
    print("Splitting Tensors")
    print("=" * 60)

    tensor = torch.arange(12).reshape(3, 4)
    print(f"\nTensor:\n{tensor}")

    # Split into chunks
    chunks = torch.chunk(tensor, 3, dim=0)
    print(f"\nSplit into 3 chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}:\n{chunk}")

    # Split with specific sizes
    parts = torch.split(tensor, 1, dim=0)
    print(f"\nSplit into parts of size 1:")
    for i, part in enumerate(parts):
        print(f"Part {i}:\n{part}")


def squeezing_unsqueezing():
    """Add or remove dimensions of size 1."""
    print("\n" + "=" * 60)
    print("Squeezing and Unsqueezing")
    print("=" * 60)

    # Unsqueeze (add dimension)
    tensor_1d = torch.tensor([1, 2, 3, 4])
    print(f"\n1D tensor: {tensor_1d} (shape: {tensor_1d.shape})")

    unsqueezed_0 = tensor_1d.unsqueeze(0)
    print(f"After unsqueeze(0): {unsqueezed_0} (shape: {unsqueezed_0.shape})")

    unsqueezed_1 = tensor_1d.unsqueeze(1)
    print(f"After unsqueeze(1): {unsqueezed_1} (shape: {unsqueezed_1.shape})")

    # Squeeze (remove dimensions of size 1)
    tensor_with_ones = torch.rand(1, 3, 1, 4)
    print(f"\nTensor with ones: shape {tensor_with_ones.shape}")

    squeezed = tensor_with_ones.squeeze()
    print(f"After squeeze(): shape {squeezed.shape}")

    squeezed_0 = tensor_with_ones.squeeze(0)
    print(f"After squeeze(0): shape {squeezed_0.shape}")


def main():
    """Run Topic 5: Tensor Manipulation."""
    reshaping()
    transposing()
    concatenating_stacking()
    splitting()
    squeezing_unsqueezing()

    print("\n" + "=" * 60)
    print("✓ You can now manipulate tensor shapes!")
    print("=" * 60)


if __name__ == "__main__":
    main()
