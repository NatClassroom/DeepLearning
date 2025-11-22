"""
Topic 2: Understanding Tensors
===============================

Tensors are multi-dimensional arrays (like NumPy but with GPU support)
- 0D = scalar, 1D = vector, 2D = matrix, 3D+ = higher dimensions
- Can run on GPU for faster computation
- Support automatic differentiation (for gradients)
"""

import torch
import numpy as np


def explore_tensor_dimensions():
    """See tensors in different dimensions."""
    print("=" * 60)
    print("Tensor Dimensions")
    print("=" * 60)

    # 0D tensor (scalar)
    scalar = torch.tensor(5)
    print(f"\n0D (scalar): {scalar}, shape: {scalar.shape}")

    # 1D tensor (vector)
    vector = torch.tensor([1, 2, 3, 4, 5])
    print(f"1D (vector): {vector}, shape: {vector.shape}")

    # 2D tensor (matrix)
    matrix = torch.tensor([[1, 2], [3, 4]])
    print(f"2D (matrix):\n{matrix}, shape: {matrix.shape}")

    # 3D tensor
    tensor_3d = torch.rand(2, 3, 4)
    print(f"3D tensor: shape {tensor_3d.shape}")


def create_tensors():
    """Different ways to create tensors."""
    print("\n" + "=" * 60)
    print("Creating Tensors")
    print("=" * 60)

    # Zeros
    zeros = torch.zeros(3, 4)
    print(f"\nZeros (3x4):\n{zeros}")

    # Ones
    ones = torch.ones(2, 3)
    print(f"\nOnes (2x3):\n{ones}")

    # Random (uniform distribution)
    random = torch.rand(2, 3)
    print(f"\nRandom [0, 1):\n{random}")

    # Random (normal distribution)
    normal = torch.randn(2, 3)
    print(f"\nRandom (normal):\n{normal}")

    # From list
    from_list = torch.tensor([1, 2, 3, 4, 5])
    print(f"\nFrom list: {from_list}")

    # Filled with specific value
    filled = torch.full((2, 3), 7.0)
    print(f"\nFilled with 7.0:\n{filled}")

    # Range
    range_tensor = torch.arange(0, 10, 2)
    print(f"\nRange (0-10, step 2): {range_tensor}")

    # Linspace
    linspace = torch.linspace(0, 1, 5)
    print(f"\nLinspace (0-1, 5 points): {linspace}")


def tensor_properties():
    """Explore tensor properties."""
    print("\n" + "=" * 60)
    print("Tensor Properties")
    print("=" * 60)

    tensor = torch.rand(3, 4, 5)

    print(f"\nTensor shape: {tensor.shape}")
    print(f"Tensor size: {tensor.size()}")
    print(f"Data type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Number of elements: {tensor.numel()}")
    print(f"Number of dimensions: {tensor.ndim}")


def numpy_to_tensor():
    """Convert NumPy arrays to tensors."""
    print("\n" + "=" * 60)
    print("NumPy to Tensor Conversion")
    print("=" * 60)

    # Create NumPy array
    numpy_array = np.array([1, 2, 3, 4, 5])
    print(f"\nNumPy array: {numpy_array} (type: {type(numpy_array).__name__})")

    # Convert to tensor
    tensor = torch.from_numpy(numpy_array)
    print(f"PyTorch tensor: {tensor} (type: {type(tensor).__name__})")


def data_types():
    """Work with different data types."""
    print("\n" + "=" * 60)
    print("Data Types")
    print("=" * 60)

    # Different types
    float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
    bool_tensor = torch.tensor([True, False, True], dtype=torch.bool)

    print(f"\nFloat32: {float_tensor} (dtype: {float_tensor.dtype})")
    print(f"Int64: {int_tensor} (dtype: {int_tensor.dtype})")
    print(f"Bool: {bool_tensor} (dtype: {bool_tensor.dtype})")

    # Type conversion
    converted = int_tensor.float()
    print(f"\nConverted int to float: {converted} (dtype: {converted.dtype})")

    # Other conversion methods
    print(f"Using .to(): {int_tensor.to(torch.float32)}")
    print(f"Using .double(): {float_tensor.double()}")


def main():
    """Run Topic 2: Understanding Tensors."""
    explore_tensor_dimensions()
    create_tensors()
    tensor_properties()
    numpy_to_tensor()
    data_types()

    print("\n" + "=" * 60)
    print("✓ You now understand tensors!")
    print("=" * 60)


if __name__ == "__main__":
    main()
