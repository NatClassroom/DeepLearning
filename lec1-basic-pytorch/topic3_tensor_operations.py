"""
Topic 3: Tensor Operations
===========================

Basic operations: +, -, *, /, ** (all element-wise)
Matrix multiplication: @ or torch.matmul()
Broadcasting: automatic shape expansion for operations
"""

import torch
import numpy as np


def basic_arithmetic():
    """Basic arithmetic operations."""
    print("=" * 60)
    print("Basic Arithmetic")
    print("=" * 60)
    
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    
    print(f"\na = {a}")
    print(f"b = {b}")
    print(f"\na + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * b = {a * b}")
    print(f"a / b = {a / b}")
    print(f"a ** 2 = {a ** 2}")
    
    # Alternative: using torch functions
    print(f"\nUsing torch functions:")
    print(f"torch.add(a, b) = {torch.add(a, b)}")
    print(f"torch.mul(a, b) = {torch.mul(a, b)}")


def elementwise_operations():
    """Element-wise operations on 2D tensors."""
    print("\n" + "=" * 60)
    print("Element-wise Operations")
    print("=" * 60)
    
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    
    print(f"\nx =\n{x}")
    print(f"\ny =\n{y}")
    print(f"\nx + y =\n{x + y}")
    print(f"\nx * y =\n{x * y}")


def matrix_operations():
    """Matrix multiplication and dot product."""
    print("\n" + "=" * 60)
    print("Matrix Operations")
    print("=" * 60)
    
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    
    print(f"\nA =\n{A}")
    print(f"\nB =\n{B}")
    
    # Matrix multiplication
    print(f"\nA @ B =\n{A @ B}")
    print(f"\ntorch.matmul(A, B) =\n{torch.matmul(A, B)}")
    
    # Dot product
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    print(f"\nDot product: torch.dot(a, b) = {torch.dot(a, b)}")


def broadcasting():
    """Broadcasting: operations between different shapes."""
    print("\n" + "=" * 60)
    print("Broadcasting")
    print("=" * 60)
    print("\nBroadcasting automatically expands dimensions for operations")
    print("Rules: dimensions must match, be 1, or be missing")
    
    # 2D + 1D
    tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tensor_1d = torch.tensor([10, 20, 30])
    
    print(f"\n2D tensor:\n{tensor_2d} (shape: {tensor_2d.shape})")
    print(f"1D tensor: {tensor_1d} (shape: {tensor_1d.shape})")
    print(f"\n2D + 1D (broadcasting):\n{tensor_2d + tensor_1d}")
    
    # Scalar broadcasting
    print(f"\n2D * scalar (5):\n{tensor_2d * 5}")
    
    # Column broadcasting
    tensor_col = torch.tensor([[10], [20]])
    print(f"\nColumn tensor:\n{tensor_col} (shape: {tensor_col.shape})")
    print(f"2D + column:\n{tensor_2d + tensor_col}")


def aggregation_functions():
    """Common aggregation and statistical functions."""
    print("\n" + "=" * 60)
    print("Aggregation Functions")
    print("=" * 60)
    
    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"\nData:\n{data}")
    
    # Sum
    print(f"\nSum of all: {data.sum()}")
    print(f"Sum along dim=0 (columns): {data.sum(dim=0)}")
    print(f"Sum along dim=1 (rows): {data.sum(dim=1)}")
    
    # Mean
    print(f"\nMean of all: {data.mean()}")
    print(f"Mean along dim=0: {data.mean(dim=0)}")
    print(f"Mean along dim=1: {data.mean(dim=1)}")
    
    # Max (returns values and indices)
    max_val, max_idx = data.max(dim=0)
    print(f"\nMax along dim=0: values={max_val}, indices={max_idx}")
    
    # Min
    min_val, min_idx = data.min(dim=0)
    print(f"Min along dim=0: values={min_val}, indices={min_idx}")
    
    # Statistics
    print(f"\nStd: {data.std()}")
    print(f"Var: {data.var()}")


def math_functions():
    """Mathematical functions."""
    print("\n" + "=" * 60)
    print("Mathematical Functions")
    print("=" * 60)
    
    values = torch.tensor([4.0, 9.0, 16.0, -1.0, -2.0])
    print(f"\nValues: {values}")
    print(f"sqrt: {torch.sqrt(torch.tensor([4.0, 9.0, 16.0]))}")
    print(f"abs: {torch.abs(values)}")
    print(f"exp: {torch.exp(torch.tensor([1.0, 2.0]))}")
    print(f"log: {torch.log(torch.tensor([1.0, 2.718, 10.0]))}")


def main():
    """Run Topic 3: Tensor Operations."""
    basic_arithmetic()
    elementwise_operations()
    matrix_operations()
    broadcasting()
    aggregation_functions()
    math_functions()
    
    print("\n" + "=" * 60)
    print("✓ You can now perform tensor operations!")
    print("=" * 60)


if __name__ == "__main__":
    main()
