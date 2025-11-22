"""
Setup and Verification Script for PyTorch
==========================================
Run this script to verify your PyTorch installation is working correctly.
"""

import torch
import numpy as np

print("=" * 60)
print("PYTORCH SETUP VERIFICATION")
print("=" * 60)

# Check PyTorch version
print(f"\n✓ PyTorch version: {torch.__version__}")

# Check NumPy version
print(f"✓ NumPy version: {np.__version__}")

# Check CUDA availability
print(f"\n✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
    print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
else:
    print("  (You can still use PyTorch on CPU)")

# Test basic tensor operations
print("\n" + "-" * 60)
print("Testing Basic Tensor Operations")
print("-" * 60)

# Create a simple tensor
test_tensor = torch.tensor([1, 2, 3, 4, 5])
print(f"✓ Created tensor: {test_tensor}")

# Test operations
result = test_tensor * 2
print(f"✓ Tensor multiplication: {result}")

# Test with NumPy
numpy_array = np.array([1, 2, 3])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"✓ NumPy to Tensor conversion: {tensor_from_numpy}")

print("\n" + "=" * 60)
print("✓ All checks passed! PyTorch is ready to use.")
print("=" * 60)
