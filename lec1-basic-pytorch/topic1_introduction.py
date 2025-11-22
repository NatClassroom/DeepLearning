"""
Topic 1: Introduction to PyTorch
==================================

What is PyTorch?
- Open-source ML framework by Facebook's AI Research (FAIR)
- Used for deep learning research and production
- Python-based, free, active community

Why PyTorch?
- Dynamic computation graphs (define-by-run)
- Pythonic and intuitive API (similar to NumPy)
- Strong community and ecosystem
- Excellent for research and prototyping
- Seamless GPU acceleration
"""

import torch
import numpy as np


def check_installation():
    """Verify PyTorch installation and environment."""
    print("=" * 60)
    print("Checking PyTorch Installation")
    print("=" * 60)

    # Check PyTorch version
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ NumPy version: {np.__version__}")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA available: {cuda_available}")

    if cuda_available:
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        print(f"✓ CUDA version: {torch.version.cuda}")
    else:
        print("Using CPU (GPU optional but recommended for larger models)")


def create_first_tensor():
    """Create your first tensor - the fundamental building block!"""
    print("\n" + "=" * 60)
    print("Creating Your First Tensor")
    print("=" * 60)

    # Create a simple tensor from a list
    my_first_tensor = torch.tensor([1, 2, 3, 4, 5])
    print(f"\nTensor: {my_first_tensor}")
    print(f"Shape: {my_first_tensor.shape}")
    print(f"Data type: {my_first_tensor.dtype}")
    print(f"Device: {my_first_tensor.device}")

    return my_first_tensor


def main():
    """Run Topic 1: Introduction to PyTorch."""
    check_installation()
    create_first_tensor()

    print("\n" + "=" * 60)
    print("✓ Setup complete! Ready to learn PyTorch!")
    print("=" * 60)


if __name__ == "__main__":
    main()
