"""
Topic 2: Max Pooling Layer
===========================

This module demonstrates:
1. Understanding MaxPool2d operations
2. How pooling reduces spatial dimensions
3. Pooling arguments (kernel_size, stride, padding)
4. Effect of pooling on feature maps
"""

import torch
import torch.nn as nn
import numpy as np


def demonstrate_basic_max_pooling():
    """
    Demonstrates basic max pooling operation.
    """
    print("\n" + "=" * 60)
    print("Basic Max Pooling Operation")
    print("=" * 60)
    
    # Create a simple feature map
    # Shape: (batch_size, channels, height, width)
    input_tensor = torch.tensor([
        [[[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
          [13, 14, 15, 16]]]
    ], dtype=torch.float32)
    
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Input tensor:\n{input_tensor.squeeze()}")
    
    # Create MaxPool2d layer
    # Arguments:
    #   kernel_size: Size of the pooling window (2x2)
    #   stride: Step size (default: same as kernel_size)
    #   padding: Zero-padding (default: 0)
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    
    print(f"\nMaxPool2d layer:")
    print(f"  kernel_size: {pool.kernel_size}")
    print(f"  stride: {pool.stride}")
    print(f"  padding: {pool.padding}")
    
    # Apply pooling
    output = pool(input_tensor)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output tensor:\n{output.squeeze()}")
    
    print(f"\nHow it works:")
    print(f"  - 2x2 window slides over input")
    print(f"  - Takes maximum value in each window")
    print(f"  - Window 1 (top-left): max(1,2,5,6) = 6")
    print(f"  - Window 2 (top-right): max(3,4,7,8) = 8")
    print(f"  - Window 3 (bottom-left): max(9,10,13,14) = 14")
    print(f"  - Window 4 (bottom-right): max(11,12,15,16) = 16")
    
    # Output shape calculation:
    # For MaxPool2d: output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
    # Height: floor((4 + 2*0 - 2) / 2) + 1 = floor(1) + 1 = 2
    # Width: floor((4 + 2*0 - 2) / 2) + 1 = floor(1) + 1 = 2
    print(f"\nOutput shape calculation:")
    print(f"  Input: 4x4, Kernel: 2x2, Padding: 0, Stride: 2")
    print(f"  Output = floor((4 + 2*0 - 2) / 2) + 1 = 2x2")


def demonstrate_pooling_stride():
    """
    Demonstrates different stride values in pooling.
    """
    print("\n" + "=" * 60)
    print("Pooling with Different Strides")
    print("=" * 60)
    
    input_tensor = torch.randn(1, 1, 8, 8)
    print(f"\nInput shape: {input_tensor.shape}")
    
    # Stride = 2 (default, same as kernel_size)
    pool_stride2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    output_stride2 = pool_stride2(input_tensor)
    print(f"\nKernel=2x2, Stride=2:")
    print(f"  Output shape: {output_stride2.shape}")
    print(f"  Calculation: floor((8 + 2*0 - 2) / 2) + 1 = 4x4")
    print(f"  Note: Halves the spatial dimensions")
    
    # Stride = 1 (overlapping windows)
    pool_stride1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    output_stride1 = pool_stride1(input_tensor)
    print(f"\nKernel=2x2, Stride=1:")
    print(f"  Output shape: {output_stride1.shape}")
    print(f"  Calculation: floor((8 + 2*0 - 2) / 1) + 1 = 7x7")
    print(f"  Note: Overlapping windows, less downsampling")
    
    # Stride = 4
    pool_stride4 = nn.MaxPool2d(kernel_size=2, stride=4, padding=0)
    output_stride4 = pool_stride4(input_tensor)
    print(f"\nKernel=2x2, Stride=4:")
    print(f"  Output shape: {output_stride4.shape}")
    print(f"  Calculation: floor((8 + 2*0 - 2) / 4) + 1 = 2x2")
    print(f"  Note: More aggressive downsampling")


def demonstrate_pooling_kernel_sizes():
    """
    Demonstrates different kernel sizes in pooling.
    """
    print("\n" + "=" * 60)
    print("Pooling with Different Kernel Sizes")
    print("=" * 60)
    
    input_tensor = torch.randn(1, 1, 8, 8)
    print(f"\nInput shape: {input_tensor.shape}")
    
    kernel_sizes = [2, 3, 4]
    
    print(f"\nKernel Size | Output Shape | Downsampling Factor")
    print(f"-" * 55)
    
    for kernel_size in kernel_sizes:
        pool = nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size, padding=0)
        output = pool(input_tensor)
        output_size = output.shape[-1]
        factor = 8 / output_size
        print(f"    {kernel_size}x{kernel_size}   |    {output_size}x{output_size}    |        {factor:.1f}x")
    
    print(f"\nKey insight: Larger kernel = more downsampling")


def demonstrate_pooling_padding():
    """
    Demonstrates padding in pooling operations.
    """
    print("\n" + "=" * 60)
    print("Padding in Pooling")
    print("=" * 60)
    
    input_tensor = torch.randn(1, 1, 5, 5)
    print(f"\nInput shape: {input_tensor.shape}")
    
    # Without padding
    pool_no_padding = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    output_no_padding = pool_no_padding(input_tensor)
    print(f"\nWithout padding (padding=0):")
    print(f"  Output shape: {output_no_padding.shape}")
    print(f"  Calculation: floor((5 + 2*0 - 2) / 2) + 1 = 2x2")
    
    # With padding=1
    pool_with_padding = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    output_with_padding = pool_with_padding(input_tensor)
    print(f"\nWith padding (padding=1):")
    print(f"  Output shape: {output_with_padding.shape}")
    print(f"  Calculation: floor((5 + 2*1 - 2) / 2) + 1 = 3x3")
    print(f"  Note: Padding can preserve more spatial information")


def demonstrate_pooling_vs_convolution():
    """
    Compares pooling and convolution operations.
    """
    print("\n" + "=" * 60)
    print("Pooling vs Convolution")
    print("=" * 60)
    
    input_tensor = torch.randn(1, 16, 32, 32)  # 16 channels, 32x32 spatial
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"  - Channels: 16")
    print(f"  - Spatial: 32x32")
    
    # Max Pooling: reduces spatial dimensions, preserves channels
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    pooled_output = pool(input_tensor)
    print(f"\nMaxPool2d(kernel_size=2, stride=2):")
    print(f"  Output shape: {pooled_output.shape}")
    print(f"  - Channels: 16 (preserved)")
    print(f"  - Spatial: 16x16 (reduced by 2x)")
    print(f"  - Parameters: 0 (no learnable parameters)")
    print(f"  - Operation: Takes maximum in each window")
    
    # Convolution: can change channels and spatial dimensions
    conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
    conv_output = conv(input_tensor)
    print(f"\nConv2d(in=16, out=32, kernel=3, stride=2, padding=1):")
    print(f"  Output shape: {conv_output.shape}")
    print(f"  - Channels: 32 (changed)")
    print(f"  - Spatial: 16x16 (reduced by 2x)")
    print(f"  - Parameters: {sum(p.numel() for p in conv.parameters())} (learnable)")
    print(f"  - Operation: Learned feature extraction")
    
    print(f"\nKey differences:")
    print(f"  1. Pooling: Fixed operation (max/avg), no parameters")
    print(f"  2. Convolution: Learned operation, has parameters")
    print(f"  3. Pooling: Only reduces spatial size, preserves channels")
    print(f"  4. Convolution: Can change both channels and spatial size")
    print(f"  5. Pooling: Faster, provides translation invariance")
    print(f"  6. Convolution: More flexible, learns features")


def demonstrate_pooling_in_sequence():
    """
    Demonstrates multiple pooling layers in sequence.
    """
    print("\n" + "=" * 60)
    print("Multiple Pooling Layers in Sequence")
    print("=" * 60)
    
    input_tensor = torch.randn(1, 3, 64, 64)  # RGB image: 64x64
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"  - Channels: 3 (RGB)")
    print(f"  - Spatial: 64x64")
    
    # First pooling
    pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    output1 = pool1(input_tensor)
    print(f"\nAfter Pool1 (2x2, stride=2):")
    print(f"  Output shape: {output1.shape}")
    print(f"  - Channels: 3 (preserved)")
    print(f"  - Spatial: 32x32 (reduced by 2x)")
    
    # Second pooling
    pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    output2 = pool2(output1)
    print(f"\nAfter Pool2 (2x2, stride=2):")
    print(f"  Output shape: {output2.shape}")
    print(f"  - Channels: 3 (preserved)")
    print(f"  - Spatial: 16x16 (reduced by 2x)")
    
    # Third pooling
    pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    output3 = pool3(output2)
    print(f"\nAfter Pool3 (2x2, stride=2):")
    print(f"  Output shape: {output3.shape}")
    print(f"  - Channels: 3 (preserved)")
    print(f"  - Spatial: 8x8 (reduced by 2x)")
    
    print(f"\nTotal downsampling: 64x64 -> 8x8 (8x reduction)")
    print(f"  - Each pooling layer reduces by 2x")
    print(f"  - 3 pooling layers: 2^3 = 8x total reduction")


def demonstrate_adaptive_pooling():
    """
    Demonstrates adaptive pooling (outputs fixed size regardless of input).
    """
    print("\n" + "=" * 60)
    print("Adaptive Pooling")
    print("=" * 60)
    
    # Different input sizes
    inputs = [
        torch.randn(1, 16, 32, 32),
        torch.randn(1, 16, 64, 64),
        torch.randn(1, 16, 128, 128),
    ]
    
    # Adaptive average pooling: always outputs specified size
    adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
    
    print(f"\nAdaptiveAvgPool2d(output_size=(7, 7))")
    print(f"  - Always outputs 7x7 regardless of input size")
    print(f"\nInput Size | Output Size")
    print(f"-" * 30)
    
    for inp in inputs:
        output = adaptive_pool(inp)
        print(f"  {inp.shape[2]}x{inp.shape[3]}   |   {output.shape[2]}x{output.shape[3]}")
    
    print(f"\nUse case: Final pooling before fully connected layers")
    print(f"  - Ensures consistent input size to FC layers")
    print(f"  - Common in transfer learning (e.g., 7x7 -> FC)")


def demonstrate_complete_pooling():
    """
    Demonstrates all key aspects of max pooling.
    """
    print("\n" + "=" * 60)
    print("Complete Max Pooling Demonstration")
    print("=" * 60)
    
    # Basic pooling
    demonstrate_basic_max_pooling()
    
    # Different strides
    demonstrate_pooling_stride()
    
    # Different kernel sizes
    demonstrate_pooling_kernel_sizes()
    
    # Padding
    demonstrate_pooling_padding()
    
    # Pooling vs convolution
    demonstrate_pooling_vs_convolution()
    
    # Multiple pooling layers
    demonstrate_pooling_in_sequence()
    
    # Adaptive pooling
    demonstrate_adaptive_pooling()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. MaxPool2d(kernel_size, stride, padding)")
    print("  2. Output shape = floor((input + 2*padding - kernel) / stride) + 1")
    print("  3. Reduces spatial dimensions (downsampling)")
    print("  4. Preserves number of channels")
    print("  5. No learnable parameters (fixed operation)")
    print("  6. Provides translation invariance")
    print("  7. Common: kernel_size=2, stride=2 (halves dimensions)")
    print("  8. Adaptive pooling: outputs fixed size regardless of input")


if __name__ == "__main__":
    demonstrate_complete_pooling()

