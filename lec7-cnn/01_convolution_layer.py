"""
Topic 1: PyTorch Convolution Layer
===================================

This module demonstrates:
1. Understanding Conv2d layer arguments (kernel_size, padding, stride, etc.)
2. How convolution operations work
3. Output shape calculations
4. Visualizing convolution effects
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def demonstrate_basic_convolution():
    """
    Demonstrates a basic convolution operation with Conv2d.
    Shows how input shape transforms to output shape.
    """
    print("\n" + "=" * 60)
    print("Basic Convolution Operation")
    print("=" * 60)
    
    # Create a simple input image: batch_size=1, channels=1, height=5, width=5
    # Shape: (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 1, 5, 5)
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Input tensor:\n{input_tensor.squeeze()}")
    
    # Create a Conv2d layer
    # Arguments:
    #   in_channels: Number of input channels (1 for grayscale)
    #   out_channels: Number of output channels (filters/feature maps)
    #   kernel_size: Size of the convolution kernel (3x3)
    #   stride: Step size for convolution (default: 1)
    #   padding: Zero-padding added to input (default: 0)
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
    
    print(f"\nConvolution layer:")
    print(f"  in_channels: {conv.in_channels}")
    print(f"  out_channels: {conv.out_channels}")
    print(f"  kernel_size: {conv.kernel_size}")
    print(f"  stride: {conv.stride}")
    print(f"  padding: {conv.padding}")
    print(f"  Weight shape: {conv.weight.shape}")  # (out_channels, in_channels, kernel_h, kernel_w)
    print(f"  Bias shape: {conv.bias.shape}")
    
    # Apply convolution
    output = conv(input_tensor)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output tensor:\n{output.squeeze()}")
    
    # Output shape calculation:
    # For Conv2d: output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
    # Height: floor((5 + 2*0 - 3) / 1) + 1 = floor(2) + 1 = 3
    # Width: floor((5 + 2*0 - 3) / 1) + 1 = floor(2) + 1 = 3
    print(f"\nOutput shape calculation:")
    print(f"  Input: 5x5, Kernel: 3x3, Padding: 0, Stride: 1")
    print(f"  Output = floor((5 + 2*0 - 3) / 1) + 1 = 3x3")


def demonstrate_padding():
    """
    Demonstrates how padding affects output size.
    Padding preserves spatial dimensions.
    """
    print("\n" + "=" * 60)
    print("Padding in Convolution")
    print("=" * 60)
    
    input_tensor = torch.randn(1, 1, 5, 5)
    print(f"\nInput shape: {input_tensor.shape}")
    
    # Without padding
    conv_no_padding = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
    output_no_padding = conv_no_padding(input_tensor)
    print(f"\nWithout padding (padding=0):")
    print(f"  Output shape: {output_no_padding.shape}")
    
    # With padding=1 (adds 1 pixel on all sides)
    conv_with_padding = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
    output_with_padding = conv_with_padding(input_tensor)
    print(f"\nWith padding (padding=1):")
    print(f"  Output shape: {output_with_padding.shape}")
    print(f"  Note: Output size matches input size (5x5)")
    
    # With padding=2
    conv_padding2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=2)
    output_padding2 = conv_padding2(input_tensor)
    print(f"\nWith padding (padding=2):")
    print(f"  Output shape: {output_padding2.shape}")
    print(f"  Note: Output is larger than input (7x7)")
    
    print(f"\nKey insight: Padding preserves or increases spatial dimensions")
    print(f"  padding='same' (or padding=kernel_size//2) keeps output size = input size")


def demonstrate_stride():
    """
    Demonstrates how stride affects output size.
    Stride > 1 reduces spatial dimensions (downsampling).
    """
    print("\n" + "=" * 60)
    print("Stride in Convolution")
    print("=" * 60)
    
    input_tensor = torch.randn(1, 1, 8, 8)
    print(f"\nInput shape: {input_tensor.shape}")
    
    # Stride = 1 (default)
    conv_stride1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
    output_stride1 = conv_stride1(input_tensor)
    print(f"\nStride = 1:")
    print(f"  Output shape: {output_stride1.shape}")
    print(f"  Calculation: floor((8 + 2*0 - 3) / 1) + 1 = 6x6")
    
    # Stride = 2
    conv_stride2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=0)
    output_stride2 = conv_stride2(input_tensor)
    print(f"\nStride = 2:")
    print(f"  Output shape: {output_stride2.shape}")
    print(f"  Calculation: floor((8 + 2*0 - 3) / 2) + 1 = 3x3")
    print(f"  Note: Stride=2 halves the spatial dimensions")
    
    # Stride = 3
    conv_stride3 = nn.Conv2d(1, 1, kernel_size=3, stride=3, padding=0)
    output_stride3 = conv_stride3(input_tensor)
    print(f"\nStride = 3:")
    print(f"  Output shape: {output_stride3.shape}")
    print(f"  Calculation: floor((8 + 2*0 - 3) / 3) + 1 = 2x2")
    
    print(f"\nKey insight: Larger stride = more downsampling = smaller output")


def demonstrate_multiple_channels():
    """
    Demonstrates convolution with multiple input and output channels.
    """
    print("\n" + "=" * 60)
    print("Multiple Channels in Convolution")
    print("=" * 60)
    
    # RGB image: 3 input channels
    input_tensor = torch.randn(1, 3, 5, 5)  # (batch, channels, height, width)
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"  - Batch size: 1")
    print(f"  - Input channels: 3 (RGB)")
    print(f"  - Spatial size: 5x5")
    
    # Convolution: 3 input channels -> 16 output channels
    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    output = conv(input_tensor)
    print(f"\nConvolution layer:")
    print(f"  in_channels: 3")
    print(f"  out_channels: 16")
    print(f"  kernel_size: 3x3")
    
    print(f"\nOutput shape: {output.shape}")
    print(f"  - Batch size: 1")
    print(f"  - Output channels: 16 (feature maps)")
    print(f"  - Spatial size: 5x5 (preserved due to padding=1)")
    
    print(f"\nWeight shape: {conv.weight.shape}")
    print(f"  - Shape: (out_channels, in_channels, kernel_h, kernel_w)")
    print(f"  - Each of 16 output channels has 3 kernels (one per input channel)")
    print(f"  - Total parameters: {conv.weight.numel() + conv.bias.numel()}")


def demonstrate_kernel_sizes():
    """
    Demonstrates different kernel sizes and their effects.
    """
    print("\n" + "=" * 60)
    print("Different Kernel Sizes")
    print("=" * 60)
    
    input_tensor = torch.randn(1, 1, 8, 8)
    print(f"\nInput shape: {input_tensor.shape}")
    
    kernel_sizes = [1, 3, 5, 7]
    
    print(f"\nKernel Size | Output Shape | Receptive Field")
    print(f"-" * 50)
    
    for kernel_size in kernel_sizes:
        conv = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=0)
        output = conv(input_tensor)
        output_size = output.shape[-1]  # Assuming square
        print(f"    {kernel_size}x{kernel_size}   |    {output_size}x{output_size}    |      {kernel_size}x{kernel_size}")
    
    print(f"\nKey insights:")
    print(f"  - Larger kernels: capture more spatial context, fewer output pixels")
    print(f"  - Smaller kernels: capture local patterns, more output pixels")
    print(f"  - 3x3 is most common (good balance of context vs. efficiency)")


def demonstrate_output_shape_formula():
    """
    Demonstrates the output shape formula for Conv2d.
    """
    print("\n" + "=" * 60)
    print("Output Shape Formula")
    print("=" * 60)
    
    print(f"\nFormula for Conv2d output size:")
    print(f"  output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1")
    
    print(f"\nExamples:")
    print(f"-" * 60)
    
    examples = [
        (32, 3, 1, 0, "Common: 32x32 input, 3x3 kernel, stride=1, no padding"),
        (32, 3, 1, 1, "Common: 32x32 input, 3x3 kernel, stride=1, padding=1 (same)"),
        (32, 3, 2, 1, "Downsampling: 32x32 input, 3x3 kernel, stride=2, padding=1"),
        (224, 7, 2, 3, "Large input: 224x224 input, 7x7 kernel, stride=2, padding=3"),
    ]
    
    for input_size, kernel, stride, padding, description in examples:
        output_size = ((input_size + 2 * padding - kernel) // stride) + 1
        print(f"\n{description}")
        print(f"  Input: {input_size}x{input_size}")
        print(f"  Kernel: {kernel}x{kernel}, Stride: {stride}, Padding: {padding}")
        print(f"  Output: {output_size}x{output_size}")
        print(f"  Calculation: floor(({input_size} + 2*{padding} - {kernel}) / {stride}) + 1 = {output_size}")


def demonstrate_convolution_on_image():
    """
    Demonstrates applying convolution to a real image.
    """
    print("\n" + "=" * 60)
    print("Convolution on Real Image")
    print("=" * 60)
    
    try:
        import imageio.v3 as iio
        import os
        
        # Load an image
        data_dir = 'data/2d_data/'
        if os.path.exists(data_dir):
            filenames = [name for name in os.listdir(data_dir)
                        if os.path.splitext(name)[-1] == '.png']
            if filenames:
                img_path = os.path.join(data_dir, filenames[0])
                img_array = iio.imread(img_path)
                print(f"\nLoaded image: {img_path}")
                print(f"Original image shape: {img_array.shape}")
                
                # Convert to tensor: (H, W, C) -> (C, H, W)
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
                # Normalize to [0, 1]
                img_tensor = img_tensor / 255.0
                # Add batch dimension: (C, H, W) -> (1, C, H, W)
                img_tensor = img_tensor.unsqueeze(0)
                
                print(f"Tensor shape: {img_tensor.shape}")
                
                # Apply convolution
                conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
                output = conv(img_tensor)
                
                print(f"\nAfter convolution:")
                print(f"  Output shape: {output.shape}")
                print(f"  - Batch: 1")
                print(f"  - Channels: 16 (feature maps)")
                print(f"  - Spatial: {output.shape[2]}x{output.shape[3]}")
                
                return img_tensor, output
        else:
            print(f"\nImage data directory not found: {data_dir}")
            print(f"Skipping image demonstration")
    except ImportError:
        print(f"\nimageio not available, skipping image demonstration")
    except Exception as e:
        print(f"\nError loading image: {e}")
    
    return None, None


def demonstrate_complete_convolution():
    """
    Demonstrates all key aspects of convolution layers.
    """
    print("\n" + "=" * 60)
    print("Complete Convolution Layer Demonstration")
    print("=" * 60)
    
    # Basic convolution
    demonstrate_basic_convolution()
    
    # Padding
    demonstrate_padding()
    
    # Stride
    demonstrate_stride()
    
    # Multiple channels
    demonstrate_multiple_channels()
    
    # Kernel sizes
    demonstrate_kernel_sizes()
    
    # Output shape formula
    demonstrate_output_shape_formula()
    
    # Real image example
    demonstrate_convolution_on_image()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Conv2d(input_channels, output_channels, kernel_size, stride, padding)")
    print("  2. Output shape = floor((input + 2*padding - kernel) / stride) + 1")
    print("  3. Padding preserves spatial dimensions (padding='same' keeps size)")
    print("  4. Stride > 1 reduces spatial dimensions (downsampling)")
    print("  5. Multiple channels allow learning complex feature combinations")
    print("  6. Kernel size controls receptive field (larger = more context)")
    print("  7. Weight shape: (out_channels, in_channels, kernel_h, kernel_w)")


if __name__ == "__main__":
    demonstrate_complete_convolution()

