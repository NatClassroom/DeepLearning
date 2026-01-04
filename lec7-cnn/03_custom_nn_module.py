"""
Topic 3: Custom nn.Module
==========================

This module demonstrates:
1. Creating custom neural network models by subclassing nn.Module
2. Defining __init__ and forward methods
3. Building complex architectures
4. Best practices for custom modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A simple custom CNN model using nn.Module.
    
    Architecture:
    - Conv2d: 3 -> 16 channels, 3x3 kernel
    - ReLU activation
    - MaxPool2d: 2x2 pooling
    - Conv2d: 16 -> 32 channels, 3x3 kernel
    - ReLU activation
    - MaxPool2d: 2x2 pooling
    - Flatten
    - Linear: 32*8*8 -> 10 outputs
    """
    
    def __init__(self, num_classes=10):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of output classes
        """
        # Call parent class constructor
        super(SimpleCNN, self).__init__()
        
        # Define layers
        # Conv layer 1: 3 input channels -> 16 output channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, 
                              kernel_size=3, stride=1, padding=1)
        
        # Pooling layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv layer 2: 16 -> 32 channels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                              kernel_size=3, stride=1, padding=1)
        
        # Pooling layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layer
        # After 2 pooling layers with stride=2, 32x32 -> 8x8
        # So: 32 channels * 8 * 8 = 2048 features
        self.fc = nn.Linear(in_features=32 * 8 * 8, out_features=num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)      # (B, 3, 32, 32) -> (B, 16, 32, 32)
        x = F.relu(x)           # ReLU activation
        x = self.pool1(x)       # (B, 16, 32, 32) -> (B, 16, 16, 16)
        
        # Conv block 2
        x = self.conv2(x)       # (B, 16, 16, 16) -> (B, 32, 16, 16)
        x = F.relu(x)           # ReLU activation
        x = self.pool2(x)       # (B, 32, 16, 16) -> (B, 32, 8, 8)
        
        # Flatten: (B, 32, 8, 8) -> (B, 32*8*8) = (B, 2048)
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch
        # Alternative: x = torch.flatten(x, 1)
        
        # Fully connected layer
        x = self.fc(x)          # (B, 2048) -> (B, num_classes)
        
        return x


def demonstrate_simple_cnn():
    """
    Demonstrates the SimpleCNN model.
    """
    print("\n" + "=" * 60)
    print("Simple Custom CNN Model")
    print("=" * 60)
    
    # Create model
    model = SimpleCNN(num_classes=10)
    print(f"\nModel created: {model.__class__.__name__}")
    
    # Print model architecture
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    print(f"\nInput shape: {dummy_input.shape}")
    
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Classes: {output.shape[1]}")


class CNNWithBatchNorm(nn.Module):
    """
    A CNN with batch normalization for better training stability.
    
    Architecture:
    - Conv2d + BatchNorm + ReLU + MaxPool
    - Conv2d + BatchNorm + ReLU + MaxPool
    - Flatten
    - Linear + ReLU (hidden layer)
    - Linear (output layer)
    """
    
    def __init__(self, num_classes=10):
        super(CNNWithBatchNorm, self).__init__()
        
        # Conv block 1 with batch norm
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch norm for 16 channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 2 with batch norm
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch norm for 32 channels
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Hidden layer
        self.fc2 = nn.Linear(128, num_classes)  # Output layer
    
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)  # Batch normalization
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)  # Batch normalization
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x


def demonstrate_cnn_with_batchnorm():
    """
    Demonstrates CNN with batch normalization.
    """
    print("\n" + "=" * 60)
    print("CNN with Batch Normalization")
    print("=" * 60)
    
    model = CNNWithBatchNorm(num_classes=10)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"\nKey addition: BatchNorm2d layers")
    print(f"  - Normalizes activations across batch")
    print(f"  - Improves training stability and convergence")
    print(f"  - Applied after convolution, before activation")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")


class ModularCNN(nn.Module):
    """
    A CNN with modular conv blocks for better code organization.
    """
    
    def __init__(self, num_classes=10):
        super(ModularCNN, self).__init__()
        
        # Define conv blocks as separate methods
        # Conv block 1
        self.conv_block1 = self._make_conv_block(3, 16)
        
        # Conv block 2
        self.conv_block2 = self._make_conv_block(16, 32)
        
        # Conv block 3
        self.conv_block3 = self._make_conv_block(32, 64)
        
        # Fully connected layers
        # After 3 pooling layers: 32 -> 16 -> 8 -> 4
        self.fc = nn.Linear(64 * 4 * 4, num_classes)
    
    def _make_conv_block(self, in_channels, out_channels):
        """
        Helper method to create a conv block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        
        Returns:
            Sequential module with conv, bn, relu, pool
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        # Apply conv blocks
        x = self.conv_block1(x)  # (B, 3, 32, 32) -> (B, 16, 16, 16)
        x = self.conv_block2(x)  # (B, 16, 16, 16) -> (B, 32, 8, 8)
        x = self.conv_block3(x)  # (B, 32, 8, 8) -> (B, 64, 4, 4)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc(x)
        
        return x


def demonstrate_modular_cnn():
    """
    Demonstrates modular CNN architecture.
    """
    print("\n" + "=" * 60)
    print("Modular CNN Architecture")
    print("=" * 60)
    
    model = ModularCNN(num_classes=10)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"\nKey feature: Modular design with helper methods")
    print(f"  - _make_conv_block() creates reusable conv blocks")
    print(f"  - Cleaner, more maintainable code")
    print(f"  - Easy to add more layers")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")


def demonstrate_model_usage():
    """
    Demonstrates how to use custom models: training, evaluation, etc.
    """
    print("\n" + "=" * 60)
    print("Using Custom Models")
    print("=" * 60)
    
    model = SimpleCNN(num_classes=10)
    
    # Set model to training mode
    model.train()
    print(f"\n1. Training mode: model.train()")
    print(f"   - BatchNorm uses batch statistics")
    print(f"   - Dropout is active")
    
    # Set model to evaluation mode
    model.eval()
    print(f"\n2. Evaluation mode: model.eval()")
    print(f"   - BatchNorm uses running statistics")
    print(f"   - Dropout is inactive")
    
    # Get model parameters
    print(f"\n3. Accessing parameters:")
    print(f"   - model.parameters(): all parameters")
    print(f"   - model.named_parameters(): parameters with names")
    
    # Count parameters per layer
    print(f"\n4. Parameters per layer:")
    for name, param in model.named_parameters():
        print(f"   {name}: {param.shape} ({param.numel()} parameters)")
    
    # Forward pass
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    print(f"\n5. Forward pass:")
    print(f"   Input: {dummy_input.shape}")
    print(f"   Output: {output.shape}")
    
    # Get intermediate activations (if needed)
    print(f"\n6. Intermediate activations:")
    print(f"   - Can modify forward() to return intermediate values")
    print(f"   - Useful for visualization and debugging")


def demonstrate_best_practices():
    """
    Demonstrates best practices for creating custom nn.Module.
    """
    print("\n" + "=" * 60)
    print("Best Practices for Custom nn.Module")
    print("=" * 60)
    
    print(f"\n1. Always call super().__init__()")
    print(f"   - Required for proper initialization")
    
    print(f"\n2. Define layers in __init__()")
    print(f"   - Layers should be instance variables")
    print(f"   - PyTorch tracks parameters automatically")
    
    print(f"\n3. Implement forward() method")
    print(f"   - Defines the forward pass")
    print(f"   - Never call forward() directly, use model(x)")
    
    print(f"\n4. Use F.relu() vs nn.ReLU()")
    print(f"   - F.relu(): functional, no parameters")
    print(f"   - nn.ReLU(): layer, can be stored in __init__")
    
    print(f"\n5. Use model.train() and model.eval()")
    print(f"   - Controls behavior of BatchNorm, Dropout, etc.")
    
    print(f"\n6. Use torch.no_grad() for inference")
    print(f"   - Saves memory and speeds up evaluation")
    
    print(f"\n7. Flatten before fully connected layers")
    print(f"   - Use x.view(x.size(0), -1) or torch.flatten(x, 1)")
    
    print(f"\n8. Calculate feature map sizes carefully")
    print(f"   - Track spatial dimensions through conv/pool layers")
    print(f"   - Use formula: floor((input + 2*padding - kernel) / stride) + 1")


def demonstrate_complete_custom_module():
    """
    Demonstrates all aspects of creating custom nn.Module.
    """
    print("\n" + "=" * 60)
    print("Complete Custom nn.Module Demonstration")
    print("=" * 60)
    
    # Simple CNN
    demonstrate_simple_cnn()
    
    # CNN with batch norm
    demonstrate_cnn_with_batchnorm()
    
    # Modular CNN
    demonstrate_modular_cnn()
    
    # Model usage
    demonstrate_model_usage()
    
    # Best practices
    demonstrate_best_practices()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Subclass nn.Module to create custom models")
    print("  2. Define layers in __init__()")
    print("  3. Implement forward() for the forward pass")
    print("  4. Always call super().__init__()")
    print("  5. Use model.train() and model.eval() appropriately")
    print("  6. Track spatial dimensions through layers")
    print("  7. Use modular design for complex architectures")
    print("  8. BatchNorm improves training stability")
    print("  9. Use torch.no_grad() for inference")


if __name__ == "__main__":
    demonstrate_complete_custom_module()

