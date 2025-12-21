"""
Topic 6: PyTorch Optimizers
===========================

This module demonstrates:
1. Using PyTorch's built-in optimizers
2. Comparing SGD (Stochastic Gradient Descent) and Adam optimizers
3. Simplifying the training loop with optimizers
"""

import torch
import torch.optim as optim
import importlib.util
import sys
import os

# Import module with number prefix using importlib
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, "01_model_and_loss.py")
spec = importlib.util.spec_from_file_location("model_and_loss", module_path)
model_and_loss = importlib.util.module_from_spec(spec)
sys.modules["model_and_loss"] = model_and_loss
spec.loader.exec_module(model_and_loss)

linear_model = model_and_loss.linear_model
load_temperature_data = model_and_loss.load_temperature_data


def loss_fn(y, y_target):
    """
    Compute mean squared error loss.
    
    Args:
        y: Predicted values
        y_target: Target values
        
    Returns:
        Mean squared error
    """
    square_error = (y - y_target) ** 2
    return square_error.mean()


def training_loop_with_optimizer(n_epoch, params, x, y_target, optimizer):
    """
    Training loop using PyTorch optimizer.
    
    Args:
        n_epoch: Number of training epochs
        params: Initial parameters [w, b] (must have requires_grad=True)
        x: Input tensor
        y_target: Target tensor
        optimizer: PyTorch optimizer (e.g., SGD, Adam)
        
    Returns:
        Trained parameters
    """
    for epoch in range(n_epoch):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        
        w, b = params
        
        # Forward pass: compute predictions and loss
        y_predict = linear_model(x, w, b)
        loss = loss_fn(y_predict, y_target)
        
        # Backward pass: compute gradients automatically
        loss.backward()
        
        # Update parameters using optimizer
        optimizer.step()
        
        # Print progress
        if epoch % 100 == 0 or epoch == n_epoch - 1:
            print(f'epoch: {epoch}, loss: {float(loss):.4f}')
    
    return params


def demonstrate_sgd_optimizer():
    """
    Demonstrate training with SGD optimizer.
    """
    print("=" * 60)
    print("SGD Optimizer Demonstration")
    print("=" * 60)
    
    # Load data
    temp_celcius, temp_measurement = load_temperature_data()
    x = 0.1 * torch.tensor(temp_measurement.values, dtype=torch.float32)
    y_target = torch.tensor(temp_celcius.values, dtype=torch.float32)
    
    # Initialize parameters
    params = torch.tensor([1.0, 0.0], requires_grad=True, dtype=torch.float32)
    
    # Create SGD optimizer
    learning_rate = 0.012
    optimizer = optim.SGD([params], lr=learning_rate)
    
    print(f"\nUsing SGD optimizer with learning_rate={learning_rate}")
    print(f"Initial parameters: w={params[0].item():.4f}, b={params[1].item():.4f}")
    
    # Train model
    trained_params = training_loop_with_optimizer(
        n_epoch=500,
        params=params,
        x=x,
        y_target=y_target,
        optimizer=optimizer
    )
    
    print(f"\nTrained parameters: w={trained_params[0].item():.4f}, b={trained_params[1].item():.4f}")
    
    # Final loss
    with torch.no_grad():
        w, b = trained_params
        y_predict_final = linear_model(x, w, b)
        final_loss = loss_fn(y_predict_final, y_target)
        print(f"Final loss: {final_loss.item():.4f}")
    
    return trained_params


def demonstrate_adam_optimizer():
    """
    Demonstrate training with Adam optimizer.
    """
    print("\n" + "=" * 60)
    print("Adam Optimizer Demonstration")
    print("=" * 60)
    
    # Load data
    temp_celcius, temp_measurement = load_temperature_data()
    x = 0.1 * torch.tensor(temp_measurement.values, dtype=torch.float32)
    y_target = torch.tensor(temp_celcius.values, dtype=torch.float32)
    
    # Initialize parameters
    params = torch.tensor([1.0, 0.0], requires_grad=True, dtype=torch.float32)
    
    # Create Adam optimizer
    learning_rate = 0.012
    optimizer = optim.Adam([params], lr=learning_rate)
    
    print(f"\nUsing Adam optimizer with learning_rate={learning_rate}")
    print(f"Initial parameters: w={params[0].item():.4f}, b={params[1].item():.4f}")
    
    # Train model
    trained_params = training_loop_with_optimizer(
        n_epoch=500,
        params=params,
        x=x,
        y_target=y_target,
        optimizer=optimizer
    )
    
    print(f"\nTrained parameters: w={trained_params[0].item():.4f}, b={trained_params[1].item():.4f}")
    
    # Final loss
    with torch.no_grad():
        w, b = trained_params
        y_predict_final = linear_model(x, w, b)
        final_loss = loss_fn(y_predict_final, y_target)
        print(f"Final loss: {final_loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("Optimizer Comparison:")
    print("SGD: Simple gradient descent, w = w - lr * grad")
    print("Adam: Adaptive learning rate, uses momentum and")
    print("      adapts learning rate per parameter")
    print("=" * 60)
    
    return trained_params


if __name__ == "__main__":
    demonstrate_sgd_optimizer()
    demonstrate_adam_optimizer()

