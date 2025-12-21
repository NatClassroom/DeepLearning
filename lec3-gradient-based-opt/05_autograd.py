"""
Topic 5: PyTorch Autograd
=========================

This module demonstrates:
1. Using PyTorch's automatic differentiation (autograd)
2. Computing gradients automatically with requires_grad=True
3. Simplifying the training loop with autograd
"""

import torch
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


def training_loop_autograd(n_epoch, learning_rate, params, x, y_target):
    """
    Training loop using PyTorch's autograd for automatic gradient computation.
    
    Args:
        n_epoch: Number of training epochs
        learning_rate: Learning rate for gradient descent
        params: Initial parameters [w, b] (must have requires_grad=True)
        x: Input tensor
        y_target: Target tensor
        
    Returns:
        Trained parameters
    """
    for epoch in range(n_epoch):
        # Zero gradients from previous iteration
        if params.grad is not None:
            params.grad.zero_()
        
        w, b = params
        
        # Forward pass: compute predictions and loss
        y_predict = linear_model(x, w, b)
        loss = loss_fn(y_predict, y_target)
        
        # Backward pass: compute gradients automatically
        loss.backward()
        
        # Update parameters (disable gradient tracking for update)
        with torch.no_grad():
            params -= learning_rate * params.grad
        
        # Print progress
        if epoch % 100 == 0 or epoch == n_epoch - 1:
            print(f'epoch: {epoch}, loss: {float(loss):.4f}')
    
    return params


def demonstrate_autograd():
    """
    Demonstrate using PyTorch's autograd for automatic gradient computation.
    """
    print("=" * 60)
    print("PyTorch Autograd Demonstration")
    print("=" * 60)
    
    # Load data
    temp_celcius, temp_measurement = load_temperature_data()
    x = 0.1 * torch.tensor(temp_measurement.values, dtype=torch.float32)
    y_target = torch.tensor(temp_celcius.values, dtype=torch.float32)
    
    # Initialize parameters with requires_grad=True
    # This tells PyTorch to track operations for gradient computation
    params = torch.tensor([1.0, 0.0], requires_grad=True, dtype=torch.float32)
    
    print(f"\nInitial parameters: w={params[0].item():.4f}, b={params[1].item():.4f}")
    print("Gradient before backward():", params.grad)
    
    # Forward pass
    w, b = params
    y_predict = linear_model(x, w, b)
    loss = loss_fn(y_predict, y_target)
    
    print(f"Initial loss: {loss.item():.4f}")
    
    # Backward pass - this computes gradients automatically
    loss.backward()
    
    print(f"\nGradient after backward(): {params.grad}")
    print("PyTorch automatically computed:")
    print(f"  dL/dw = {params.grad[0].item():.4f}")
    print(f"  dL/db = {params.grad[1].item():.4f}")
    
    # Train model using autograd
    print("\nTraining model with autograd...")
    params = torch.tensor([1.0, 0.0], requires_grad=True, dtype=torch.float32)
    
    trained_params = training_loop_autograd(
        n_epoch=500,
        learning_rate=0.012,
        params=params,
        x=x,
        y_target=y_target
    )
    
    print(f"\nTrained parameters: w={trained_params[0].item():.4f}, b={trained_params[1].item():.4f}")
    
    # Final loss
    with torch.no_grad():
        w, b = trained_params
        y_predict_final = linear_model(x, w, b)
        final_loss = loss_fn(y_predict_final, y_target)
        print(f"Final loss: {final_loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("Key Benefits of Autograd:")
    print("1. No need to manually compute derivatives")
    print("2. Works with any differentiable operation")
    print("3. Handles complex computation graphs automatically")
    print("=" * 60)
    
    return trained_params


if __name__ == "__main__":
    demonstrate_autograd()

