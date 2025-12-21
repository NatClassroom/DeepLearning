"""
Topic 3: Training Loop with Manual Gradients
============================================

This module demonstrates:
1. Computing gradients using chain rule
2. Implementing a training loop
3. Updating parameters iteratively to minimize loss
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
average_loss = model_and_loss.average_loss
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
    square_loss = (y - y_target) ** 2
    return square_loss.mean()


def dloss_dy(y, y_target):
    """
    Compute derivative of loss with respect to predictions.

    dL/dy = 2 * (y - y_target) / n

    Args:
        y: Predicted values
        y_target: Target values

    Returns:
        Gradient of loss w.r.t. predictions
    """
    n = y.size(0)
    dsquare_loss_dy = 2 * (y - y_target) / n  # division comes from derivative of mean()
    return dsquare_loss_dy


def dy_dw(x, w, b):
    """
    Compute derivative of model output with respect to weight.

    Since y = w*x + b, we have dy/dw = x

    Args:
        x: Input tensor
        w: Weight parameter
        b: Bias parameter

    Returns:
        Gradient of output w.r.t. weight
    """
    return x


def dy_db(x, w, b):
    """
    Compute derivative of model output with respect to bias.

    Since y = w*x + b, we have dy/db = 1

    Args:
        x: Input tensor
        w: Weight parameter
        b: Bias parameter

    Returns:
        Gradient of output w.r.t. bias
    """
    return torch.ones_like(x)


def grad_fn(x, y_predict, y_target, w, b):
    """
    Compute gradients using chain rule.

    Chain rule: dL/dw = dL/dy * dy/dw
                dL/db = dL/dy * dy/db

    Args:
        x: Input tensor
        y_predict: Predicted values
        y_target: Target values
        w: Weight parameter
        b: Bias parameter

    Returns:
        Stacked gradients [dL/dw, dL/db]
    """
    dl_dy = dloss_dy(y_predict, y_target)
    dl_dw = dl_dy * dy_dw(x, w, b)
    dl_db = dl_dy * dy_db(x, w, b)
    return torch.stack([dl_dw.sum(), dl_db.sum()])


def training_loop(n_epoch, learning_rate, params, x, y_target):
    """
    Training loop that updates parameters using manually computed gradients.

    Args:
        n_epoch: Number of training epochs
        learning_rate: Learning rate for gradient descent
        params: Initial parameters [w, b]
        x: Input tensor
        y_target: Target tensor

    Returns:
        Trained parameters
    """
    for epoch in range(n_epoch):
        w, b = params

        # Forward pass: compute predictions and loss
        y_predict = linear_model(x, w, b)
        loss = loss_fn(y_predict, y_target)

        # Backward pass: compute gradients
        grad = grad_fn(x, y_predict, y_target, w, b)

        # Update parameters
        params = params - learning_rate * grad

        # Print progress
        if epoch % 10 == 0 or epoch == n_epoch - 1:
            print(f"epoch: {epoch}, loss: {float(loss):.4f}")

    return params


def demonstrate_training_loop():
    """
    Demonstrate training a model using manual gradient computation.
    """
    print("=" * 60)
    print("Training Loop with Manual Gradients")
    print("=" * 60)

    # Load data
    temp_celcius, temp_measurement = load_temperature_data()
    x = torch.tensor(temp_measurement.values, dtype=torch.float32)
    y_target = torch.tensor(temp_celcius.values, dtype=torch.float32)

    # Initialize parameters
    params = torch.tensor([1.0, 0.0], dtype=torch.float32)

    print(f"\nInitial parameters: w={params[0].item():.4f}, b={params[1].item():.4f}")

    # Train model
    print("\nTraining model...")
    trained_params = training_loop(
        n_epoch=100, learning_rate=0.0001, params=params, x=x, y_target=y_target
    )

    print(
        f"\nTrained parameters: w={trained_params[0].item():.4f}, b={trained_params[1].item():.4f}"
    )

    # Final predictions
    w_final, b_final = trained_params
    y_predict_final = linear_model(x, w_final, b_final)
    final_loss = loss_fn(y_predict_final, y_target)
    print(f"Final loss: {final_loss.item():.4f}")

    return trained_params


if __name__ == "__main__":
    demonstrate_training_loop()
