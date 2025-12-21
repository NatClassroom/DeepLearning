"""
Topic 2: Gradient Descent
=========================

This module demonstrates:
1. Computing gradients manually using finite differences
2. Understanding how gradients guide parameter updates
3. The general update rule: theta = theta - learning_rate * gradient
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


def compute_gradient_finite_difference(x, y_target, w, b, delta=0.1):
    """
    Compute gradient using finite difference method.

    This approximates the derivative by computing:
    df/dw ≈ (f(w+delta) - f(w-delta)) / (2*delta)

    Args:
        x: Input tensor
        y_target: Target tensor
        w: Current weight value
        b: Current bias value
        delta: Small change for finite difference

    Returns:
        gradient_w: Gradient with respect to weight
        gradient_b: Gradient with respect to bias
    """
    # Compute loss for w - delta
    y_predict_0 = linear_model(x, w - delta, b)
    loss_0 = average_loss(y_predict_0, y_target)

    # Compute loss for w + delta
    y_predict_1 = linear_model(x, w + delta, b)
    loss_1 = average_loss(y_predict_1, y_target)

    # Approximate gradient using finite difference
    gradient_w = (loss_1 - loss_0) / (2 * delta)

    # Compute gradient for bias
    y_predict_b0 = linear_model(x, w, b - delta)
    loss_b0 = average_loss(y_predict_b0, y_target)

    y_predict_b1 = linear_model(x, w, b + delta)
    loss_b1 = average_loss(y_predict_b1, y_target)

    gradient_b = (loss_b1 - loss_b0) / (2 * delta)

    return gradient_w, gradient_b


def demonstrate_gradient_descent():
    """
    Demonstrate gradient computation and parameter update.
    """
    print("=" * 60)
    print("Gradient Descent Demonstration")
    print("=" * 60)

    # Load data
    temp_celcius, temp_measurement = load_temperature_data()
    x = torch.tensor(temp_measurement.values, dtype=torch.float32)
    y_target = torch.tensor(temp_celcius.values, dtype=torch.float32)

    # Initial parameters
    w = 0.5
    b = 0.0

    print(f"\nInitial parameters: w={w}, b={b}")
    y_predict = linear_model(x, w, b)
    initial_loss = average_loss(y_predict, y_target)
    print(f"Initial loss: {initial_loss.item():.4f}")

    # Compute gradients
    gradient_w, gradient_b = compute_gradient_finite_difference(x, y_target, w, b)
    print(f"\nGradient w.r.t. w: {gradient_w.item():.4f}")
    print(f"Gradient w.r.t. b: {gradient_b.item():.4f}")

    # Positive gradient means loss increases when we increase w
    # So we should decrease w to reduce loss
    print("\nInterpretation:")
    print("Positive gradient w.r.t. w means loss increases when w increases")
    print("Therefore, we should decrease w to reduce loss")

    # Update parameters using gradient descent
    learning_rate = 1e-2
    print(f"\nUpdating parameters with learning_rate={learning_rate}")
    print("Update rule: theta = theta - learning_rate * gradient")

    w_new = w - learning_rate * gradient_w
    b_new = b - learning_rate * gradient_b

    print(f"\nUpdated parameters: w={w_new.item():.4f}, b={b_new.item():.4f}")

    # Check new loss
    y_predict_new = linear_model(x, w_new, b_new)
    new_loss = average_loss(y_predict_new, y_target)
    print(f"New loss: {new_loss.item():.4f}")
    print(f"Loss reduction: {initial_loss.item() - new_loss.item():.4f}")

    return w_new, b_new


if __name__ == "__main__":
    demonstrate_gradient_descent()
