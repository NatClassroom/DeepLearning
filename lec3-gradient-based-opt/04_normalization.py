"""
Topic 4: Input Normalization
============================

This module demonstrates:
1. Why normalization is important for gradient descent
2. How different parameter scales affect gradient magnitudes
3. Normalizing inputs to stabilize training
"""

import torch
import matplotlib.pyplot as plt
import importlib.util
import sys
import os

# Import module with number prefix using importlib
current_dir = os.path.dirname(os.path.abspath(__file__))

spec1 = importlib.util.spec_from_file_location(
    "model_and_loss", os.path.join(current_dir, "01_model_and_loss.py")
)
model_and_loss = importlib.util.module_from_spec(spec1)
sys.modules["model_and_loss"] = model_and_loss
spec1.loader.exec_module(model_and_loss)

spec2 = importlib.util.spec_from_file_location(
    "training_loop", os.path.join(current_dir, "03_training_loop.py")
)
training_loop_module = importlib.util.module_from_spec(spec2)
sys.modules["training_loop"] = training_loop_module
spec2.loader.exec_module(training_loop_module)

linear_model = model_and_loss.linear_model
load_temperature_data = model_and_loss.load_temperature_data
loss_fn = training_loop_module.loss_fn
grad_fn = training_loop_module.grad_fn
training_loop = training_loop_module.training_loop


def demonstrate_normalization():
    """
    Demonstrate the importance of input normalization for stable training.
    """
    print("=" * 60)
    print("Input Normalization Demonstration")
    print("=" * 60)

    # Load data
    temp_celcius, temp_measurement = load_temperature_data()
    x_original = torch.tensor(temp_measurement.values, dtype=torch.float32)
    y_target = torch.tensor(temp_celcius.values, dtype=torch.float32)

    # Normalize input (scale by 0.1)
    x_normalized = 0.1 * x_original

    print(f"\nOriginal input range: [{x_original.min():.1f}, {x_original.max():.1f}]")
    print(
        f"Normalized input range: [{x_normalized.min():.1f}, {x_normalized.max():.1f}]"
    )

    # Train with normalized input
    print("\nTraining with normalized input...")
    params = torch.tensor([1.0, 0.0], dtype=torch.float32)

    trained_params = training_loop(
        n_epoch=500,
        learning_rate=0.012,
        params=params,
        x=x_normalized,
        y_target=y_target,
    )

    print(
        f"\nTrained parameters: w={trained_params[0].item():.4f}, b={trained_params[1].item():.4f}"
    )

    # Visualize results
    w, b = trained_params
    x_input = torch.arange(0, 100, 1, dtype=torch.float32)
    x_input_normalized = 0.1 * x_input
    y_predict = linear_model(x_input_normalized, w, b)

    fig = plt.figure()
    plt.xlabel("Measurement")
    plt.ylabel("Temperature (°Celsius)")
    plt.scatter(temp_measurement, temp_celcius, color="blue", label="Data")
    plt.plot(x_input.numpy(), y_predict.numpy(), color="red", label="Trained Model")
    plt.legend()
    plt.title("Model trained with normalized input")
    plt.show()

    # Final loss
    y_predict_final = linear_model(x_normalized, w, b)
    final_loss = loss_fn(y_predict_final, y_target)
    print(f"Final loss: {final_loss.item():.4f}")

    print("\n" + "=" * 60)
    print("Key Insight:")
    print("When parameter scales differ significantly, gradients have")
    print("different magnitudes. Normalizing inputs helps balance")
    print("gradient magnitudes and makes training more stable.")
    print("=" * 60)

    return trained_params


if __name__ == "__main__":
    demonstrate_normalization()
