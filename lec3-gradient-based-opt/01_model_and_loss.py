"""
Topic 1: Model and Loss Function
=================================

This module demonstrates:
1. Loading and visualizing data
2. Defining a linear model
3. Computing loss function

We'll use a simple problem: mapping thermometer measurements to Celsius.
"""

import pandas as pd
import torch
import matplotlib.pyplot as plt


def load_temperature_data(data_path="data/temp_data.csv"):
    """
    Load temperature data from CSV file.

    Args:
        data_path: Path to the CSV file containing temperature data.
                   Default: 'data/temp_data.csv' (relative to current working directory)

    Returns:
        temp_celcius: Temperature in Celsius (target values)
        temp_measurement: Thermometer measurements (input values)
    """
    data = pd.read_csv(data_path, header=None)
    temp_celcius = data.iloc[0]
    temp_measurement = data.iloc[1]
    return temp_celcius, temp_measurement


def visualize_data(temp_measurement, temp_celcius):
    """
    Visualize the temperature data as a scatter plot.

    Args:
        temp_measurement: Thermometer measurements
        temp_celcius: Temperature in Celsius
    """
    fig = plt.figure()
    plt.xlabel("Measurement")
    plt.ylabel("Temperature (°Celsius)")
    plt.scatter(temp_measurement, temp_celcius)
    plt.show()


def linear_model(x, w, b):
    """
    Simple linear model: y = w * x + b

    Args:
        x: Input tensor
        w: Weight parameter
        b: Bias parameter

    Returns:
        Output tensor
    """
    return w * x + b


def average_loss(y_predict, y_target):
    """
    Compute mean squared error loss.

    Args:
        y_predict: Predicted values
        y_target: Target values

    Returns:
        Mean squared error
    """
    square_loss = (y_predict - y_target) ** 2
    return square_loss.mean()


def demonstrate_model_and_loss():
    """
    Demonstrate loading data, creating a model, and computing loss.
    """
    print("=" * 60)
    print("Model and Loss Function Demonstration")
    print("=" * 60)

    # Load data
    temp_celcius, temp_measurement = load_temperature_data()
    print(f"\nLoaded {len(temp_celcius)} data points")

    # Visualize data
    print("\nVisualizing data...")
    visualize_data(temp_measurement, temp_celcius)

    # Convert to tensors
    y_target = torch.tensor(temp_celcius.values, dtype=torch.float32)
    x = torch.tensor(temp_measurement.values, dtype=torch.float32)

    # Define model parameters
    w = 1.0
    b = 0.0

    # Make predictions
    y_predict = linear_model(x, w, b)

    # Compute loss
    loss = average_loss(y_predict, y_target)
    print(f"\nInitial loss with w={w}, b={b}: {loss.item():.4f}")

    # Try different parameters
    w = 0.5
    b = 0.0
    y_predict = linear_model(x, w, b)
    loss = average_loss(y_predict, y_target)
    print(f"Loss with w={w}, b={b}: {loss.item():.4f}")

    # Visualize model predictions
    x_input = torch.arange(0, 100, 1, dtype=torch.float32)
    y_predict_plot = linear_model(x_input, w, b)

    fig = plt.figure()
    plt.xlabel("Measurement")
    plt.ylabel("Temperature (°Celsius)")
    plt.scatter(temp_measurement, temp_celcius, color="blue", label="Data")
    plt.plot(x_input.numpy(), y_predict_plot.numpy(), color="red", label="Model")
    plt.legend()
    plt.show()

    return x, y_target, w, b


if __name__ == "__main__":
    demonstrate_model_and_loss()
