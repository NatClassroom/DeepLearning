# Lecture 3: The Mechanics of Learning

This lecture covers gradient-based optimization and backpropagation in PyTorch.

## Topics

1. Model and Loss function in PyTorch
2. Computing Gradient and training loop
3. PyTorch's autograd and optimizer

## Files

- `01_model_and_loss.py`: Loading data, defining a linear model, and computing loss
- `02_gradient_descent.py`: Manual gradient computation using finite differences
- `03_training_loop.py`: Training loop with manually computed gradients using chain rule
- `04_normalization.py`: Importance of input normalization for stable training
- `05_autograd.py`: Using PyTorch's automatic differentiation
- `06_optimizer.py`: Using PyTorch optimizers (SGD and Adam)

## Setup

You need to have the temperature data file `temp_data.csv` in a `data/` directory.

The data file should have two rows:
- Row 0: Temperature in Celsius (target values)
- Row 1: Thermometer measurements (input values)

## Running the Code

Each file can be run independently:

```bash
python 01_model_and_loss.py
python 02_gradient_descent.py
python 03_training_loop.py
python 04_normalization.py
python 05_autograd.py
python 06_optimizer.py
```

