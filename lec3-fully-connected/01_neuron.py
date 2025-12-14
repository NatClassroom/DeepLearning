import numpy as np


def demonstrate_single_neuron():
    """
    Demonstrates a single neuron computation using numpy.
    A neuron performs: output = activation(input @ weights + bias)
    """
    # Define input: a single sample with multiple features
    # Example: 3 features (e.g., height, weight, age)
    n_features = 3
    input_vector = np.array([1.5, 2.0, 0.8])  # Shape: (3,)

    # Define weights: one weight per input feature
    # Shape must match the number of input features
    weights = np.array([0.5, -0.3, 0.7])  # Shape: (3,)

    # Define bias: a single scalar value
    bias = 0.1

    # Compute weighted sum: input @ weights + bias
    # This is a dot product (element-wise multiplication + sum)
    weighted_sum = np.dot(input_vector, weights) + bias

    # Alternative: weighted_sum = input_vector @ weights + bias
    element_wise = input_vector * weights
    sum_products = np.sum(element_wise)

    # Apply activation function (e.g., ReLU)
    def relu(x):
        return np.maximum(0, x)

    output = relu(weighted_sum)

    return output


def demonstrate_matrix_multiplication():
    """
    Demonstrates the matrix multiplication aspect of a neuron.
    Shows how the dot product works step by step.
    """
    print("\n" + "=" * 60)
    print("Matrix Multiplication Details")
    print("=" * 60)

    input_vector = np.array([1.5, 2.0, 0.8])
    weights = np.array([0.5, -0.3, 0.7])

    print(f"\nInput vector (row vector): {input_vector}")
    print(f"Weight vector (column vector): {weights}")

    # Dot product: sum of element-wise products
    dot_product = np.dot(input_vector, weights)

    print(f"\nDot Product Calculation:")
    print(f"  {input_vector[0]} * {weights[0]} = {input_vector[0] * weights[0]}")
    print(f"  {input_vector[1]} * {weights[1]} = {input_vector[1] * weights[1]}")
    print(f"  {input_vector[2]} * {weights[2]} = {input_vector[2] * weights[2]}")
    print(f"  Sum: {dot_product}")

    print(f"\nUsing numpy.dot(): {dot_product}")
    print(f"Using @ operator: {input_vector @ weights}")


if __name__ == "__main__":
    demonstrate_single_neuron()
    demonstrate_matrix_multiplication()
