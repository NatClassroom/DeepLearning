import numpy as np


def demonstrate_fully_connected_layer():
    """
    Demonstrates a fully connected layer computation using numpy and matrix multiplication.
    A fully connected layer performs: output = activation(input @ weights + bias)
    The key difference from a single neuron is that the output has multiple dimensions.
    """
    # Define input: a single sample with multiple features
    # Example: 3 features (e.g., height, weight, age)
    n_features = 3
    input_vector = np.array([1.5, 2.0, 0.8])  # Shape: (3,)

    # Define weights: a matrix where each column represents weights for one output neuron
    # Shape: (n_features, n_outputs)
    # This layer has 4 output neurons, each with 3 weights (one per input feature)
    n_outputs = 4
    weights = np.array(
        [
            [0.5, -0.3, 0.7, 0.2],  # weights for feature 1 to all 4 outputs
            [0.1, 0.4, -0.2, 0.6],  # weights for feature 2 to all 4 outputs
            [-0.1, 0.3, 0.5, -0.4],  # weights for feature 3 to all 4 outputs
        ]
    )  # Shape: (3, 4)

    # Define bias: a vector with one bias value per output neuron
    # Shape: (n_outputs,)
    bias = np.array([0.1, -0.2, 0.3, 0.0])  # Shape: (4,)

    # Compute weighted sum using matrix multiplication: input @ weights + bias
    # input_vector shape: (3,)
    # weights shape: (3, 4)
    # Result: (3,) @ (3, 4) = (4,)
    # This performs: for each output neuron, compute dot product of input with its weights
    weighted_sum = input_vector @ weights + bias  # Shape: (4,)

    # Alternative explicit calculation showing how matrix multiplication works:
    # For each output neuron j:
    #   output[j] = sum(input[i] * weights[i, j] for i in range(n_features)) + bias[j]
    # This is equivalent to:
    #   output[0] = input[0]*weights[0,0] + input[1]*weights[1,0] + input[2]*weights[2,0] + bias[0]
    #   output[1] = input[0]*weights[0,1] + input[1]*weights[1,1] + input[2]*weights[2,1] + bias[1]
    #   output[2] = input[0]*weights[0,2] + input[1]*weights[1,2] + input[2]*weights[2,2] + bias[2]
    #   output[3] = input[0]*weights[0,3] + input[1]*weights[1,3] + input[2]*weights[2,3] + bias[3]

    # Apply activation function (e.g., ReLU) element-wise to each output
    def relu(x):
        return np.maximum(0, x)

    output = relu(weighted_sum)  # Shape: (4,)

    # The key difference from 01_neuron.py:
    # - 01_neuron.py: output is a scalar (single value)
    # - 02_fc_layer.py: output is a vector (multiple values, one per output neuron)
    # This allows the layer to learn multiple different patterns simultaneously

    return output


def demonstrate_batch_processing():
    """
    Demonstrates how a fully connected layer processes multiple samples at once.
    This is more efficient than processing samples one at a time.
    """
    # Define input: a batch of samples, each with multiple features
    # Shape: (batch_size, n_features)
    batch_size = 2
    n_features = 3
    input_batch = np.array(
        [[1.5, 2.0, 0.8], [0.9, 1.2, 1.5]]  # Sample 1  # Sample 2
    )  # Shape: (2, 3)

    # Define weights: same as before
    # Shape: (n_features, n_outputs)
    n_outputs = 4
    weights = np.array(
        [[0.5, -0.3, 0.7, 0.2], [0.1, 0.4, -0.2, 0.6], [-0.1, 0.3, 0.5, -0.4]]
    )  # Shape: (3, 4)

    # Define bias: same as before
    # Shape: (n_outputs,)
    bias = np.array([0.1, -0.2, 0.3, 0.0])  # Shape: (4,)

    # Compute weighted sum using matrix multiplication: input_batch @ weights + bias
    # input_batch shape: (2, 3)
    # weights shape: (3, 4)
    # Result: (2, 3) @ (3, 4) = (2, 4)
    # This processes both samples simultaneously:
    #   - Row 0: output for sample 1
    #   - Row 1: output for sample 2
    weighted_sum = input_batch @ weights + bias  # Shape: (2, 4)

    # Apply activation function element-wise
    def relu(x):
        return np.maximum(0, x)

    output_batch = relu(weighted_sum)  # Shape: (2, 4)

    # The output shape is (batch_size, n_outputs)
    # Each row represents the output for one input sample
    # Each column represents one output neuron's activation across all samples

    return output_batch


if __name__ == "__main__":
    # Demonstrate single sample processing
    output = demonstrate_fully_connected_layer()

    # Demonstrate batch processing
    output_batch = demonstrate_batch_processing()
