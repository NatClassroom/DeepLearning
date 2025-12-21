import numpy as np


def demonstrate_batch_forward_pass():
    """
    Demonstrates forward pass through a fully connected layer with batch processing.
    Batch processing allows us to process multiple samples simultaneously, which is
    much more efficient than processing samples one at a time.
    """
    # Define input: a batch of multiple samples, each with multiple features
    # Shape: (batch_size, n_features)
    # In practice, batch_size could be 32, 64, 128, etc.
    batch_size = 4
    n_features = 3
    input_batch = np.array(
        [
            [1.5, 2.0, 0.8],  # Sample 1
            [0.9, 1.2, 1.5],  # Sample 2
            [2.1, 0.5, 1.0],  # Sample 3
            [1.0, 1.8, 0.6],  # Sample 4
        ]
    )  # Shape: (4, 3)

    # Define weights: a matrix where each column represents weights for one output neuron
    # Shape: (n_features, n_outputs)
    # These weights are shared across all samples in the batch
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
    # Bias is added to each sample's output (broadcasting)
    bias = np.array([0.1, -0.2, 0.3, 0.0])  # Shape: (4,)

    # Forward pass: compute weighted sum using matrix multiplication
    # input_batch shape: (4, 3)
    # weights shape: (3, 4)
    # Result: (4, 3) @ (3, 4) = (4, 4)
    # Matrix multiplication works as follows:
    #   - Each row of input_batch (one sample) is multiplied with the weights matrix
    #   - Result: each row in output corresponds to one input sample
    #   - Each column in output corresponds to one output neuron
    weighted_sum = input_batch @ weights + bias  # Shape: (4, 4)

    # Broadcasting explanation:
    # bias has shape (4,), which gets broadcasted to (4, 4) when added
    # This means the same bias vector is added to each sample's output
    # Equivalent to: weighted_sum = input_batch @ weights + bias[np.newaxis, :]

    # Apply activation function (e.g., ReLU) element-wise
    # The activation is applied independently to each element in the batch
    def relu(x):
        return np.maximum(0, x)

    output_batch = relu(weighted_sum)  # Shape: (4, 4)

    # Output interpretation:
    # - output_batch[i, j] = activation of output neuron j for input sample i
    # - Row i: all output neuron activations for sample i
    # - Column j: activation of neuron j across all samples in the batch

    return output_batch


def compare_single_vs_batch():
    """
    Demonstrates the difference between processing samples one at a time
    versus processing them as a batch.
    """
    # Same weights and bias for both approaches
    n_features = 3
    n_outputs = 4
    weights = np.array(
        [
            [0.5, -0.3, 0.7, 0.2],
            [0.1, 0.4, -0.2, 0.6],
            [-0.1, 0.3, 0.5, -0.4],
        ]
    )  # Shape: (3, 4)
    bias = np.array([0.1, -0.2, 0.3, 0.0])  # Shape: (4,)

    # Three samples to process
    sample1 = np.array([1.5, 2.0, 0.8])  # Shape: (3,)
    sample2 = np.array([0.9, 1.2, 1.5])  # Shape: (3,)
    sample3 = np.array([2.1, 0.5, 1.0])  # Shape: (3,)

    # Approach 1: Process samples one at a time (inefficient)
    # This requires multiple separate matrix multiplications
    output1_single = sample1 @ weights + bias  # Shape: (4,)
    output2_single = sample2 @ weights + bias  # Shape: (4,)
    output3_single = sample3 @ weights + bias  # Shape: (4,)
    # Would need to stack results: np.stack([output1_single, output2_single, output3_single])

    # Approach 2: Process samples as a batch (efficient)
    # Single matrix multiplication processes all samples at once
    batch = np.array([sample1, sample2, sample3])  # Shape: (3, 3)
    output_batch = batch @ weights + bias  # Shape: (3, 4)

    # The batch approach is more efficient because:
    # 1. Single matrix multiplication instead of multiple
    # 2. Better utilization of parallel computation (GPUs, vectorized operations)
    # 3. More cache-friendly memory access patterns
    # 4. Easier to implement and reason about

    return output_batch


def demonstrate_broadcasting_in_batch():
    """
    Demonstrates how broadcasting works when adding bias to batch outputs.
    """
    batch_size = 3
    n_features = 3
    n_outputs = 4

    input_batch = np.array(
        [
            [1.5, 2.0, 0.8],
            [0.9, 1.2, 1.5],
            [2.1, 0.5, 1.0],
        ]
    )  # Shape: (3, 3)

    weights = np.array(
        [
            [0.5, -0.3, 0.7, 0.2],
            [0.1, 0.4, -0.2, 0.6],
            [-0.1, 0.3, 0.5, -0.4],
        ]
    )  # Shape: (3, 4)

    bias = np.array([0.1, -0.2, 0.3, 0.0])  # Shape: (4,)

    # Step 1: Matrix multiplication without bias
    # (3, 3) @ (3, 4) = (3, 4)
    weighted_sum_no_bias = input_batch @ weights  # Shape: (3, 4)

    # Step 2: Add bias using broadcasting
    # bias shape: (4,) gets broadcasted to (3, 4)
    # NumPy automatically adds bias to each row of weighted_sum_no_bias
    weighted_sum_with_bias = weighted_sum_no_bias + bias  # Shape: (3, 4)

    # Broadcasting is equivalent to:
    # bias_broadcasted = np.tile(bias, (batch_size, 1))  # Shape: (3, 4)
    # weighted_sum_with_bias = weighted_sum_no_bias + bias_broadcasted

    # This broadcasting behavior is why we can use a 1D bias vector
    # instead of a 2D matrix, making the code simpler and more memory efficient

    return weighted_sum_with_bias


if __name__ == "__main__":
    # Demonstrate batch forward pass
    output_batch = demonstrate_batch_forward_pass()

    # Compare single vs batch processing
    batch_output = compare_single_vs_batch()

    # Demonstrate broadcasting
    broadcasted_output = demonstrate_broadcasting_in_batch()
