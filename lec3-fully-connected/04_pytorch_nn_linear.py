import torch
import torch.nn as nn


def demonstrate_single_neuron_pytorch():
    """
    Demonstrates a single neuron using PyTorch's nn.Linear (equivalent to 01_neuron.py).
    A single neuron is just a Linear layer with 1 output neuron.
    """
    # Define input: a single sample with multiple features
    # Shape: (n_features,)
    n_features = 3
    input_vector = torch.tensor([1.5, 2.0, 0.8], dtype=torch.float32)  # Shape: (3,)

    # Create a Linear layer: nn.Linear(in_features, out_features)
    # This creates a single neuron with n_features inputs and 1 output
    # Internally, it has:
    #   - weights: shape (1, n_features) - but stored as (n_features, 1) and transposed
    #   - bias: shape (1,)
    linear_layer = nn.Linear(in_features=n_features, out_features=1)

    # PyTorch Linear layer expects input with at least 1 batch dimension
    # Add batch dimension: (3,) -> (1, 3)
    input_batch = input_vector.unsqueeze(0)  # Shape: (1, 3)

    # Forward pass through the linear layer
    # Linear layer performs: output = input @ weight.T + bias
    # Input shape: (1, 3)
    # Weight shape: (1, 3) - stored internally, transposed during computation
    # Output shape: (1, 1)
    output = linear_layer(input_batch)  # Shape: (1, 1)

    # Remove batch dimension to get scalar output
    output_scalar = output.squeeze()  # Shape: ()

    # Apply activation function (e.g., ReLU)
    relu = nn.ReLU()
    activated_output = relu(output_scalar)  # Shape: ()

    # Note: PyTorch's Linear layer handles all the matrix multiplication
    # and bias addition automatically, making it much simpler than manual numpy code

    return activated_output


def demonstrate_fully_connected_layer_pytorch():
    """
    Demonstrates a fully connected layer using PyTorch's nn.Linear (equivalent to 02_fc_layer.py).
    A fully connected layer has multiple output neurons.
    """
    # Define input: a single sample with multiple features
    # Shape: (n_features,)
    n_features = 3
    input_vector = torch.tensor([1.5, 2.0, 0.8], dtype=torch.float32)  # Shape: (3,)

    # Create a Linear layer with multiple outputs
    # nn.Linear(in_features, out_features)
    # This creates a layer with n_features inputs and n_outputs outputs
    n_outputs = 4
    linear_layer = nn.Linear(in_features=n_features, out_features=n_outputs)

    # The Linear layer internally stores:
    #   - weight: shape (n_outputs, n_features) - stored as (out_features, in_features)
    #   - bias: shape (n_outputs,)

    # Add batch dimension: (3,) -> (1, 3)
    input_batch = input_vector.unsqueeze(0)  # Shape: (1, 3)

    # Forward pass: output = input @ weight.T + bias
    # Input shape: (1, 3)
    # Weight shape: (4, 3) - transposed during computation
    # Output shape: (1, 4)
    output = linear_layer(input_batch)  # Shape: (1, 4)

    # Remove batch dimension: (1, 4) -> (4,)
    output_vector = output.squeeze(0)  # Shape: (4,)

    # Apply activation function element-wise
    relu = nn.ReLU()
    activated_output = relu(output_vector)  # Shape: (4,)

    # The key difference from single neuron:
    # - Single neuron: 1 output (scalar)
    # - Fully connected layer: multiple outputs (vector)
    # PyTorch handles this automatically - just change out_features parameter

    return activated_output


def demonstrate_batch_forward_pytorch():
    """
    Demonstrates batch forward pass using PyTorch's nn.Linear (equivalent to 03_batch_forward.py).
    Batch processing is handled automatically by PyTorch - just pass a batch of inputs.
    """
    # Define input: a batch of multiple samples, each with multiple features
    # Shape: (batch_size, n_features)
    batch_size = 4
    n_features = 3
    input_batch = torch.tensor(
        [
            [1.5, 2.0, 0.8],  # Sample 1
            [0.9, 1.2, 1.5],  # Sample 2
            [2.1, 0.5, 1.0],  # Sample 3
            [1.0, 1.8, 0.6],  # Sample 4
        ],
        dtype=torch.float32,
    )  # Shape: (4, 3)

    # Create a Linear layer
    n_outputs = 4
    linear_layer = nn.Linear(in_features=n_features, out_features=n_outputs)

    # Forward pass with batch input
    # Input shape: (4, 3)
    # Weight shape: (4, 3) - transposed during computation
    # Output shape: (4, 4)
    # PyTorch automatically handles batch processing:
    #   - Each row of input_batch is processed independently
    #   - Matrix multiplication is done efficiently for the entire batch
    output_batch = linear_layer(input_batch)  # Shape: (4, 4)

    # Apply activation function element-wise
    # ReLU is applied independently to each element in the batch
    relu = nn.ReLU()
    activated_output = relu(output_batch)  # Shape: (4, 4)

    # Output interpretation:
    # - activated_output[i, j] = activation of output neuron j for input sample i
    # - Row i: all output neuron activations for sample i
    # - Column j: activation of neuron j across all samples in the batch

    # PyTorch's batch processing is efficient because:
    # 1. Single matrix multiplication processes all samples
    # 2. Optimized for GPU acceleration
    # 3. Automatic broadcasting of bias to all samples

    return activated_output


def demonstrate_layer_parameters():
    """
    Demonstrates how to access and inspect the parameters of a Linear layer.
    """
    n_features = 3
    n_outputs = 4

    # Create a Linear layer
    linear_layer = nn.Linear(in_features=n_features, out_features=n_outputs)

    # Access weight matrix
    # Shape: (n_outputs, n_features) = (4, 3)
    # Note: PyTorch stores weights as (out_features, in_features)
    # This is transposed during forward pass: input @ weight.T
    weight = linear_layer.weight  # Shape: (4, 3)

    # Access bias vector
    # Shape: (n_outputs,) = (4,)
    bias = linear_layer.bias  # Shape: (4,)

    # By default, weights are initialized randomly (Kaiming/He initialization)
    # and bias is initialized to zeros
    # You can access and modify these parameters if needed

    # Example: manually set weights and bias (for demonstration)
    # Original weights from numpy examples were (in_features, out_features) = (3, 4)
    # PyTorch stores weights as (out_features, in_features) = (4, 3)
    # So we need to transpose: (3, 4).T = (4, 3)
    with torch.no_grad():  # Disable gradient tracking when setting values
        linear_layer.weight.data = torch.tensor(
            [
                [0.5, 0.1, -0.1],  # weights for output 0 from all inputs
                [-0.3, 0.4, 0.3],  # weights for output 1 from all inputs
                [0.7, -0.2, 0.5],  # weights for output 2 from all inputs
                [0.2, 0.6, -0.4],  # weights for output 3 from all inputs
            ],
            dtype=torch.float32,
        )  # Shape: (4, 3) - matches PyTorch's (out_features, in_features) format

        linear_layer.bias.data = torch.tensor(
            [0.1, -0.2, 0.3, 0.0], dtype=torch.float32
        )

    return linear_layer


def demonstrate_sequential_layers():
    """
    Demonstrates stacking multiple Linear layers to create a multi-layer network.
    """
    n_features = 3
    hidden_size = 8
    n_outputs = 4

    # Create a sequential model with multiple layers
    # This is equivalent to: input -> hidden_layer -> output_layer
    model = nn.Sequential(
        nn.Linear(in_features=n_features, out_features=hidden_size),  # First layer
        nn.ReLU(),  # Activation function
        nn.Linear(in_features=hidden_size, out_features=n_outputs),  # Second layer
        nn.ReLU(),  # Activation function
    )

    # Create batch input
    batch_size = 4
    input_batch = torch.tensor(
        [
            [1.5, 2.0, 0.8],
            [0.9, 1.2, 1.5],
            [2.1, 0.5, 1.0],
            [1.0, 1.8, 0.6],
        ],
        dtype=torch.float32,
    )  # Shape: (4, 3)

    # Forward pass through the entire network
    # Input: (4, 3)
    # After first Linear: (4, 8)
    # After ReLU: (4, 8)
    # After second Linear: (4, 4)
    # After ReLU: (4, 4)
    output = model(input_batch)  # Shape: (4, 4)

    # PyTorch's Sequential makes it easy to stack layers
    # Each layer's output becomes the next layer's input automatically

    return output


if __name__ == "__main__":
    # Demonstrate single neuron (equivalent to 01_neuron.py)
    single_output = demonstrate_single_neuron_pytorch()

    # Demonstrate fully connected layer (equivalent to 02_fc_layer.py)
    fc_output = demonstrate_fully_connected_layer_pytorch()

    # Demonstrate batch forward pass (equivalent to 03_batch_forward.py)
    batch_output = demonstrate_batch_forward_pytorch()

    # Demonstrate layer parameters
    layer = demonstrate_layer_parameters()

    # Demonstrate sequential layers
    sequential_output = demonstrate_sequential_layers()
