import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def sigmoid_numpy(x):
    """
    Sigmoid activation function using numpy.
    Maps any real number to a value between 0 and 1.
    
    Formula: σ(x) = 1 / (1 + exp(-x))
    
    Properties:
    - Output range: (0, 1)
    - S-shaped curve (sigmoid curve)
    - σ(0) = 0.5
    - As x -> +∞, σ(x) -> 1
    - As x -> -∞, σ(x) -> 0
    """
    return 1 / (1 + np.exp(-x))


def demonstrate_sigmoid_basic():
    """
    Demonstrates basic sigmoid function behavior.
    Shows how different input values map to probabilities.
    """
    print("\n" + "=" * 60)
    print("Basic Sigmoid Function")
    print("=" * 60)
    
    # Test different input values
    test_values = np.array([-5, -2, -1, 0, 1, 2, 5])
    sigmoid_outputs = sigmoid_numpy(test_values)
    
    print("\nInput (logit) -> Output (probability):")
    print("-" * 40)
    for x, prob in zip(test_values, sigmoid_outputs):
        print(f"  {x:6.1f} -> {prob:.4f} ({prob*100:.1f}%)")
    
    print("\nKey observations:")
    print("  - Negative values -> probabilities < 0.5")
    print("  - Zero -> probability = 0.5 (50%)")
    print("  - Positive values -> probabilities > 0.5")
    print("  - Large positive values -> probabilities close to 1.0")
    print("  - Large negative values -> probabilities close to 0.0")


def demonstrate_binary_classification():
    """
    Demonstrates how sigmoid is used for binary classification.
    Shows the relationship between linear output (logit) and probability.
    """
    print("\n" + "=" * 60)
    print("Binary Classification with Sigmoid")
    print("=" * 60)
    
    # Example: Predicting if an email is spam (1) or not spam (0)
    # Features: [word_count, exclamation_count, sender_reputation]
    n_features = 3
    input_vector = np.array([150, 5, 0.3])  # Example email features
    
    # Learned weights from training
    # Higher weights mean the feature is more important for spam detection
    weights = np.array([0.01, 0.5, -2.0])  # exclamation_count is strong spam indicator
    bias = -2.0  # Negative bias means default is "not spam"
    
    # Step 1: Compute linear combination (logit)
    # This is the raw output before sigmoid
    logit = np.dot(input_vector, weights) + bias
    
    # Step 2: Apply sigmoid to get probability
    probability = sigmoid_numpy(logit)
    
    print(f"\nInput features: {input_vector}")
    print(f"  - Word count: {input_vector[0]}")
    print(f"  - Exclamation count: {input_vector[1]}")
    print(f"  - Sender reputation: {input_vector[2]}")
    
    print(f"\nWeights: {weights}")
    print(f"Bias: {bias}")
    
    print(f"\nStep 1 - Compute logit (linear output):")
    print(f"  logit = input @ weights + bias")
    print(f"  logit = {logit:.4f}")
    
    print(f"\nStep 2 - Apply sigmoid to get probability:")
    print(f"  probability = sigmoid(logit)")
    print(f"  probability = {probability:.4f} ({probability*100:.2f}%)")
    
    print(f"\nInterpretation:")
    if probability > 0.5:
        print(f"  Prediction: SPAM (probability = {probability*100:.2f}% > 50%)")
    else:
        print(f"  Prediction: NOT SPAM (probability = {probability*100:.2f}% < 50%)")
    
    return logit, probability


def demonstrate_multiple_samples():
    """
    Demonstrates binary classification for multiple samples.
    Shows how sigmoid converts a batch of logits to probabilities.
    """
    print("\n" + "=" * 60)
    print("Binary Classification for Multiple Samples")
    print("=" * 60)
    
    # Batch of input samples
    # Each row is a sample with 3 features
    input_batch = np.array([
        [100, 1, 0.8],   # Sample 1: few words, few exclamations, good sender
        [200, 10, 0.1],  # Sample 2: many words, many exclamations, bad sender
        [50, 0, 0.9],    # Sample 3: very few words, no exclamations, excellent sender
        [300, 15, 0.2],  # Sample 4: very many words, many exclamations, bad sender
    ])  # Shape: (4, 3)
    
    # Same weights and bias as before
    weights = np.array([0.01, 0.5, -2.0])
    bias = -2.0
    
    # Compute logits for all samples
    logits = input_batch @ weights + bias  # Shape: (4,)
    
    # Apply sigmoid to get probabilities
    probabilities = sigmoid_numpy(logits)  # Shape: (4,)
    
    print(f"\nInput batch shape: {input_batch.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    print(f"\nResults for each sample:")
    print("-" * 60)
    for i, (logit, prob) in enumerate(zip(logits, probabilities)):
        prediction = "SPAM" if prob > 0.5 else "NOT SPAM"
        print(f"Sample {i+1}:")
        print(f"  Features: {input_batch[i]}")
        print(f"  Logit: {logit:.4f}")
        print(f"  Probability: {prob:.4f} ({prob*100:.2f}%)")
        print(f"  Prediction: {prediction}")
        print()
    
    return logits, probabilities


def demonstrate_sigmoid_pytorch():
    """
    Demonstrates sigmoid using PyTorch.
    Shows how to use nn.Sigmoid() for binary classification.
    """
    print("\n" + "=" * 60)
    print("Sigmoid with PyTorch")
    print("=" * 60)
    
    # Create a simple binary classifier
    n_features = 3
    model = nn.Sequential(
        nn.Linear(in_features=n_features, out_features=1),  # Single output neuron
        nn.Sigmoid()  # Apply sigmoid to get probability
    )
    
    # Example input (single sample)
    input_vector = torch.tensor([150.0, 5.0, 0.3], dtype=torch.float32)
    input_batch = input_vector.unsqueeze(0)  # Add batch dimension: (3,) -> (1, 3)
    
    # Forward pass
    output = model(input_batch)  # Shape: (1, 1)
    probability = output.item()  # Extract scalar value
    
    print(f"\nInput: {input_vector}")
    print(f"Model architecture:")
    print(f"  Linear(3 -> 1): computes logit")
    print(f"  Sigmoid(): converts logit to probability")
    print(f"\nOutput probability: {probability:.4f} ({probability*100:.2f}%)")
    
    # Demonstrate with batch
    input_batch = torch.tensor([
        [100.0, 1.0, 0.8],
        [200.0, 10.0, 0.1],
        [50.0, 0.0, 0.9],
        [300.0, 15.0, 0.2],
    ], dtype=torch.float32)  # Shape: (4, 3)
    
    output_batch = model(input_batch)  # Shape: (4, 1)
    probabilities = output_batch.squeeze(1)  # Shape: (4,)
    
    print(f"\nBatch input shape: {input_batch.shape}")
    print(f"Batch output shape: {output_batch.shape}")
    print(f"\nProbabilities for batch:")
    for i, prob in enumerate(probabilities):
        print(f"  Sample {i+1}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
    
    return model, probabilities


def demonstrate_sigmoid_curve():
    """
    Visualizes the sigmoid curve to show how it maps logits to probabilities.
    """
    print("\n" + "=" * 60)
    print("Sigmoid Curve Visualization")
    print("=" * 60)
    
    # Generate a range of logit values
    logits = np.linspace(-10, 10, 1000)
    probabilities = sigmoid_numpy(logits)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(logits, probabilities, 'b-', linewidth=2, label='Sigmoid')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision threshold (0.5)')
    plt.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='Logit = 0')
    
    # Mark some key points
    key_logits = [-5, -2, 0, 2, 5]
    key_probs = sigmoid_numpy(np.array(key_logits))
    plt.scatter(key_logits, key_probs, color='red', s=100, zorder=5)
    
    for x, y in zip(key_logits, key_probs):
        plt.annotate(f'({x}, {y:.2f})', (x, y), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Logit (linear output)', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Sigmoid Function: Mapping Logits to Probabilities', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-10, 10)
    plt.ylim(0, 1)
    
    # Save the plot
    plt.savefig('sigmoid_curve.png', dpi=150, bbox_inches='tight')
    print("\nSigmoid curve saved to 'sigmoid_curve.png'")
    print("\nKey observations from the curve:")
    print("  - S-shaped curve (sigmoid shape)")
    print("  - Symmetric around (0, 0.5)")
    print("  - Steepest slope at logit = 0")
    print("  - Saturates at 0 and 1 for extreme logit values")
    
    return logits, probabilities


def demonstrate_decision_boundary():
    """
    Demonstrates how the decision boundary works in binary classification.
    Shows that probability > 0.5 means class 1, probability < 0.5 means class 0.
    """
    print("\n" + "=" * 60)
    print("Decision Boundary")
    print("=" * 60)
    
    # Different logit values
    logits = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    probabilities = sigmoid_numpy(logits)
    
    print("\nLogit -> Probability -> Class Prediction:")
    print("-" * 50)
    for logit, prob in zip(logits, probabilities):
        predicted_class = 1 if prob > 0.5 else 0
        print(f"Logit: {logit:6.2f} -> Prob: {prob:.4f} -> Class: {predicted_class}")
    
    print("\nDecision rule:")
    print("  - If probability > 0.5: predict class 1 (positive)")
    print("  - If probability < 0.5: predict class 0 (negative)")
    print("  - If probability = 0.5: exactly at decision boundary")
    print("\nNote: The decision threshold can be adjusted (e.g., 0.3 or 0.7)")
    print("      depending on the application (precision vs recall trade-off)")


if __name__ == "__main__":
    # Demonstrate basic sigmoid function
    demonstrate_sigmoid_basic()
    
    # Demonstrate binary classification
    logit, prob = demonstrate_binary_classification()
    
    # Demonstrate multiple samples
    logits, probs = demonstrate_multiple_samples()
    
    # Demonstrate PyTorch implementation
    model, pytorch_probs = demonstrate_sigmoid_pytorch()
    
    # Visualize sigmoid curve
    try:
        logits_curve, probs_curve = demonstrate_sigmoid_curve()
    except Exception as e:
        print(f"\nNote: Could not generate plot: {e}")
        print("      (matplotlib may not be installed or display not available)")
    
    # Demonstrate decision boundary
    demonstrate_decision_boundary()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Sigmoid maps any real number (logit) to a probability (0-1)")
    print("  2. For binary classification, sigmoid converts linear output to probability")
    print("  3. Probability > 0.5 typically means class 1, < 0.5 means class 0")
    print("  4. Sigmoid is smooth and differentiable, making it good for gradient-based learning")
    print("  5. PyTorch provides nn.Sigmoid() for easy use in neural networks")

