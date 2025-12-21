import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def negative_log_likelihood_numpy(probability, true_label):
    """
    Compute negative log likelihood for binary classification using numpy.
    
    Formula:
    - If true_label = 1: NLL = -log(probability)
    - If true_label = 0: NLL = -log(1 - probability)
    
    Or equivalently: NLL = -[y * log(p) + (1-y) * log(1-p)]
    
    Properties:
    - When prediction is correct and confident: NLL is small (close to 0)
    - When prediction is wrong: NLL is large (approaches infinity)
    - When prediction is uncertain (p ≈ 0.5): NLL is moderate (~0.69)
    
    Args:
        probability: Predicted probability of class 1 (from sigmoid), in range [0, 1]
        true_label: True class label, either 0 or 1
    
    Returns:
        Negative log likelihood value
    """
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-15
    probability = np.clip(probability, epsilon, 1 - epsilon)
    
    if true_label == 1:
        nll = -np.log(probability)
    else:  # true_label == 0
        nll = -np.log(1 - probability)
    
    return nll


def negative_log_likelihood_batch_numpy(probabilities, true_labels):
    """
    Compute negative log likelihood for a batch of samples.
    
    Formula: NLL = -[y * log(p) + (1-y) * log(1-p)]
    
    Args:
        probabilities: Array of predicted probabilities, shape (n_samples,)
        true_labels: Array of true labels, shape (n_samples,)
    
    Returns:
        Array of NLL values, shape (n_samples,)
    """
    epsilon = 1e-15
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
    
    # Vectorized computation: NLL = -[y * log(p) + (1-y) * log(1-p)]
    nll = -(true_labels * np.log(probabilities) + (1 - true_labels) * np.log(1 - probabilities))
    
    return nll


def demonstrate_nll_basic():
    """
    Demonstrates basic negative log likelihood computation.
    Shows how NLL behaves for different predictions and true labels.
    """
    print("\n" + "=" * 60)
    print("Basic Negative Log Likelihood")
    print("=" * 60)
    
    # Example scenarios
    scenarios = [
        (0.9, 1, "Correct, confident (high prob for class 1)"),
        (0.1, 0, "Correct, confident (low prob for class 1)"),
        (0.6, 1, "Correct, uncertain"),
        (0.4, 0, "Correct, uncertain"),
        (0.1, 1, "Wrong prediction (predicted 0, actual 1)"),
        (0.9, 0, "Wrong prediction (predicted 1, actual 0)"),
        (0.5, 1, "Completely uncertain (50/50)"),
        (0.5, 0, "Completely uncertain (50/50)"),
    ]
    
    print("\nPredicted Probability | True Label | NLL      | Interpretation")
    print("-" * 70)
    for prob, label, description in scenarios:
        nll = negative_log_likelihood_numpy(prob, label)
        print(f"  {prob:.2f}              |     {label}      | {nll:.4f}  | {description}")
    
    print("\nKey observations:")
    print("  - Correct + confident predictions → low NLL (close to 0)")
    print("  - Wrong predictions → high NLL (large penalty)")
    print("  - Uncertain predictions (p ≈ 0.5) → moderate NLL (~0.69)")
    print("  - NLL increases as confidence in wrong direction increases")


def demonstrate_nll_from_sigmoid():
    """
    Demonstrates computing NLL from sigmoid outputs.
    Shows the complete pipeline: logit → sigmoid → probability → NLL.
    """
    print("\n" + "=" * 60)
    print("NLL from Sigmoid Output")
    print("=" * 60)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Example: Binary classification problem
    # We have logits (raw outputs) from a model
    logits = np.array([2.0, -1.0, 0.5, -2.0, 3.0])
    true_labels = np.array([1, 0, 1, 0, 1])  # True class labels
    
    # Step 1: Apply sigmoid to get probabilities
    probabilities = sigmoid(logits)
    
    # Step 2: Compute NLL for each sample
    nll_values = negative_log_likelihood_batch_numpy(probabilities, true_labels)
    
    print("\nStep-by-step computation:")
    print("-" * 70)
    print("Logit    | Probability | True Label | NLL      | Prediction | Correct?")
    print("-" * 70)
    for i, (logit, prob, label, nll) in enumerate(zip(logits, probabilities, true_labels, nll_values)):
        prediction = 1 if prob > 0.5 else 0
        correct = "✓" if prediction == label else "✗"
        print(f"{logit:8.2f} | {prob:11.4f} |     {label}      | {nll:8.4f} |     {prediction}      |   {correct}")
    
    # Compute average NLL (this is what we minimize during training)
    average_nll = np.mean(nll_values)
    print(f"\nAverage NLL (loss): {average_nll:.4f}")
    print("  - This is the value we minimize during training")
    print("  - Lower average NLL means better predictions overall")


def demonstrate_nll_penalty():
    """
    Demonstrates how NLL penalizes wrong predictions more heavily.
    Visualizes the relationship between predicted probability and NLL.
    """
    print("\n" + "=" * 60)
    print("NLL Penalty Visualization")
    print("=" * 60)
    
    # Generate range of probabilities
    probabilities = np.linspace(0.01, 0.99, 1000)
    
    # Compute NLL for both classes
    nll_class_1 = -np.log(probabilities)  # When true label is 1
    nll_class_0 = -np.log(1 - probabilities)  # When true label is 0
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(probabilities, nll_class_1, 'b-', linewidth=2, label='NLL when true label = 1')
    plt.plot(probabilities, nll_class_0, 'r-', linewidth=2, label='NLL when true label = 0')
    
    # Mark key points
    key_probs = [0.1, 0.5, 0.9]
    for p in key_probs:
        nll_1 = -np.log(p)
        nll_0 = -np.log(1 - p)
        plt.scatter([p, p], [nll_1, nll_0], color='black', s=50, zorder=5)
    
    plt.axvline(x=0.5, color='g', linestyle='--', alpha=0.5, label='Uncertain (p=0.5)')
    plt.xlabel('Predicted Probability (p)', fontsize=12)
    plt.ylabel('Negative Log Likelihood', fontsize=12)
    plt.title('NLL as a Function of Predicted Probability', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 5)  # Limit y-axis for better visualization
    
    # Save the plot
    plt.savefig("nll_penalty.png", dpi=150, bbox_inches="tight")
    print("\nNLL penalty visualization saved to 'nll_penalty.png'")
    
    print("\nKey observations:")
    print("  - When true label = 1: NLL is high when p is low (wrong prediction)")
    print("  - When true label = 0: NLL is high when p is high (wrong prediction)")
    print("  - NLL approaches infinity as prediction becomes very wrong")
    print("  - NLL is minimized when prediction matches true label with high confidence")
    
    return probabilities, nll_class_1, nll_class_0


def demonstrate_nll_pytorch():
    """
    Demonstrates computing NLL using PyTorch.
    Shows both manual computation and using built-in loss functions.
    """
    print("\n" + "=" * 60)
    print("NLL with PyTorch")
    print("=" * 60)
    
    # Example: Binary classification with sigmoid output
    # Model outputs logits
    logits = torch.tensor([[2.0], [-1.0], [0.5], [-2.0], [3.0]], dtype=torch.float32)
    true_labels = torch.tensor([1, 0, 1, 0, 1], dtype=torch.float32)
    
    # Apply sigmoid to get probabilities
    probabilities = torch.sigmoid(logits).squeeze()  # Shape: (5,)
    
    # Method 1: Manual computation
    epsilon = 1e-15
    probs_clipped = torch.clamp(probabilities, epsilon, 1 - epsilon)
    nll_manual = -(true_labels * torch.log(probs_clipped) + 
                   (1 - true_labels) * torch.log(1 - probs_clipped))
    
    # Method 2: Using BCE loss (Binary Cross Entropy = NLL for binary classification)
    # Note: BCEWithLogitsLoss combines sigmoid + NLL for numerical stability
    bce_loss = nn.BCELoss()
    nll_bce = bce_loss(probabilities, true_labels)
    
    # Method 3: Using BCEWithLogitsLoss (recommended - more numerically stable)
    # This applies sigmoid internally and computes NLL
    bce_with_logits = nn.BCEWithLogitsLoss()
    nll_bce_logits = bce_with_logits(logits.squeeze(), true_labels)
    
    # Method 4: Using nn.NLLLoss (requires log-probabilities)
    # NLLLoss expects log-probabilities and class indices (0, 1) as targets
    # For binary classification, we need to convert probabilities to log-probabilities
    # Shape: (batch_size, num_classes) = (5, 2) for binary classification
    log_probs = torch.stack([
        torch.log(1 - probabilities + 1e-15),  # log P(class=0)
        torch.log(probabilities + 1e-15)       # log P(class=1)
    ], dim=1)  # Shape: (5, 2)
    
    true_labels_indices = true_labels.long()  # Convert to class indices (0 or 1)
    nll_loss = nn.NLLLoss()
    nll_nllloss = nll_loss(log_probs, true_labels_indices)
    
    print("\nInput logits:", logits.squeeze().tolist())
    print("True labels:", true_labels.tolist())
    print("\nProbabilities (after sigmoid):", probabilities.tolist())
    
    print("\nNLL computation methods:")
    print("-" * 50)
    print("Manual NLL per sample:")
    for i, nll in enumerate(nll_manual):
        print(f"  Sample {i+1}: {nll.item():.4f}")
    
    print(f"\nAverage NLL (manual): {nll_manual.mean().item():.4f}")
    print(f"BCE Loss (from probabilities): {nll_bce.item():.4f}")
    print(f"BCEWithLogitsLoss (from logits): {nll_bce_logits.item():.4f}")
    print(f"NLLLoss (from log-probabilities): {nll_nllloss.item():.4f}")
    
    print("\nNote: BCEWithLogitsLoss is preferred for binary classification because:")
    print("  - It's numerically stable (avoids log(0) issues)")
    print("  - It combines sigmoid + NLL in one operation")
    print("  - It's more efficient")
    print("  - It's designed specifically for binary classification")
    print("\nNote: NLLLoss is more commonly used for multi-class classification")
    print("  - It expects log-probabilities (from log-softmax)")
    print("  - It expects class indices (0, 1, 2, ...) as targets")
    
    return nll_manual, nll_bce, nll_bce_logits, nll_nllloss


def demonstrate_nllloss_pytorch():
    """
    Demonstrates nn.NLLLoss in detail.
    Shows how it differs from BCELoss and when to use each.
    """
    print("\n" + "=" * 60)
    print("nn.NLLLoss in Detail")
    print("=" * 60)
    
    print("\nKey differences between loss functions:")
    print("-" * 60)
    print("BCELoss / BCEWithLogitsLoss:")
    print("  - For binary classification")
    print("  - Input: probabilities (BCELoss) or logits (BCEWithLogitsLoss)")
    print("  - Target: probabilities (0.0 or 1.0)")
    print("  - Output: single loss value")
    print("\nNLLLoss:")
    print("  - For multi-class classification (can be used for binary too)")
    print("  - Input: log-probabilities (shape: batch_size, num_classes)")
    print("  - Target: class indices (0, 1, 2, ...)")
    print("  - Output: single loss value")
    
    # Example 1: Binary classification with NLLLoss
    print("\n" + "-" * 60)
    print("Example 1: Binary Classification with NLLLoss")
    print("-" * 60)
    
    logits = torch.tensor([[2.0], [-1.0], [0.5]], dtype=torch.float32)
    true_labels = torch.tensor([1, 0, 1], dtype=torch.long)  # Class indices
    
    # Convert logits to log-probabilities for binary classification
    # Option 1: Using sigmoid then log
    probabilities = torch.sigmoid(logits.squeeze())
    log_probs_binary = torch.stack([
        torch.log(1 - probabilities + 1e-15),  # log P(class=0)
        torch.log(probabilities + 1e-15)        # log P(class=1)
    ], dim=1)  # Shape: (3, 2)
    
    # Option 2: Using log-softmax (more common for multi-class)
    # For binary, this is equivalent to sigmoid + log
    log_probs_logsoftmax = torch.nn.functional.log_softmax(
        torch.cat([torch.zeros_like(logits), logits], dim=1), dim=1
    )  # Shape: (3, 2)
    
    nll_loss = nn.NLLLoss()
    loss1 = nll_loss(log_probs_binary, true_labels)
    loss2 = nll_loss(log_probs_logsoftmax, true_labels)
    
    print(f"Logits: {logits.squeeze().tolist()}")
    print(f"True labels (indices): {true_labels.tolist()}")
    print(f"\nLog-probabilities (from sigmoid):")
    print(log_probs_binary)
    print(f"\nNLLLoss (sigmoid method): {loss1.item():.4f}")
    print(f"NLLLoss (log-softmax method): {loss2.item():.4f}")
    
    # Example 2: Multi-class classification (where NLLLoss is more common)
    print("\n" + "-" * 60)
    print("Example 2: Multi-Class Classification with NLLLoss")
    print("-" * 60)
    
    # 3-class classification problem
    logits_multi = torch.tensor([
        [2.0, 1.0, 0.1],  # Sample 1: class 0 is most likely
        [0.5, 2.5, 0.3],  # Sample 2: class 1 is most likely
        [0.1, 0.2, 3.0],  # Sample 3: class 2 is most likely
    ], dtype=torch.float32)
    
    true_labels_multi = torch.tensor([0, 1, 2], dtype=torch.long)  # Class indices
    
    # Apply log-softmax to get log-probabilities
    log_probs_multi = torch.nn.functional.log_softmax(logits_multi, dim=1)
    
    # Compute NLLLoss
    nll_loss_multi = nn.NLLLoss()
    loss_multi = nll_loss_multi(log_probs_multi, true_labels_multi)
    
    print(f"Logits shape: {logits_multi.shape}")
    print(f"Log-probabilities shape: {log_probs_multi.shape}")
    print(f"True labels (indices): {true_labels_multi.tolist()}")
    print(f"\nLog-probabilities:")
    print(log_probs_multi)
    print(f"\nNLLLoss: {loss_multi.item():.4f}")
    
    # Compare with CrossEntropyLoss (which combines log-softmax + NLLLoss)
    print("\n" + "-" * 60)
    print("Note: CrossEntropyLoss = LogSoftmax + NLLLoss")
    print("-" * 60)
    ce_loss = nn.CrossEntropyLoss()
    loss_ce = ce_loss(logits_multi, true_labels_multi)
    print(f"CrossEntropyLoss (from logits): {loss_ce.item():.4f}")
    print(f"NLLLoss (from log-probabilities): {loss_multi.item():.4f}")
    print("  → They should be the same!")
    
    print("\n" + "-" * 60)
    print("Summary: When to use which loss?")
    print("-" * 60)
    print("Binary Classification:")
    print("  ✓ BCEWithLogitsLoss (recommended)")
    print("  ✓ BCELoss (if you already have probabilities)")
    print("  → NLLLoss (works but less common)")
    print("\nMulti-Class Classification:")
    print("  ✓ CrossEntropyLoss (recommended - combines log-softmax + NLLLoss)")
    print("  ✓ NLLLoss (if you already have log-probabilities)")
    
    return loss1, loss_multi, loss_ce


def demonstrate_nll_training_example():
    """
    Demonstrates how NLL is used during training.
    Shows how the loss decreases as predictions improve.
    """
    print("\n" + "=" * 60)
    print("NLL in Training Context")
    print("=" * 60)
    
    # Simulate training progress
    # True labels for a batch
    true_labels = np.array([1, 0, 1, 0, 1])
    
    # Simulate predictions at different training stages
    # Early: random/bad predictions
    # Middle: improving predictions
    # Late: good predictions
    stages = {
        "Early (random)": np.array([0.4, 0.6, 0.5, 0.5, 0.4]),
        "Middle (improving)": np.array([0.7, 0.3, 0.6, 0.2, 0.8]),
        "Late (good)": np.array([0.95, 0.05, 0.9, 0.1, 0.98]),
    }
    
    print("\nTraining progress simulation:")
    print("-" * 70)
    print("Stage           | Predictions          | Avg NLL | Accuracy")
    print("-" * 70)
    
    for stage_name, predictions in stages.items():
        nll_values = negative_log_likelihood_batch_numpy(predictions, true_labels)
        avg_nll = np.mean(nll_values)
        predictions_binary = (predictions > 0.5).astype(int)
        accuracy = np.mean(predictions_binary == true_labels)
        
        pred_str = " ".join([f"{p:.2f}" for p in predictions])
        print(f"{stage_name:15} | {pred_str:20} | {avg_nll:.4f}  | {accuracy:.2%}")
    
    print("\nObservations:")
    print("  - As model learns, predictions get closer to true labels")
    print("  - Average NLL decreases as predictions improve")
    print("  - Lower NLL correlates with higher accuracy")
    print("  - During training, we use gradient descent to minimize NLL")


def demonstrate_nll_vs_other_losses():
    """
    Compares NLL with other loss functions to show why NLL is preferred.
    """
    print("\n" + "=" * 60)
    print("NLL vs Other Loss Functions")
    print("=" * 60)
    
    # Example: predicted probability and true label
    probability = 0.1
    true_label = 1  # Wrong prediction!
    
    # Different loss functions
    nll = negative_log_likelihood_numpy(probability, true_label)
    
    # Mean Squared Error (MSE) - not ideal for classification
    mse = (probability - true_label) ** 2
    
    # Mean Absolute Error (MAE) - also not ideal
    mae = abs(probability - true_label)
    
    print(f"\nExample: Predicted probability = {probability}, True label = {true_label}")
    print(f"  (This is a WRONG prediction - model predicted class 0 but true class is 1)")
    print("\nLoss values:")
    print(f"  NLL (Negative Log Likelihood): {nll:.4f}")
    print(f"  MSE (Mean Squared Error):      {mse:.4f}")
    print(f"  MAE (Mean Absolute Error):     {mae:.4f}")
    
    print("\nWhy NLL is preferred for classification:")
    print("  1. NLL heavily penalizes wrong predictions (exponential penalty)")
    print("  2. NLL encourages confident correct predictions")
    print("  3. NLL has nice theoretical properties (maximum likelihood estimation)")
    print("  4. NLL works well with gradient descent")
    print("  5. MSE/MAE don't distinguish between 'confidently wrong' and 'uncertain'")


if __name__ == "__main__":
    # Demonstrate basic NLL computation
    demonstrate_nll_basic()
    
    # Demonstrate NLL from sigmoid outputs
    demonstrate_nll_from_sigmoid()
    
    # Demonstrate NLL penalty visualization
    try:
        probs, nll_1, nll_0 = demonstrate_nll_penalty()
    except Exception as e:
        print(f"\nNote: Could not generate plot: {e}")
        print("      (matplotlib may not be installed or display not available)")
    
    # Demonstrate PyTorch implementation
    nll_manual, nll_bce, nll_bce_logits, nll_nllloss = demonstrate_nll_pytorch()
    
    # Demonstrate nn.NLLLoss in detail
    loss1, loss_multi, loss_ce = demonstrate_nllloss_pytorch()
    
    # Demonstrate NLL in training context
    demonstrate_nll_training_example()
    
    # Compare NLL with other loss functions
    demonstrate_nll_vs_other_losses()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Negative Log Likelihood (NLL) is the standard loss for binary classification")
    print("  2. NLL = -log(p) when true label = 1, or -log(1-p) when true label = 0")
    print("  3. NLL heavily penalizes wrong predictions (exponential penalty)")
    print("  4. NLL encourages confident correct predictions (low loss)")
    print("  5. During training, we minimize average NLL over all training samples")
    print("  6. PyTorch provides multiple loss functions:")
    print("     - BCEWithLogitsLoss: recommended for binary classification (sigmoid + NLL)")
    print("     - BCELoss: for binary classification with probabilities")
    print("     - NLLLoss: for multi-class classification with log-probabilities")
    print("     - CrossEntropyLoss: for multi-class (combines log-softmax + NLLLoss)")
    print("  7. NLL is preferred over MSE/MAE for classification problems")

