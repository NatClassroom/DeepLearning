import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def generate_1d_multiclass_data(n_samples=300, noise=0.1, random_seed=42):
    """
    Generate a simple 1D multi-class classification dataset.
    
    Creates data where:
    - x is 1-dimensional (single feature)
    - y is multi-class (0, 1, or 2)
    - Classes are separated by thresholds on x
    
    Args:
        n_samples: Number of data points to generate
        noise: Amount of noise to add (controls class overlap)
        random_seed: Random seed for reproducibility
    
    Returns:
        x: 1D feature array, shape (n_samples, 1)
        y: Multi-class labels, shape (n_samples,)
    """
    np.random.seed(random_seed)
    
    # Generate x values uniformly in range [-3, 3]
    x = np.random.uniform(-3, 3, size=(n_samples, 1))
    
    # Create decision boundaries: 
    # y = 0 if x < -1
    # y = 1 if -1 <= x < 1
    # y = 2 if x >= 1
    y_true = np.zeros(n_samples, dtype=int)
    y_true[(x.flatten() >= -1) & (x.flatten() < 1)] = 1
    y_true[x.flatten() >= 1] = 2
    
    # Add noise: randomly flip some labels based on distance from boundaries
    noise_mask = np.random.random(n_samples) < noise
    y = y_true.copy()
    for i in range(n_samples):
        if noise_mask[i]:
            # Randomly assign to a different class
            possible_classes = [0, 1, 2]
            possible_classes.remove(y_true[i])
            y[i] = np.random.choice(possible_classes)
    
    print(f"\nGenerated {n_samples} samples")
    print(f"  - x shape: {x.shape} (1D feature)")
    print(f"  - y shape: {y.shape} (multi-class labels: 0, 1, or 2)")
    print(f"  - Class 0: {np.sum(y == 0)} samples")
    print(f"  - Class 1: {np.sum(y == 1)} samples")
    print(f"  - Class 2: {np.sum(y == 2)} samples")
    
    return x, y


def demonstrate_data_generation():
    """
    Demonstrates the generated 1D multi-class classification data.
    """
    print("\n" + "=" * 60)
    print("1D Multi-Class Classification Data Generation")
    print("=" * 60)
    
    x, y = generate_1d_multiclass_data(n_samples=300, noise=0.1)
    
    # Visualize the data
    plt.figure(figsize=(10, 6))
    
    # Plot class 0 points
    class_0_mask = y == 0
    plt.scatter(x[class_0_mask], np.zeros(np.sum(class_0_mask)), 
                c='red', marker='o', s=50, alpha=0.6, label='Class 0', zorder=3)
    
    # Plot class 1 points
    class_1_mask = y == 1
    plt.scatter(x[class_1_mask], np.ones(np.sum(class_1_mask)), 
                c='blue', marker='s', s=50, alpha=0.6, label='Class 1', zorder=3)
    
    # Plot class 2 points
    class_2_mask = y == 2
    plt.scatter(x[class_2_mask], np.full(np.sum(class_2_mask), 2), 
                c='green', marker='^', s=50, alpha=0.6, label='Class 2', zorder=3)
    
    # Add decision boundary lines
    plt.axvline(x=-1, color='purple', linestyle='--', linewidth=2, 
                alpha=0.7, label='True Decision Boundary (x=-1)')
    plt.axvline(x=1, color='purple', linestyle='--', linewidth=2, 
                alpha=0.7)
    
    plt.xlabel('Feature x (1D)', fontsize=12)
    plt.ylabel('Class Label y', fontsize=12)
    plt.title('1D Multi-Class Classification Dataset\n(x: 1D feature, y: multi-class labels 0, 1, or 2)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.2, 2.2)
    plt.xlim(-3.2, 3.2)
    
    # Save the plot
    plt.savefig("multiclass_classification_1d_data.png", dpi=150, bbox_inches="tight")
    print("\nData visualization saved to 'multiclass_classification_1d_data.png'")
    
    return x, y


def create_multiclass_classifier(input_dim=1, num_classes=3):
    """
    Create a simple multi-class classifier using PyTorch.
    
    Architecture:
    - Linear layer: 1 input -> 3 outputs (logits for each class)
    - Softmax: converts logits to probabilities (applied during inference)
    
    Note: We don't include softmax in the model because:
    - CrossEntropyLoss expects raw logits (it applies log-softmax internally)
    - This is more numerically stable
    
    Args:
        input_dim: Number of input features (1 for 1D data)
        num_classes: Number of classes (3 for this problem)
    
    Returns:
        PyTorch model
    """
    model = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=num_classes)  # 1 -> 3 logits
        # Note: No softmax here! CrossEntropyLoss expects raw logits
    )
    
    print(f"\nModel architecture:")
    print(f"  Linear({input_dim} -> {num_classes}): computes logits from 1D input")
    print(f"  Output: 3 logits (one for each class)")
    print(f"  Softmax: applied during inference to get probabilities")
    
    return model


def train_multiclass_classifier(model, x_train, y_train, epochs=200, lr=0.1):
    """
    Train a multi-class classifier on 1D data.
    
    Args:
        model: PyTorch model
        x_train: Training features, shape (n_samples, 1)
        y_train: Training labels, shape (n_samples,)
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        List of loss values during training
    """
    print("\n" + "=" * 60)
    print("Training Multi-Class Classifier")
    print("=" * 60)
    
    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)  # Long for class indices
    
    # Loss function: Cross Entropy (combines log-softmax + NLLLoss)
    # This is the standard loss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: Stochastic Gradient Descent
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Training loop
    losses = []
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Forward pass
        logits = model(x_tensor)  # Shape: (n_samples, 3)
        loss = criterion(logits, y_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            # Compute accuracy
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == y_tensor).float().mean().item()
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2%}")
    
    print("-" * 60)
    final_loss = losses[-1]
    with torch.no_grad():
        final_logits = model(x_tensor)
        final_predictions = torch.argmax(final_logits, dim=1)
        final_accuracy = (final_predictions == y_tensor).float().mean().item()
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Final Accuracy: {final_accuracy:.2%}")
    
    return losses


def visualize_training_progress(losses):
    """
    Visualize the training loss over epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Cross Entropy)', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig("multiclass_classification_training.png", dpi=150, bbox_inches="tight")
    print("\nTraining progress saved to 'multiclass_classification_training.png'")


def visualize_decision_boundary(model, x_data, y_data):
    """
    Visualize the learned decision boundaries of the trained model.
    
    Args:
        model: Trained PyTorch model
        x_data: Feature data, shape (n_samples, 1)
        y_data: True labels, shape (n_samples,)
    """
    print("\n" + "=" * 60)
    print("Visualizing Learned Decision Boundaries")
    print("=" * 60)
    
    # Create a fine grid of x values for plotting the decision boundaries
    x_grid = np.linspace(-3.5, 3.5, 1000).reshape(-1, 1)
    x_grid_tensor = torch.tensor(x_grid, dtype=torch.float32)
    
    # Get model predictions for the grid
    with torch.no_grad():
        model.eval()
        logits = model(x_grid_tensor)  # Shape: (1000, 3)
        probabilities = torch.softmax(logits, dim=1).numpy()  # Shape: (1000, 3)
        predictions = torch.argmax(logits, dim=1).numpy()  # Shape: (1000,)
    
    # Create the plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top plot: Class probabilities
    ax1 = axes[0]
    ax1.plot(x_grid, probabilities[:, 0], 'r-', linewidth=2, 
             label='P(y=0|x)', alpha=0.8)
    ax1.plot(x_grid, probabilities[:, 1], 'b-', linewidth=2, 
             label='P(y=1|x)', alpha=0.8)
    ax1.plot(x_grid, probabilities[:, 2], 'g-', linewidth=2, 
             label='P(y=2|x)', alpha=0.8)
    
    # Plot class 0 points
    class_0_mask = y_data == 0
    ax1.scatter(x_data[class_0_mask], np.zeros(np.sum(class_0_mask)), 
                c='red', marker='o', s=50, alpha=0.6, 
                label='Class 0 (True)', zorder=3, edgecolors='darkred')
    
    # Plot class 1 points
    class_1_mask = y_data == 1
    ax1.scatter(x_data[class_1_mask], np.ones(np.sum(class_1_mask)), 
                c='blue', marker='s', s=50, alpha=0.6, 
                label='Class 1 (True)', zorder=3, edgecolors='darkblue')
    
    # Plot class 2 points
    class_2_mask = y_data == 2
    ax1.scatter(x_data[class_2_mask], np.full(np.sum(class_2_mask), 2), 
                c='green', marker='^', s=50, alpha=0.6, 
                label='Class 2 (True)', zorder=3, edgecolors='darkgreen')
    
    # Add true decision boundaries
    ax1.axvline(x=-1, color='purple', linestyle='--', linewidth=2, 
                alpha=0.5, label='True Boundaries')
    ax1.axvline(x=1, color='purple', linestyle='--', linewidth=2, alpha=0.5)
    
    ax1.set_xlabel('Feature x (1D)', fontsize=12)
    ax1.set_ylabel('Probability / Class Label', fontsize=12)
    ax1.set_title('Multi-Class Classification: Class Probabilities\n(x: 1D feature, y: multi-class labels 0, 1, or 2)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    ax1.set_ylim(-0.1, 2.1)
    ax1.set_xlim(-3.5, 3.5)
    
    # Bottom plot: Predicted class regions
    ax2 = axes[1]
    # Create colored regions for each predicted class
    for i in range(len(x_grid) - 1):
        pred_class = predictions[i]
        if pred_class == 0:
            ax2.axvspan(x_grid[i, 0], x_grid[i+1, 0], alpha=0.3, color='red')
        elif pred_class == 1:
            ax2.axvspan(x_grid[i, 0], x_grid[i+1, 0], alpha=0.3, color='blue')
        else:  # pred_class == 2
            ax2.axvspan(x_grid[i, 0], x_grid[i+1, 0], alpha=0.3, color='green')
    
    # Plot data points
    ax2.scatter(x_data[class_0_mask], np.zeros(np.sum(class_0_mask)), 
                c='red', marker='o', s=50, alpha=0.8, 
                label='Class 0', zorder=3, edgecolors='darkred')
    ax2.scatter(x_data[class_1_mask], np.ones(np.sum(class_1_mask)), 
                c='blue', marker='s', s=50, alpha=0.8, 
                label='Class 1', zorder=3, edgecolors='darkblue')
    ax2.scatter(x_data[class_2_mask], np.full(np.sum(class_2_mask), 2), 
                c='green', marker='^', s=50, alpha=0.8, 
                label='Class 2', zorder=3, edgecolors='darkgreen')
    
    # Add true decision boundaries
    ax2.axvline(x=-1, color='purple', linestyle='--', linewidth=2, 
                alpha=0.7, label='True Boundaries')
    ax2.axvline(x=1, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Feature x (1D)', fontsize=12)
    ax2.set_ylabel('Class Label y', fontsize=12)
    ax2.set_title('Learned Decision Regions', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_ylim(-0.2, 2.2)
    ax2.set_xlim(-3.5, 3.5)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("multiclass_classification_decision_boundary.png", dpi=150, bbox_inches="tight")
    print("\nDecision boundary visualization saved to 'multiclass_classification_decision_boundary.png'")
    
    # Print model parameters
    with torch.no_grad():
        weights = model[0].weight.numpy()  # Shape: (3, 1)
        biases = model[0].bias.numpy()  # Shape: (3,)
        print(f"\nLearned model parameters:")
        print(f"  Weights (3 classes): {weights.flatten()}")
        print(f"  Biases (3 classes): {biases}")
        print(f"\n  Class 0: logit = {weights[0, 0]:.4f} * x + {biases[0]:.4f}")
        print(f"  Class 1: logit = {weights[1, 0]:.4f} * x + {biases[1]:.4f}")
        print(f"  Class 2: logit = {weights[2, 0]:.4f} * x + {biases[2]:.4f}")


def demonstrate_prediction_examples(model, x_data, y_data):
    """
    Show some example predictions from the trained model.
    """
    print("\n" + "=" * 60)
    print("Example Predictions")
    print("=" * 60)
    
    # Select some example points
    example_indices = [0, 50, 100, 150, 200, 250, 299] if len(x_data) > 300 else list(range(len(x_data)))[::len(x_data)//7]
    
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    with torch.no_grad():
        model.eval()
        logits = model(x_tensor)  # Shape: (n_samples, 3)
        probabilities = torch.softmax(logits, dim=1).numpy()  # Shape: (n_samples, 3)
        predictions = torch.argmax(logits, dim=1).numpy()  # Shape: (n_samples,)
    
    print("\nExample predictions:")
    print("-" * 90)
    print("x (1D)    | True y | P(y=0|x) | P(y=1|x) | P(y=2|x) | Predicted y | Correct?")
    print("-" * 90)
    
    for idx in example_indices[:10]:  # Show up to 10 examples
        x_val = x_data[idx, 0]
        y_true = y_data[idx]
        p0, p1, p2 = probabilities[idx]
        y_pred = predictions[idx]
        correct = "✓" if y_pred == y_true else "✗"
        print(f"{x_val:8.3f} |   {y_true}   |  {p0:.4f}  |  {p1:.4f}  |  {p2:.4f}  |     {y_pred}      |   {correct}")


def demonstrate_softmax_function():
    """
    Demonstrates the softmax function and how it converts logits to probabilities.
    """
    print("\n" + "=" * 60)
    print("Softmax Function Demonstration")
    print("=" * 60)
    
    # Example logits for 3 classes
    logits_examples = [
        ([2.0, 1.0, 0.1], "Class 0 is most likely"),
        ([0.5, 2.5, 0.3], "Class 1 is most likely"),
        ([0.1, 0.2, 3.0], "Class 2 is most likely"),
        ([1.0, 1.0, 1.0], "All classes equally likely"),
        ([5.0, 2.0, 1.0], "Class 0 is very confident"),
    ]
    
    print("\nSoftmax converts logits to probabilities:")
    print("-" * 80)
    print("Logits (3 classes)      | Probabilities (after softmax)    | Interpretation")
    print("-" * 80)
    
    for logits, description in logits_examples:
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        probs = torch.softmax(logits_tensor, dim=0).numpy()
        pred_class = np.argmax(probs)
        
        logits_str = f"[{logits[0]:.1f}, {logits[1]:.1f}, {logits[2]:.1f}]"
        probs_str = f"[{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}]"
        print(f"{logits_str:25} | {probs_str:35} | {description} (pred: {pred_class})")
    
    print("\nKey properties of softmax:")
    print("  1. All probabilities sum to 1.0")
    print("  2. All probabilities are between 0 and 1")
    print("  3. Higher logit → higher probability")
    print("  4. The class with highest logit gets highest probability")
    print("  5. Differences in logits are amplified in probabilities")


def demonstrate_complete_pipeline():
    """
    Demonstrates the complete multi-class classification pipeline:
    1. Generate 1D data with multi-class labels
    2. Create a model
    3. Train the model
    4. Visualize results
    """
    print("\n" + "=" * 60)
    print("Complete Multi-Class Classification Pipeline")
    print("=" * 60)
    
    # Step 1: Generate data
    x, y = generate_1d_multiclass_data(n_samples=300, noise=0.1)
    
    # Step 2: Demonstrate softmax
    demonstrate_softmax_function()
    
    # Step 3: Create model
    model = create_multiclass_classifier(input_dim=1, num_classes=3)
    
    # Step 4: Train model
    losses = train_multiclass_classifier(model, x, y, epochs=200, lr=0.1)
    
    # Step 5: Visualize training progress
    visualize_training_progress(losses)
    
    # Step 6: Visualize decision boundaries
    visualize_decision_boundary(model, x, y)
    
    # Step 7: Show example predictions
    demonstrate_prediction_examples(model, x, y)
    
    return model, x, y


if __name__ == "__main__":
    # Demonstrate data generation
    x_data, y_data = demonstrate_data_generation()
    
    # Demonstrate complete pipeline
    trained_model, x_final, y_final = demonstrate_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Multi-class classification: predict one of multiple classes (0, 1, 2, ...)")
    print("  2. For 1D input (single feature x), we use a linear model with multiple outputs")
    print("  3. The model outputs logits (one per class), which are converted to probabilities via softmax")
    print("  4. Training minimizes Cross Entropy loss (combines log-softmax + NLLLoss)")
    print("  5. The softmax function ensures probabilities sum to 1 and are all positive")
    print("  6. Decision boundaries separate multiple class regions (more complex than binary)")
    print("  7. CrossEntropyLoss expects raw logits (not probabilities) for numerical stability")
    print("\nDifferences from binary classification:")
    print("  - Binary: 1 output logit → sigmoid → probability")
    print("  - Multi-class: N output logits → softmax → N probabilities")
    print("  - Binary: BCELoss or BCEWithLogitsLoss")
    print("  - Multi-class: CrossEntropyLoss (or NLLLoss with log-softmax)")
    print("\nGenerated files:")
    print("  - multiclass_classification_1d_data.png: Original data visualization")
    print("  - multiclass_classification_training.png: Training loss over time")
    print("  - multiclass_classification_decision_boundary.png: Learned decision boundaries")

