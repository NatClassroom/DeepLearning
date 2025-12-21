import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def generate_1d_binary_data(n_samples=200, noise=0.1, random_seed=42):
    """
    Generate a simple 1D binary classification dataset.
    
    Creates data where:
    - x is 1-dimensional (single feature)
    - y is binary (0 or 1)
    - Classes are separated by a threshold on x
    
    Args:
        n_samples: Number of data points to generate
        noise: Amount of noise to add (controls class overlap)
        random_seed: Random seed for reproducibility
    
    Returns:
        x: 1D feature array, shape (n_samples, 1)
        y: Binary labels, shape (n_samples,)
    """
    np.random.seed(random_seed)
    
    # Generate x values uniformly in range [-2, 2]
    x = np.random.uniform(-2, 2, size=(n_samples, 1))
    
    # Create a simple decision boundary: y = 1 if x > 0, else y = 0
    # Add some noise to make it more realistic
    y_true = (x.flatten() > 0).astype(int)
    
    # Add noise: randomly flip some labels based on distance from boundary
    noise_mask = np.random.random(n_samples) < noise
    y = y_true.copy()
    y[noise_mask] = 1 - y[noise_mask]  # Flip labels for noisy samples
    
    print(f"\nGenerated {n_samples} samples")
    print(f"  - x shape: {x.shape} (1D feature)")
    print(f"  - y shape: {y.shape} (binary labels: 0 or 1)")
    print(f"  - Class 0: {np.sum(y == 0)} samples")
    print(f"  - Class 1: {np.sum(y == 1)} samples")
    
    return x, y


def demonstrate_data_generation():
    """
    Demonstrates the generated 1D binary classification data.
    """
    print("\n" + "=" * 60)
    print("1D Binary Classification Data Generation")
    print("=" * 60)
    
    x, y = generate_1d_binary_data(n_samples=200, noise=0.1)
    
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
    
    # Add decision boundary line
    plt.axvline(x=0, color='green', linestyle='--', linewidth=2, 
                alpha=0.7, label='True Decision Boundary (x=0)')
    
    plt.xlabel('Feature x (1D)', fontsize=12)
    plt.ylabel('Class Label y', fontsize=12)
    plt.title('1D Binary Classification Dataset\n(x: 1D feature, y: binary labels 0 or 1)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.2, 1.2)
    plt.xlim(-2.2, 2.2)
    
    # Save the plot
    plt.savefig("binary_classification_1d_data.png", dpi=150, bbox_inches="tight")
    print("\nData visualization saved to 'binary_classification_1d_data.png'")
    
    return x, y


def create_binary_classifier(input_dim=1):
    """
    Create a simple binary classifier using PyTorch.
    
    Architecture:
    - Linear layer: 1 input -> 1 output (logit)
    - Sigmoid: converts logit to probability
    
    Args:
        input_dim: Number of input features (1 for 1D data)
    
    Returns:
        PyTorch model
    """
    model = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=1),  # Single neuron
        nn.Sigmoid()  # Convert logit to probability
    )
    
    print(f"\nModel architecture:")
    print(f"  Linear({input_dim} -> 1): computes logit from 1D input")
    print(f"  Sigmoid(): converts logit to probability [0, 1]")
    
    return model


def train_binary_classifier(model, x_train, y_train, epochs=100, lr=0.1):
    """
    Train a binary classifier on 1D data.
    
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
    print("Training Binary Classifier")
    print("=" * 60)
    
    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    # Loss function: Binary Cross Entropy (NLL for binary classification)
    criterion = nn.BCELoss()
    
    # Optimizer: Stochastic Gradient Descent
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Training loop
    losses = []
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(x_tensor).squeeze()  # Shape: (n_samples,)
        loss = criterion(outputs, y_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            # Compute accuracy
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_tensor).float().mean().item()
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2%}")
    
    print("-" * 60)
    final_loss = losses[-1]
    final_outputs = model(x_tensor).squeeze()
    final_predictions = (final_outputs > 0.5).float()
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
    plt.ylabel('Loss (BCE)', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig("binary_classification_training.png", dpi=150, bbox_inches="tight")
    print("\nTraining progress saved to 'binary_classification_training.png'")


def visualize_decision_boundary(model, x_data, y_data):
    """
    Visualize the learned decision boundary of the trained model.
    
    Args:
        model: Trained PyTorch model
        x_data: Feature data, shape (n_samples, 1)
        y_data: True labels, shape (n_samples,)
    """
    print("\n" + "=" * 60)
    print("Visualizing Learned Decision Boundary")
    print("=" * 60)
    
    # Create a fine grid of x values for plotting the decision boundary
    x_grid = np.linspace(-2.5, 2.5, 1000).reshape(-1, 1)
    x_grid_tensor = torch.tensor(x_grid, dtype=torch.float32)
    
    # Get model predictions for the grid
    with torch.no_grad():
        model.eval()
        probabilities = model(x_grid_tensor).squeeze().numpy()
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot the decision boundary (probability curve)
    plt.plot(x_grid, probabilities, 'g-', linewidth=3, 
             label='Learned Decision Boundary (P(y=1|x))', zorder=2)
    
    # Plot class 0 points
    class_0_mask = y_data == 0
    plt.scatter(x_data[class_0_mask], y_data[class_0_mask], 
                c='red', marker='o', s=50, alpha=0.6, 
                label='Class 0 (True)', zorder=3, edgecolors='darkred')
    
    # Plot class 1 points
    class_1_mask = y_data == 1
    plt.scatter(x_data[class_1_mask], y_data[class_1_mask], 
                c='blue', marker='s', s=50, alpha=0.6, 
                label='Class 1 (True)', zorder=3, edgecolors='darkblue')
    
    # Add threshold line at y=0.5
    plt.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, 
                alpha=0.7, label='Decision Threshold (p=0.5)')
    
    # Add true decision boundary
    plt.axvline(x=0, color='purple', linestyle='--', linewidth=2, 
                alpha=0.5, label='True Decision Boundary (x=0)')
    
    plt.xlabel('Feature x (1D)', fontsize=12)
    plt.ylabel('Probability / Class Label', fontsize=12)
    plt.title('Binary Classification: Learned Decision Boundary\n(x: 1D feature, y: binary labels 0 or 1)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.ylim(-0.1, 1.1)
    plt.xlim(-2.5, 2.5)
    
    # Save the plot
    plt.savefig("binary_classification_decision_boundary.png", dpi=150, bbox_inches="tight")
    print("\nDecision boundary visualization saved to 'binary_classification_decision_boundary.png'")
    
    # Print model parameters
    with torch.no_grad():
        weight = model[0].weight.item()
        bias = model[0].bias.item()
        print(f"\nLearned model parameters:")
        print(f"  Weight: {weight:.4f}")
        print(f"  Bias: {bias:.4f}")
        print(f"  Decision boundary (where p=0.5): x = {-bias/weight:.4f}")


def demonstrate_prediction_examples(model, x_data, y_data):
    """
    Show some example predictions from the trained model.
    """
    print("\n" + "=" * 60)
    print("Example Predictions")
    print("=" * 60)
    
    # Select some example points
    example_indices = [0, 50, 100, 150, 199] if len(x_data) > 200 else list(range(len(x_data)))[::len(x_data)//5]
    
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    with torch.no_grad():
        model.eval()
        probabilities = model(x_tensor).squeeze().numpy()
        predictions = (probabilities > 0.5).astype(int)
    
    print("\nExample predictions:")
    print("-" * 70)
    print("x (1D)    | True y | Predicted p | Predicted y | Correct?")
    print("-" * 70)
    
    for idx in example_indices[:10]:  # Show up to 10 examples
        x_val = x_data[idx, 0]
        y_true = y_data[idx]
        p_pred = probabilities[idx]
        y_pred = predictions[idx]
        correct = "✓" if y_pred == y_true else "✗"
        print(f"{x_val:8.3f} |   {y_true}   |   {p_pred:.4f}   |     {y_pred}      |   {correct}")


def demonstrate_complete_pipeline():
    """
    Demonstrates the complete binary classification pipeline:
    1. Generate 1D data with binary labels
    2. Create a model
    3. Train the model
    4. Visualize results
    """
    print("\n" + "=" * 60)
    print("Complete Binary Classification Pipeline")
    print("=" * 60)
    
    # Step 1: Generate data
    x, y = generate_1d_binary_data(n_samples=200, noise=0.1)
    
    # Step 2: Create model
    model = create_binary_classifier(input_dim=1)
    
    # Step 3: Train model
    losses = train_binary_classifier(model, x, y, epochs=200, lr=0.1)
    
    # Step 4: Visualize training progress
    visualize_training_progress(losses)
    
    # Step 5: Visualize decision boundary
    visualize_decision_boundary(model, x, y)
    
    # Step 6: Show example predictions
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
    print("  1. Binary classification: predict one of two classes (0 or 1)")
    print("  2. For 1D input (single feature x), we use a simple linear model + sigmoid")
    print("  3. The model learns a decision boundary that separates the two classes")
    print("  4. Training minimizes Binary Cross Entropy (BCE) loss")
    print("  5. The sigmoid output gives probability P(y=1|x), threshold at 0.5 for prediction")
    print("  6. The learned decision boundary is visualized as a smooth curve")
    print("\nGenerated files:")
    print("  - binary_classification_1d_data.png: Original data visualization")
    print("  - binary_classification_training.png: Training loss over time")
    print("  - binary_classification_decision_boundary.png: Learned decision boundary")

