"""
Topic 1: Autoencoder Basics
===========================

This module demonstrates:
1. What is an autoencoder and why it's useful
2. Encoder-decoder architecture
3. Bottleneck/latent representation
4. Training an autoencoder for dimensionality reduction
5. Reconstruction loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# Create results directory if it doesn't exist
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def explain_autoencoder_concept():
    """
    Explain the concept of autoencoders with ASCII diagrams.
    """
    print("\n" + "=" * 70)
    print("What is an Autoencoder?")
    print("=" * 70)

    print("""
An autoencoder is a neural network that learns to compress data into a
lower-dimensional representation (encoding) and then reconstruct it back.

Architecture:
                          Latent Space
                          (Bottleneck)
                              |
    Input ──> [Encoder] ──> [z] ──> [Decoder] ──> Output (Reconstruction)
    (784)       ↓           (2-32)      ↓           (784)
              compress              decompress

Key Components:
---------------
1. ENCODER: Compresses input into a smaller latent representation
   - Input: High-dimensional data (e.g., 784 pixels for MNIST)
   - Output: Low-dimensional latent vector z (e.g., 2-32 dimensions)

2. LATENT SPACE (Bottleneck):
   - The compressed representation of the data
   - Forces the network to learn important features
   - Can be used for visualization (2D), clustering, or generation

3. DECODER: Reconstructs the original input from latent representation
   - Input: Latent vector z
   - Output: Reconstructed data (same dimension as original input)

Training Objective:
-------------------
Minimize reconstruction loss: ||x - decoder(encoder(x))||^2

The network learns to compress and reconstruct such that:
   reconstruction ≈ original input
""")


def explain_why_autoencoders():
    """
    Explain use cases for autoencoders.
    """
    print("\n" + "=" * 70)
    print("Why Use Autoencoders?")
    print("=" * 70)

    print("""
1. DIMENSIONALITY REDUCTION
   - Like PCA but can capture non-linear relationships
   - Compress high-dimensional data for storage/transmission

2. FEATURE LEARNING
   - Learn meaningful representations of data
   - Latent features can be used for downstream tasks

3. DENOISING
   - Train on noisy inputs, target clean outputs
   - Network learns to remove noise

4. ANOMALY DETECTION
   - Train on normal data
   - High reconstruction error = anomaly

5. GENERATIVE MODELS (VAE)
   - Variational autoencoders can generate new samples
   - Sample from latent space to create new data

6. DATA VISUALIZATION
   - Use 2D latent space to visualize high-dimensional data
   - Similar items cluster together in latent space
""")


class SimpleAutoencoder(nn.Module):
    """
    A simple autoencoder for understanding the basic architecture.

    Architecture:
    - Encoder: Linear(input_dim -> hidden_dim) -> ReLU -> Linear(hidden_dim -> latent_dim)
    - Decoder: Linear(latent_dim -> hidden_dim) -> ReLU -> Linear(hidden_dim -> input_dim)
    """

    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=32):
        super(SimpleAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()  # Optional: can help with latent space structure
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Output in [0, 1] for normalized images
        )

    def encode(self, x):
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, z):
        """Decode latent representation to reconstruction."""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass: encode then decode."""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z


def generate_simple_data(n_samples=500, random_seed=42):
    """
    Generate simple 2D data that lies on a 1D manifold (circle).
    This demonstrates how autoencoder can learn the underlying structure.

    Args:
        n_samples: Number of data points
        random_seed: Random seed for reproducibility

    Returns:
        data: Tensor of shape (n_samples, 2)
    """
    np.random.seed(random_seed)

    # Generate points on a circle with some noise
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    noise = np.random.randn(n_samples, 2) * 0.1

    x = np.cos(theta) + noise[:, 0]
    y = np.sin(theta) + noise[:, 1]

    data = np.stack([x, y], axis=1).astype(np.float32)

    print(f"\nGenerated {n_samples} samples on a noisy circle")
    print(f"  - Data shape: {data.shape}")
    print(f"  - Data lies on a 1D manifold (circle) in 2D space")

    return torch.tensor(data), theta


class TinyAutoencoder(nn.Module):
    """
    A tiny autoencoder: 2D input -> 1D latent -> 2D output.
    This demonstrates compression of a circle to its angle parameter.
    """

    def __init__(self):
        super(TinyAutoencoder, self).__init__()

        # 2D -> 1D (encode circle to angle-like representation)
        self.encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # 1D -> 2D (decode angle back to point on circle)
        self.decoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z


def train_autoencoder(model, data, epochs=500, lr=0.01):
    """
    Train an autoencoder using MSE reconstruction loss.

    Args:
        model: Autoencoder model
        data: Training data tensor
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        List of losses during training
    """
    print("\n" + "=" * 70)
    print("Training Autoencoder")
    print("=" * 70)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 70)

    for epoch in range(epochs):
        # Forward pass
        reconstruction, z = model(data)
        loss = criterion(reconstruction, data)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {loss.item():.6f}")

    print("-" * 70)
    print(f"Final Loss: {losses[-1]:.6f}")

    return losses


def visualize_autoencoder_results(model, data, theta, losses):
    """
    Visualize the autoencoder results: original, reconstruction, and latent space.
    """
    print("\n" + "=" * 70)
    print("Visualizing Results")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        reconstruction, z = model(data)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Training loss
    axes[0, 0].plot(losses, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('MSE Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Original data
    axes[0, 1].scatter(data[:, 0].numpy(), data[:, 1].numpy(),
                       c=theta, cmap='hsv', alpha=0.6, s=20)
    axes[0, 1].set_xlabel('x', fontsize=12)
    axes[0, 1].set_ylabel('y', fontsize=12)
    axes[0, 1].set_title('Original Data (2D)', fontsize=14)
    axes[0, 1].set_aspect('equal')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Latent space (1D)
    z_np = z.numpy().flatten()
    axes[1, 0].scatter(z_np, np.zeros_like(z_np), c=theta, cmap='hsv', alpha=0.6, s=20)
    axes[1, 0].set_xlabel('Latent z', fontsize=12)
    axes[1, 0].set_title('Latent Space (1D)', fontsize=14)
    axes[1, 0].set_ylim(-0.5, 0.5)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Reconstruction
    axes[1, 1].scatter(reconstruction[:, 0].numpy(), reconstruction[:, 1].numpy(),
                       c=theta, cmap='hsv', alpha=0.6, s=20)
    axes[1, 1].set_xlabel('x', fontsize=12)
    axes[1, 1].set_ylabel('y', fontsize=12)
    axes[1, 1].set_title('Reconstruction (2D)', fontsize=14)
    axes[1, 1].set_aspect('equal')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "autoencoder_basics.png"), dpi=150, bbox_inches="tight")
    print("\nVisualization saved to 'autoencoder_basics.png'")


def demonstrate_latent_interpolation(model, data):
    """
    Demonstrate interpolation in latent space.
    Moving through latent space should produce smooth transitions.
    """
    print("\n" + "=" * 70)
    print("Latent Space Interpolation")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        _, z = model(data)

    z_min, z_max = z.min().item(), z.max().item()

    # Interpolate through latent space
    z_interp = torch.linspace(z_min, z_max, 100).reshape(-1, 1)

    with torch.no_grad():
        reconstructed = model.decoder(z_interp)

    plt.figure(figsize=(10, 5))

    # Plot original data
    plt.scatter(data[:, 0].numpy(), data[:, 1].numpy(),
                alpha=0.3, s=10, c='gray', label='Original Data')

    # Plot interpolation path
    plt.plot(reconstructed[:, 0].numpy(), reconstructed[:, 1].numpy(),
             'r-', linewidth=2, label='Latent Interpolation')
    plt.scatter(reconstructed[::10, 0].numpy(), reconstructed[::10, 1].numpy(),
                c='red', s=50, zorder=5)

    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Interpolation Through Latent Space\n(Red line shows decoded path)', fontsize=14)
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(RESULTS_DIR, "autoencoder_interpolation.png"), dpi=150, bbox_inches="tight")
    print("\nInterpolation visualization saved to 'autoencoder_interpolation.png'")

    print("\nKey insight:")
    print("  - Moving smoothly through latent space produces smooth")
    print("    transitions in the output space")
    print("  - The 1D latent variable captures the 'angle' of the circle")


def demonstrate_autoencoder_basics():
    """
    Main demonstration of autoencoder basics.
    """
    # Explain concepts
    explain_autoencoder_concept()
    explain_why_autoencoders()

    # Generate simple data
    print("\n" + "=" * 70)
    print("Demonstration: 2D Circle -> 1D Latent -> 2D Reconstruction")
    print("=" * 70)

    data, theta = generate_simple_data(n_samples=500)

    # Create and train model
    model = TinyAutoencoder()
    print(f"\nModel architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")

    # Train
    losses = train_autoencoder(model, data, epochs=500, lr=0.01)

    # Visualize
    visualize_autoencoder_results(model, data, theta, losses)

    # Demonstrate interpolation
    demonstrate_latent_interpolation(model, data)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key takeaways:
  1. Autoencoders learn to compress and reconstruct data
  2. The bottleneck forces learning of essential features
  3. Latent space captures the underlying structure (1D angle for circle)
  4. Interpolation in latent space produces smooth transitions
  5. Reconstruction loss (MSE) measures quality of compression

Next: Apply autoencoders to MNIST digit images!
""")


if __name__ == "__main__":
    demonstrate_autoencoder_basics()
