"""
Topic 3: Variational Autoencoder (VAE)
======================================

This module demonstrates:
1. The problem with regular autoencoders for generation
2. VAE theory: probabilistic encoding and the reparameterization trick
3. The VAE loss function: reconstruction + KL divergence
4. Training a VAE on MNIST
5. Why VAE produces better generative results
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import ssl
import os

# Fix SSL certificate issues for downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context

# Create results directory if it doesn't exist
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def explain_vae_motivation():
    """
    Explain why we need VAEs instead of regular autoencoders.
    """
    print("\n" + "=" * 70)
    print("Why Variational Autoencoders?")
    print("=" * 70)

    print("""
Problem with Regular Autoencoders:
----------------------------------
Regular autoencoders learn to map inputs to specific points in latent space.
This creates several issues:

1. GAPS IN LATENT SPACE
   - Not all points in latent space decode to valid images
   - Sampling random z may produce garbage

2. NO STRUCTURE
   - Latent space can have irregular structure
   - Distance in latent space doesn't mean similarity

3. NOT PROBABILISTIC
   - No way to sample "new" realistic data points
   - Only reconstructs existing data

Solution: Variational Autoencoder
---------------------------------
Instead of encoding to a point, encode to a DISTRIBUTION!

    Regular AE:    x -> z (single point)
    VAE:           x -> N(μ, σ²) -> sample z -> decode

Key Ideas:
1. Encoder outputs mean (μ) and variance (σ²) of a Gaussian
2. Sample z from this Gaussian (reparameterization trick)
3. Regularize the latent space to be close to N(0, 1)
4. This fills the latent space with valid decodings!
""")


def explain_vae_architecture():
    """
    Explain VAE architecture with diagrams.
    """
    print("\n" + "=" * 70)
    print("VAE Architecture")
    print("=" * 70)

    print("""
                    Encoder                    Decoder
                 (Recognition)              (Generative)
                      |                          |
    Input x ──────────┼──────────────────────────┼──────> Output x̂
      │               │                          │
      │       ┌───────┴───────┐                  │
      └──────>│   Neural Net  │                  │
              │   (encoder)   │                  │
              └───────┬───────┘                  │
                      │                          │
              ┌───────┴───────┐                  │
              │               │                  │
              ▼               ▼                  │
            [μ]             [σ²]                 │
              │               │                  │
              └───────┬───────┘                  │
                      │                          │
                      ▼                          │
              ┌─────────────┐                    │
              │  z ~ N(μ,σ²) │  <── Reparameterization
              │             │      z = μ + σ·ε
              │  (sampling) │      ε ~ N(0,1)
              └──────┬──────┘
                     │
                     └──────────────────────────>│
                                         ┌───────┴───────┐
                                         │   Neural Net  │
                                         │   (decoder)   │
                                         └───────────────┘

Reparameterization Trick:
-------------------------
Problem: Sampling z ~ N(μ, σ²) is not differentiable!
Solution: z = μ + σ · ε, where ε ~ N(0, 1)

This separates the randomness (ε) from the learnable parameters (μ, σ).
Gradients flow through μ and σ, not through ε.
""")


def explain_vae_loss():
    """
    Explain the VAE loss function.
    """
    print("\n" + "=" * 70)
    print("VAE Loss Function")
    print("=" * 70)

    print("""
The VAE loss has two components:

Loss = Reconstruction Loss + β · KL Divergence

1. RECONSTRUCTION LOSS (same as regular autoencoder)
   - Measures how well we can reconstruct the input
   - L_recon = ||x - x̂||² (MSE) or BCE for binary images
   - Encourages the model to encode useful information

2. KL DIVERGENCE (regularization term)
   - Measures how different q(z|x) is from p(z) = N(0, 1)
   - KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
   - Encourages the latent space to be "regular"

Why KL Divergence?
------------------
- Pushes encoded distributions toward N(0, 1)
- Ensures latent space is continuous (no gaps)
- Allows meaningful sampling from N(0, 1)
- Prevents mode collapse (encoding everything to one point)

The β (beta) parameter controls the trade-off:
- β = 1: Standard VAE (ELBO loss)
- β > 1: β-VAE (more regularization, smoother latent space)
- β < 1: Less regularization, better reconstruction
""")


def load_mnist_data(batch_size=128):
    """
    Load MNIST dataset for VAE training.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class VAE(nn.Module):
    """
    Variational Autoencoder for MNIST.

    Architecture:
    - Encoder: 784 -> 256 -> 128 -> (μ, log_var) of dimension latent_dim
    - Decoder: latent_dim -> 128 -> 256 -> 784

    Key difference from regular autoencoder:
    - Encoder outputs TWO vectors: mean (μ) and log variance (log_var)
    - We sample z from N(μ, exp(log_var)) using reparameterization trick
    """

    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        # Encoder: outputs mean and log variance
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)      # Mean
        self.fc_log_var = nn.Linear(128, latent_dim)  # Log variance

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Encode input to mean and log variance.

        Args:
            x: Input images (B, 1, 28, 28) or (B, 784)

        Returns:
            mu: Mean of latent distribution (B, latent_dim)
            log_var: Log variance of latent distribution (B, latent_dim)
        """
        x = x.view(x.size(0), -1)  # Flatten
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: sample z = mu + std * epsilon

        This allows gradients to flow through mu and log_var.

        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution

        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)  # Standard deviation
        epsilon = torch.randn_like(std)  # Random noise from N(0, 1)
        z = mu + std * epsilon
        return z

    def decode(self, z):
        """
        Decode latent vector to image.

        Args:
            z: Latent vector (B, latent_dim)

        Returns:
            Reconstructed image (B, 784)
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Full forward pass.

        Returns:
            reconstruction: Reconstructed image
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var, z


def vae_loss(reconstruction, x, mu, log_var, beta=1.0):
    """
    VAE loss function: Reconstruction + KL divergence.

    Args:
        reconstruction: Reconstructed image (B, 784)
        x: Original image (B, 1, 28, 28) or (B, 784)
        mu: Mean of latent distribution
        log_var: Log variance of latent distribution
        beta: Weight for KL divergence (β-VAE)

    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence loss
    """
    x = x.view(x.size(0), -1)  # Flatten

    # Reconstruction loss (BCE for binary images)
    recon_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')

    # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def train_vae(model, train_loader, epochs=20, lr=1e-3, beta=1.0, device='cpu'):
    """
    Train the VAE.

    Args:
        model: VAE model
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        lr: Learning rate
        beta: Weight for KL divergence
        device: Device to train on

    Returns:
        Dictionary of loss histories
    """
    print("\n" + "=" * 70)
    print(f"Training VAE (β={beta})")
    print("=" * 70)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'total': [], 'recon': [], 'kl': []}

    print(f"\nTraining for {epochs} epochs on {device}...")
    print("-" * 70)

    for epoch in range(epochs):
        model.train()
        epoch_total = 0
        epoch_recon = 0
        epoch_kl = 0

        for images, _ in train_loader:
            images = images.to(device)

            # Forward pass
            reconstruction, mu, log_var, z = model(images)
            total_loss, recon_loss, kl_loss = vae_loss(
                reconstruction, images, mu, log_var, beta
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

        # Average over dataset
        n_samples = len(train_loader.dataset)
        history['total'].append(epoch_total / n_samples)
        history['recon'].append(epoch_recon / n_samples)
        history['kl'].append(epoch_kl / n_samples)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Total: {history['total'][-1]:.2f} | "
                  f"Recon: {history['recon'][-1]:.2f} | "
                  f"KL: {history['kl'][-1]:.2f}")

    print("-" * 70)
    print(f"Final - Total: {history['total'][-1]:.2f} | "
          f"Recon: {history['recon'][-1]:.2f} | "
          f"KL: {history['kl'][-1]:.2f}")

    return history


def visualize_vae_training(history):
    """
    Visualize VAE training losses.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total loss
    axes[0].plot(history['total'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Total Loss', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[1].plot(history['recon'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Reconstruction Loss', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    # KL loss
    axes[2].plot(history['kl'], 'r-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Loss', fontsize=12)
    axes[2].set_title('KL Divergence', fontsize=14)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "vae_training.png"), dpi=150, bbox_inches="tight")
    print("\nTraining visualization saved to 'vae_training.png'")


def visualize_reconstructions(model, test_loader, n_samples=10, device='cpu'):
    """
    Visualize original images and their reconstructions.
    """
    model.eval()
    model = model.to(device)

    images, labels = next(iter(test_loader))
    images = images[:n_samples].to(device)

    with torch.no_grad():
        reconstructions, mu, log_var, z = model(images)
        reconstructions = reconstructions.view(-1, 1, 28, 28)

    fig, axes = plt.subplots(2, n_samples, figsize=(15, 3))

    for i in range(n_samples):
        axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        axes[1, i].imshow(reconstructions[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstruction', fontsize=10)

    plt.suptitle('VAE Reconstructions', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "vae_reconstructions.png"), dpi=150, bbox_inches="tight")
    print("\nReconstructions saved to 'vae_reconstructions.png'")


def visualize_latent_space(model, test_loader, device='cpu'):
    """
    Visualize the VAE latent space (2D).
    """
    print("\n" + "=" * 70)
    print("Visualizing VAE Latent Space")
    print("=" * 70)

    model.eval()
    model = model.to(device)

    all_mu = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            mu, log_var = model.encode(images)
            all_mu.append(mu.cpu())
            all_labels.append(labels)

    all_mu = torch.cat(all_mu, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_mu[:, 0], all_mu[:, 1], c=all_labels, cmap='tab10',
                          alpha=0.5, s=5)
    plt.colorbar(scatter, label='Digit')
    plt.xlabel('Latent dimension 1 (μ₁)', fontsize=12)
    plt.ylabel('Latent dimension 2 (μ₂)', fontsize=12)
    plt.title('VAE Latent Space (2D)\nColored by digit class', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(RESULTS_DIR, "vae_latent_space.png"), dpi=150, bbox_inches="tight")
    print("\nLatent space visualization saved to 'vae_latent_space.png'")

    print("\nKey observation:")
    print("  - VAE latent space is more regular than regular autoencoder")
    print("  - Distribution is approximately centered at origin")
    print("  - Digits form clusters but space is more continuous")


def generate_from_latent_space(model, device='cpu'):
    """
    Generate new images by sampling from the prior N(0, 1).
    """
    print("\n" + "=" * 70)
    print("Generating from Latent Space")
    print("=" * 70)

    model.eval()
    model = model.to(device)

    # Sample from prior N(0, 1)
    n_samples = 20
    z_samples = torch.randn(n_samples, model.latent_dim).to(device)

    with torch.no_grad():
        generated = model.decode(z_samples).view(-1, 1, 28, 28)

    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    axes = axes.flatten()

    for i in range(n_samples):
        axes[i].imshow(generated[i].cpu().squeeze(), cmap='gray')
        axes[i].axis('off')

    plt.suptitle('VAE: Images Generated by Sampling z ~ N(0, 1)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "vae_generated.png"), dpi=150, bbox_inches="tight")
    print("\nGenerated images saved to 'vae_generated.png'")

    print("\nKey observation:")
    print("  - VAE generates more realistic digits than regular autoencoder!")
    print("  - Sampling from N(0, 1) works because we regularized toward N(0, 1)")


def generate_grid_from_latent_space(model, device='cpu', grid_size=20):
    """
    Generate a grid of images by uniformly sampling the 2D latent space.
    This shows how the decoder maps different latent regions.
    """
    print("\n" + "=" * 70)
    print("Generating Grid from Latent Space")
    print("=" * 70)

    model.eval()
    model = model.to(device)

    # Create a grid of latent values
    # Sample from a range that covers most of the distribution
    z_range = 3  # -3 to 3 covers 99.7% of N(0,1)
    z1 = torch.linspace(-z_range, z_range, grid_size)
    z2 = torch.linspace(-z_range, z_range, grid_size)

    # Create all combinations
    grid_images = []

    with torch.no_grad():
        for z2_val in reversed(z1):  # Reversed so top = high z2
            row_images = []
            for z1_val in z2:
                z = torch.tensor([[z1_val, z2_val]]).to(device)
                img = model.decode(z).view(28, 28)
                row_images.append(img.cpu().numpy())
            grid_images.append(row_images)

    # Plot grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    for i in range(grid_size):
        for j in range(grid_size):
            axes[i, j].imshow(grid_images[i][j], cmap='gray')
            axes[i, j].axis('off')

    plt.suptitle('VAE Latent Space Grid\n(Uniformly sampling z₁ and z₂ from -3 to 3)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "vae_latent_grid.png"), dpi=150, bbox_inches="tight")
    print("\nLatent grid saved to 'vae_latent_grid.png'")

    print("\nKey observations:")
    print("  - Moving through latent space produces smooth transitions")
    print("  - Different regions correspond to different digit types")
    print("  - The decoder has learned a continuous mapping from z to images")


def interpolate_between_digits(model, test_loader, device='cpu'):
    """
    Interpolate between two digits in VAE latent space.
    """
    print("\n" + "=" * 70)
    print("Interpolating Between Digits")
    print("=" * 70)

    model.eval()
    model = model.to(device)

    images, labels = next(iter(test_loader))

    # Find different digit pairs to interpolate
    digit_pairs = [(0, 1), (2, 7), (3, 8), (4, 9)]

    fig, axes = plt.subplots(len(digit_pairs), 12, figsize=(15, 5))

    n_steps = 10

    for row, (d1, d2) in enumerate(digit_pairs):
        idx1 = (labels == d1).nonzero(as_tuple=True)[0][0]
        idx2 = (labels == d2).nonzero(as_tuple=True)[0][0]

        img1 = images[idx1:idx1+1].to(device)
        img2 = images[idx2:idx2+1].to(device)

        with torch.no_grad():
            mu1, _ = model.encode(img1)
            mu2, _ = model.encode(img2)

        # Show original digit 1
        axes[row, 0].imshow(images[idx1].squeeze(), cmap='gray')
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title('Start', fontsize=10)

        # Interpolate
        for i, alpha in enumerate(torch.linspace(0, 1, n_steps)):
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            with torch.no_grad():
                img_interp = model.decode(z_interp).view(28, 28)
            axes[row, i + 1].imshow(img_interp.cpu().numpy(), cmap='gray')
            axes[row, i + 1].axis('off')

        # Show original digit 2
        axes[row, 11].imshow(images[idx2].squeeze(), cmap='gray')
        axes[row, 11].axis('off')
        if row == 0:
            axes[row, 11].set_title('End', fontsize=10)

        axes[row, 0].set_ylabel(f'{d1}→{d2}', fontsize=10, rotation=0, labelpad=20)

    plt.suptitle('VAE Latent Space Interpolation', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "vae_interpolation.png"), dpi=150, bbox_inches="tight")
    print("\nInterpolation saved to 'vae_interpolation.png'")


def demonstrate_vae():
    """
    Main demonstration of Variational Autoencoder.
    """
    # Explain concepts
    explain_vae_motivation()
    explain_vae_architecture()
    explain_vae_loss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load data
    print("\n" + "=" * 70)
    print("Loading MNIST Dataset")
    print("=" * 70)
    train_loader, test_loader = load_mnist_data(batch_size=128)

    # Create model
    print("\n" + "=" * 70)
    print("Creating VAE with 2D Latent Space")
    print("=" * 70)

    model = VAE(latent_dim=2)
    print(f"\nModel architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Train
    history = train_vae(model, train_loader, epochs=30, lr=1e-3, beta=1.0, device=device)

    # Visualize
    visualize_vae_training(history)
    visualize_reconstructions(model, test_loader, n_samples=10, device=device)
    visualize_latent_space(model, test_loader, device=device)
    generate_from_latent_space(model, device=device)
    generate_grid_from_latent_space(model, device=device, grid_size=15)
    interpolate_between_digits(model, test_loader, device=device)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key takeaways:
  1. VAE encodes to a distribution (μ, σ²), not a point
  2. Reparameterization trick allows gradient flow: z = μ + σ·ε
  3. KL divergence regularizes latent space toward N(0, 1)
  4. This enables meaningful sampling from the prior
  5. Generated samples are realistic digits!
  6. Latent space is continuous and smooth

VAE vs Regular Autoencoder:
  - VAE has structured, regular latent space
  - VAE allows sampling new realistic data
  - Trade-off: slightly worse reconstruction for better generation

Generated files:
  - vae_training.png: Training loss curves
  - vae_reconstructions.png: Original vs reconstructed
  - vae_latent_space.png: 2D latent visualization
  - vae_generated.png: Random samples from N(0,1)
  - vae_latent_grid.png: Grid sampling of latent space
  - vae_interpolation.png: Interpolation between digits
""")


if __name__ == "__main__":
    demonstrate_vae()
