"""
Topic 2: Autoencoder on MNIST
=============================

This module demonstrates:
1. Building a deeper autoencoder for images
2. Training on MNIST digit dataset
3. Visualizing reconstructions
4. Exploring the latent space with 2D embeddings
5. Understanding what the autoencoder learns
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


def load_mnist_data(batch_size=128):
    """
    Load MNIST dataset for autoencoder training.

    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    print("\n" + "=" * 70)
    print("Loading MNIST Dataset")
    print("=" * 70)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] range
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

    print(f"\nDataset info:")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")
    print(f"  - Image shape: {train_dataset[0][0].shape} (C, H, W)")
    print(f"  - Pixel range: [0, 1]")
    print(f"  - Batch size: {batch_size}")

    return train_loader, test_loader


class MNISTAutoencoder(nn.Module):
    """
    Autoencoder for MNIST images (28x28 = 784 pixels).

    Architecture:
    - Encoder: 784 -> 256 -> 128 -> latent_dim
    - Decoder: latent_dim -> 128 -> 256 -> 784

    The latent_dim controls the compression ratio.
    """

    def __init__(self, latent_dim=32):
        super(MNISTAutoencoder, self).__init__()

        self.latent_dim = latent_dim

        # Encoder: compress 784 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Decoder: expand latent_dim -> 784
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()  # Output in [0, 1] for pixel values
        )

    def encode(self, x):
        """Encode image to latent representation."""
        # Flatten image: (B, 1, 28, 28) -> (B, 784)
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, z):
        """Decode latent representation to image."""
        # Output is (B, 784), needs reshaping to view as image
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass."""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for MNIST.

    Uses convolutional layers for better feature extraction.
    Architecture preserves spatial structure through encoding.
    """

    def __init__(self, latent_dim=32):
        super(ConvAutoencoder, self).__init__()

        self.latent_dim = latent_dim

        # Encoder: (1, 28, 28) -> (32, 7, 7) -> latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> (16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 7, 7)
            nn.ReLU(),
        )
        self.encoder_fc = nn.Linear(32 * 7 * 7, latent_dim)

        # Decoder: latent_dim -> (32, 7, 7) -> (1, 28, 28)
        self.decoder_fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (1, 28, 28)
            nn.Sigmoid()
        )

    def encode(self, x):
        """Encode image to latent representation."""
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        z = self.encoder_fc(x)
        return z

    def decode(self, z):
        """Decode latent representation to image."""
        x = self.decoder_fc(z)
        x = x.view(-1, 32, 7, 7)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        """Full forward pass."""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z


def train_autoencoder(model, train_loader, epochs=10, lr=1e-3, device='cpu'):
    """
    Train the autoencoder on MNIST.

    Args:
        model: Autoencoder model
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on

    Returns:
        List of average losses per epoch
    """
    print("\n" + "=" * 70)
    print("Training Autoencoder")
    print("=" * 70)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    print(f"\nTraining for {epochs} epochs on {device}...")
    print("-" * 70)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)

            # Forward pass
            if isinstance(model, ConvAutoencoder):
                reconstruction, z = model(images)
                target = images
            else:
                reconstruction, z = model(images)
                target = images.view(images.size(0), -1)

            loss = criterion(reconstruction, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")

    print("-" * 70)
    print(f"Final Loss: {losses[-1]:.6f}")

    return losses


def visualize_reconstructions(model, test_loader, n_samples=10, device='cpu'):
    """
    Visualize original images and their reconstructions.
    """
    print("\n" + "=" * 70)
    print("Visualizing Reconstructions")
    print("=" * 70)

    model.eval()
    model = model.to(device)

    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images = images[:n_samples].to(device)
    labels = labels[:n_samples]

    with torch.no_grad():
        if isinstance(model, ConvAutoencoder):
            reconstructions, _ = model(images)
        else:
            reconstructions, _ = model(images)
            reconstructions = reconstructions.view(-1, 1, 28, 28)

    # Plot
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 3))

    for i in range(n_samples):
        # Original
        axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        # Reconstruction
        axes[1, i].imshow(reconstructions[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstruction', fontsize=10)

    plt.suptitle(f'MNIST Autoencoder Reconstructions (Latent dim={model.latent_dim})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "mnist_reconstructions.png"), dpi=150, bbox_inches="tight")
    print("\nReconstructions saved to 'mnist_reconstructions.png'")


def visualize_latent_space_2d(model, test_loader, device='cpu'):
    """
    Visualize the latent space by encoding test images.
    Only works well when latent_dim=2, otherwise uses first 2 dimensions.
    """
    print("\n" + "=" * 70)
    print("Visualizing Latent Space")
    print("=" * 70)

    model.eval()
    model = model.to(device)

    all_z = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            z = model.encode(images)
            all_z.append(z.cpu())
            all_labels.append(labels)

    all_z = torch.cat(all_z, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Use first 2 dimensions if latent_dim > 2
    if all_z.shape[1] > 2:
        print(f"\nLatent dim={all_z.shape[1]}, using first 2 dimensions for visualization")
        z_2d = all_z[:, :2]
    else:
        z_2d = all_z

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=all_labels, cmap='tab10',
                          alpha=0.5, s=5)
    plt.colorbar(scatter, label='Digit')
    plt.xlabel('Latent dimension 1', fontsize=12)
    plt.ylabel('Latent dimension 2', fontsize=12)
    plt.title(f'MNIST Latent Space Visualization\n(Latent dim={model.latent_dim})', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(RESULTS_DIR, "mnist_latent_space.png"), dpi=150, bbox_inches="tight")
    print("\nLatent space visualization saved to 'mnist_latent_space.png'")

    print("\nObservations:")
    print("  - Points with same color (digit) should cluster together")
    print("  - Similar digits (like 4 and 9) may be closer in latent space")
    print("  - The structure shows what the autoencoder has learned")


def visualize_latent_interpolation(model, test_loader, device='cpu'):
    """
    Interpolate between two digits in latent space.
    """
    print("\n" + "=" * 70)
    print("Latent Space Interpolation")
    print("=" * 70)

    model.eval()
    model = model.to(device)

    # Get two different digits
    images, labels = next(iter(test_loader))

    # Find indices of two different digits (e.g., 0 and 1)
    idx_0 = (labels == 0).nonzero(as_tuple=True)[0][0]
    idx_1 = (labels == 1).nonzero(as_tuple=True)[0][0]

    img_0 = images[idx_0:idx_0+1].to(device)
    img_1 = images[idx_1:idx_1+1].to(device)

    with torch.no_grad():
        z_0 = model.encode(img_0)
        z_1 = model.encode(img_1)

    # Interpolate
    n_steps = 10
    alphas = torch.linspace(0, 1, n_steps)

    interpolated_images = []
    for alpha in alphas:
        z_interp = (1 - alpha) * z_0 + alpha * z_1
        with torch.no_grad():
            if isinstance(model, ConvAutoencoder):
                img_interp = model.decode(z_interp)
            else:
                img_interp = model.decode(z_interp).view(-1, 1, 28, 28)
            interpolated_images.append(img_interp.cpu().squeeze())

    # Plot
    fig, axes = plt.subplots(1, n_steps, figsize=(15, 2))

    for i, img in enumerate(interpolated_images):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        if i == 0:
            axes[i].set_title(f'Digit 0', fontsize=10)
        elif i == n_steps - 1:
            axes[i].set_title(f'Digit 1', fontsize=10)

    plt.suptitle('Interpolation in Latent Space (0 → 1)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "mnist_interpolation.png"), dpi=150, bbox_inches="tight")
    print("\nInterpolation saved to 'mnist_interpolation.png'")

    print("\nObservation:")
    print("  - Smooth transition between digits shows the latent space is continuous")
    print("  - Intermediate images may look like blends of both digits")


def generate_from_latent_space(model, device='cpu'):
    """
    Generate new images by sampling from the latent space.
    """
    print("\n" + "=" * 70)
    print("Generating from Latent Space")
    print("=" * 70)

    model.eval()
    model = model.to(device)

    # Sample random points in latent space
    # Note: For regular autoencoders, the latent space distribution is not known,
    # so we sample from a reasonable range based on typical values
    n_samples = 20
    z_samples = torch.randn(n_samples, model.latent_dim).to(device) * 2

    with torch.no_grad():
        if isinstance(model, ConvAutoencoder):
            generated = model.decode(z_samples)
        else:
            generated = model.decode(z_samples).view(-1, 1, 28, 28)

    # Plot
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    axes = axes.flatten()

    for i in range(n_samples):
        axes[i].imshow(generated[i].cpu().squeeze(), cmap='gray')
        axes[i].axis('off')

    plt.suptitle('Images Generated by Sampling Latent Space\n(Random samples may not look like digits)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "mnist_generated.png"), dpi=150, bbox_inches="tight")
    print("\nGenerated images saved to 'mnist_generated.png'")

    print("\nNote:")
    print("  - Regular autoencoders don't have a structured latent space")
    print("  - Random samples may not produce realistic digits")
    print("  - This is why we need VAEs for proper generation!")


def compare_latent_dimensions():
    """
    Compare autoencoders with different latent dimensions.
    """
    print("\n" + "=" * 70)
    print("Comparing Different Latent Dimensions")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = load_mnist_data(batch_size=128)

    latent_dims = [2, 8, 32]
    models = {}
    losses_dict = {}

    for latent_dim in latent_dims:
        print(f"\n--- Training with latent_dim={latent_dim} ---")
        model = MNISTAutoencoder(latent_dim=latent_dim)
        losses = train_autoencoder(model, train_loader, epochs=5, lr=1e-3, device=device)
        models[latent_dim] = model
        losses_dict[latent_dim] = losses

    # Compare reconstructions
    fig, axes = plt.subplots(len(latent_dims) + 1, 10, figsize=(15, 6))

    images, _ = next(iter(test_loader))
    images = images[:10].to(device)

    # Original images
    for i in range(10):
        axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=10)

    # Reconstructions for each latent dim
    for row, latent_dim in enumerate(latent_dims, 1):
        model = models[latent_dim].to(device)
        model.eval()
        with torch.no_grad():
            recon, _ = model(images)
            recon = recon.view(-1, 1, 28, 28)

        for i in range(10):
            axes[row, i].imshow(recon[i].cpu().squeeze(), cmap='gray')
            axes[row, i].axis('off')
            if i == 0:
                axes[row, i].set_ylabel(f'z={latent_dim}', fontsize=10)

    plt.suptitle('Effect of Latent Dimension on Reconstruction Quality', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "mnist_latent_dim_comparison.png"), dpi=150, bbox_inches="tight")
    print("\nComparison saved to 'mnist_latent_dim_comparison.png'")

    print("\nObservation:")
    print("  - Larger latent dimension = better reconstruction (less compression)")
    print("  - Smaller latent dimension = more compression, blurrier results")
    print("  - Trade-off between compression and quality")


def demonstrate_mnist_autoencoder():
    """
    Main demonstration of MNIST autoencoder.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load data
    train_loader, test_loader = load_mnist_data(batch_size=128)

    # Create model with 2D latent space for visualization
    print("\n" + "=" * 70)
    print("Creating Autoencoder with 2D Latent Space")
    print("=" * 70)

    model = MNISTAutoencoder(latent_dim=2)
    print(f"\nModel architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Train
    losses = train_autoencoder(model, train_loader, epochs=20, lr=1e-3, device=device)

    # Visualize results
    visualize_reconstructions(model, test_loader, n_samples=10, device=device)
    visualize_latent_space_2d(model, test_loader, device=device)
    visualize_latent_interpolation(model, test_loader, device=device)
    generate_from_latent_space(model, device=device)

    # Compare different latent dimensions
    compare_latent_dimensions()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key takeaways:
  1. Autoencoders can compress MNIST images (784 -> latent_dim)
  2. 2D latent space allows visualization of digit clusters
  3. Similar digits cluster together in latent space
  4. Interpolation produces smooth transitions between digits
  5. Regular autoencoders have unstructured latent space
     -> Random sampling doesn't produce good results
     -> This motivates Variational Autoencoders (VAE)

Generated files:
  - mnist_reconstructions.png: Original vs reconstructed images
  - mnist_latent_space.png: 2D visualization of latent embeddings
  - mnist_interpolation.png: Interpolation between digits
  - mnist_generated.png: Randomly sampled images
  - mnist_latent_dim_comparison.png: Effect of latent dimension
""")


if __name__ == "__main__":
    demonstrate_mnist_autoencoder()
