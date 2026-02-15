"""
Topic 4: Latent Space Visualization
====================================

This module focuses on:
1. Understanding and visualizing latent spaces
2. Comparing AE vs VAE latent spaces
3. Interactive latent space exploration
4. Latent arithmetic (adding/subtracting features)
5. t-SNE and PCA visualization of higher-dimensional latent spaces
6. Manifold traversal and interpolation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import ssl
import os

# Fix SSL certificate issues for downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context

# Create results directory if it doesn't exist
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_mnist_data(batch_size=128):
    """Load MNIST dataset."""
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class Autoencoder(nn.Module):
    """Standard autoencoder for comparison."""

    def __init__(self, latent_dim=2):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class VAE(nn.Module):
    """Variational autoencoder."""

    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z


def train_autoencoder(model, train_loader, epochs=20, device='cpu'):
    """Train standard autoencoder."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            recon, z = model(images)
            loss = criterion(recon, images.view(-1, 784))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"AE Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    return model


def train_vae(model, train_loader, epochs=20, device='cpu'):
    """Train VAE."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            recon, mu, log_var, z = model(images)

            # Reconstruction loss
            recon_loss = F.binary_cross_entropy(recon, images.view(-1, 784), reduction='sum')
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"VAE Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader.dataset):.4f}")

    return model


def compare_latent_spaces(ae_model, vae_model, test_loader, device='cpu'):
    """
    Compare latent spaces of AE and VAE side by side.
    """
    print("\n" + "=" * 70)
    print("Comparing AE vs VAE Latent Spaces")
    print("=" * 70)

    ae_model.eval()
    vae_model.eval()

    ae_z_list, vae_mu_list, labels_list = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            _, ae_z = ae_model(images)
            mu, log_var = vae_model.encode(images)

            ae_z_list.append(ae_z.cpu())
            vae_mu_list.append(mu.cpu())
            labels_list.append(labels)

    ae_z = torch.cat(ae_z_list).numpy()
    vae_mu = torch.cat(vae_mu_list).numpy()
    labels = torch.cat(labels_list).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # AE latent space
    scatter1 = axes[0].scatter(ae_z[:, 0], ae_z[:, 1], c=labels, cmap='tab10',
                                alpha=0.5, s=5)
    axes[0].set_xlabel('z₁', fontsize=12)
    axes[0].set_ylabel('z₂', fontsize=12)
    axes[0].set_title('Standard Autoencoder Latent Space', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # VAE latent space
    scatter2 = axes[1].scatter(vae_mu[:, 0], vae_mu[:, 1], c=labels, cmap='tab10',
                                alpha=0.5, s=5)
    axes[1].set_xlabel('μ₁', fontsize=12)
    axes[1].set_ylabel('μ₂', fontsize=12)
    axes[1].set_title('VAE Latent Space (μ)', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    # Add colorbars
    plt.colorbar(scatter1, ax=axes[0], label='Digit')
    plt.colorbar(scatter2, ax=axes[1], label='Digit')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "latent_space_comparison.png"), dpi=150, bbox_inches="tight")
    print("\nComparison saved to 'latent_space_comparison.png'")

    print("\nObservations:")
    print("  - AE: Latent space can have irregular structure, gaps")
    print("  - VAE: Latent space is more regular, centered around origin")
    print("  - VAE enforces N(0,1) prior, making it suitable for generation")


def visualize_latent_with_images(model, test_loader, device='cpu', n_images=100):
    """
    Visualize latent space with actual digit images embedded.
    """
    print("\n" + "=" * 70)
    print("Latent Space with Embedded Images")
    print("=" * 70)

    model.eval()

    images_list, z_list, labels_list = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            if isinstance(model, VAE):
                mu, _ = model.encode(images)
                z = mu
            else:
                _, z = model(images)

            images_list.append(images.cpu())
            z_list.append(z.cpu())
            labels_list.append(labels)

            if len(torch.cat(z_list)) >= n_images:
                break

    all_images = torch.cat(images_list)[:n_images]
    all_z = torch.cat(z_list)[:n_images].numpy()
    all_labels = torch.cat(labels_list)[:n_images].numpy()

    fig, ax = plt.subplots(figsize=(14, 12))

    # First, scatter plot for context
    scatter = ax.scatter(all_z[:, 0], all_z[:, 1], c=all_labels, cmap='tab10',
                         alpha=0.3, s=50)

    # Then, embed actual images
    for i in range(0, n_images, 5):  # Every 5th image to avoid clutter
        img = all_images[i].squeeze().numpy()
        imagebox = OffsetImage(img, zoom=0.7, cmap='gray')
        ab = AnnotationBbox(imagebox, (all_z[i, 0], all_z[i, 1]),
                           frameon=True, pad=0.1)
        ax.add_artist(ab)

    ax.set_xlabel('Latent dimension 1', fontsize=12)
    ax.set_ylabel('Latent dimension 2', fontsize=12)
    ax.set_title('Latent Space with Embedded MNIST Images', fontsize=14)
    plt.colorbar(scatter, label='Digit')

    plt.savefig(os.path.join(RESULTS_DIR, "latent_space_with_images.png"), dpi=150, bbox_inches="tight")
    print("\nVisualization saved to 'latent_space_with_images.png'")


def visualize_decoder_manifold(model, device='cpu', grid_size=20, z_range=3):
    """
    Visualize the decoder manifold by sampling uniformly from latent space.
    Creates a comprehensive view of what the decoder generates.
    """
    print("\n" + "=" * 70)
    print("Decoder Manifold Visualization")
    print("=" * 70)

    model.eval()

    # Create a grid in latent space
    z1 = torch.linspace(-z_range, z_range, grid_size)
    z2 = torch.linspace(-z_range, z_range, grid_size)

    # Create full image canvas
    canvas = np.zeros((28 * grid_size, 28 * grid_size))

    with torch.no_grad():
        for i, z2_val in enumerate(reversed(z2)):
            for j, z1_val in enumerate(z1):
                z = torch.tensor([[z1_val, z2_val]]).to(device)
                if isinstance(model, VAE):
                    img = model.decode(z).view(28, 28)
                else:
                    img = model.decode(z).view(28, 28)
                canvas[i*28:(i+1)*28, j*28:(j+1)*28] = img.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(canvas, cmap='gray')
    ax.set_xlabel(f'z₁ (from {-z_range} to {z_range})', fontsize=12)
    ax.set_ylabel(f'z₂ (from {z_range} to {-z_range})', fontsize=12)
    ax.set_title('Decoder Manifold: Generated Images from Latent Space', fontsize=14)

    # Set tick labels
    tick_positions = np.linspace(0, 28*grid_size, 5)
    tick_labels = np.linspace(-z_range, z_range, 5)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f'{x:.1f}' for x in tick_labels])
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f'{x:.1f}' for x in reversed(tick_labels)])

    plt.savefig(os.path.join(RESULTS_DIR, "decoder_manifold.png"), dpi=150, bbox_inches="tight")
    print("\nManifold saved to 'decoder_manifold.png'")

    print("\nObservations:")
    print("  - Each position in the grid corresponds to a latent vector z")
    print("  - The decoder maps each z to a digit image")
    print("  - Smooth transitions indicate a well-learned manifold")
    print("  - Different regions generate different digit types")


def latent_arithmetic(model, test_loader, device='cpu'):
    """
    Demonstrate latent arithmetic: adding/subtracting latent vectors.
    Example: average(1s) - average(0s) + specific_0 = 1-like digit
    """
    print("\n" + "=" * 70)
    print("Latent Arithmetic")
    print("=" * 70)

    model.eval()

    # Collect latent vectors by digit
    digit_latents = {i: [] for i in range(10)}
    digit_images = {i: [] for i in range(10)}

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            if isinstance(model, VAE):
                mu, _ = model.encode(images)
                z = mu
            else:
                _, z = model(images)

            for i in range(len(labels)):
                digit = labels[i].item()
                digit_latents[digit].append(z[i].cpu())
                digit_images[digit].append(images[i].cpu())

    # Compute average latent for each digit
    avg_latents = {}
    for digit in range(10):
        avg_latents[digit] = torch.stack(digit_latents[digit]).mean(dim=0)

    print("\nAverage latent vectors computed for each digit (0-9)")

    # Demonstrate arithmetic: digit_a + (avg_b - avg_a) should look like digit_b
    operations = [
        (0, 1, "0 + (avg_1 - avg_0) → should look like 1"),
        (3, 8, "3 + (avg_8 - avg_3) → should look like 8"),
        (4, 9, "4 + (avg_9 - avg_4) → should look like 9"),
        (7, 1, "7 + (avg_1 - avg_7) → should look like 1"),
    ]

    fig, axes = plt.subplots(len(operations), 5, figsize=(12, 10))

    for row, (digit_a, digit_b, title) in enumerate(operations):
        # Get a sample of digit_a
        sample_z = digit_latents[digit_a][0].unsqueeze(0).to(device)
        sample_img = digit_images[digit_a][0]

        # Compute transformation vector
        transform = (avg_latents[digit_b] - avg_latents[digit_a]).unsqueeze(0).to(device)

        # Apply transformation
        new_z = sample_z + transform

        with torch.no_grad():
            if isinstance(model, VAE):
                new_img = model.decode(new_z).view(28, 28)
            else:
                new_img = model.decode(new_z).view(28, 28)

        # Also show average digit images
        with torch.no_grad():
            avg_a_img = model.decode(avg_latents[digit_a].unsqueeze(0).to(device)).view(28, 28)
            avg_b_img = model.decode(avg_latents[digit_b].unsqueeze(0).to(device)).view(28, 28)

        # Plot
        axes[row, 0].imshow(sample_img.squeeze(), cmap='gray')
        axes[row, 0].set_title(f'Input ({digit_a})', fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(avg_a_img.cpu().squeeze(), cmap='gray')
        axes[row, 1].set_title(f'Avg {digit_a}', fontsize=10)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(avg_b_img.cpu().squeeze(), cmap='gray')
        axes[row, 2].set_title(f'Avg {digit_b}', fontsize=10)
        axes[row, 2].axis('off')

        axes[row, 3].imshow(new_img.cpu().squeeze(), cmap='gray')
        axes[row, 3].set_title('Result', fontsize=10)
        axes[row, 3].axis('off')

        # Show expected target
        axes[row, 4].imshow(digit_images[digit_b][0].squeeze(), cmap='gray')
        axes[row, 4].set_title(f'Target ({digit_b})', fontsize=10)
        axes[row, 4].axis('off')

        axes[row, 0].set_ylabel(title, fontsize=9)

    plt.suptitle('Latent Arithmetic: Input + (Avg_target - Avg_source) = Result', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "latent_arithmetic.png"), dpi=150, bbox_inches="tight")
    print("\nLatent arithmetic saved to 'latent_arithmetic.png'")

    print("\nKey insight:")
    print("  - Latent vectors capture semantic features")
    print("  - Vector differences encode transformations")
    print("  - Adding transformation vectors changes features!")


def visualize_high_dim_latent(model, test_loader, device='cpu', latent_dim=32):
    """
    For higher-dimensional latent spaces, use t-SNE or PCA to visualize.
    """
    print("\n" + "=" * 70)
    print(f"Visualizing High-Dimensional Latent Space (dim={latent_dim})")
    print("=" * 70)

    # Train a new model with higher latent dimension
    if isinstance(model, VAE):
        model = VAE(latent_dim=latent_dim).to(device)
    else:
        model = Autoencoder(latent_dim=latent_dim).to(device)

    train_loader, _ = load_mnist_data(batch_size=128)

    print("\nTraining model with higher latent dimension...")
    if isinstance(model, VAE):
        model = train_vae(model, train_loader, epochs=10, device=device)
    else:
        model = train_autoencoder(model, train_loader, epochs=10, device=device)

    # Collect latent vectors
    model.eval()
    z_list, labels_list = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            if isinstance(model, VAE):
                mu, _ = model.encode(images)
                z = mu
            else:
                _, z = model(images)

            z_list.append(z.cpu())
            labels_list.append(labels)

    all_z = torch.cat(z_list).numpy()
    all_labels = torch.cat(labels_list).numpy()

    print(f"\nLatent vectors shape: {all_z.shape}")

    # Apply t-SNE and PCA
    print("\nApplying PCA...")
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(all_z)

    print("Applying t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_tsne = tsne.fit_transform(all_z[:5000])  # Use subset for speed
    labels_tsne = all_labels[:5000]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # PCA
    scatter1 = axes[0].scatter(z_pca[:, 0], z_pca[:, 1], c=all_labels,
                                cmap='tab10', alpha=0.5, s=5)
    axes[0].set_xlabel('PC1', fontsize=12)
    axes[0].set_ylabel('PC2', fontsize=12)
    axes[0].set_title(f'PCA of {latent_dim}D Latent Space', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Digit')

    # t-SNE
    scatter2 = axes[1].scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels_tsne,
                                cmap='tab10', alpha=0.5, s=5)
    axes[1].set_xlabel('t-SNE 1', fontsize=12)
    axes[1].set_ylabel('t-SNE 2', fontsize=12)
    axes[1].set_title(f't-SNE of {latent_dim}D Latent Space', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Digit')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "high_dim_latent_visualization.png"), dpi=150, bbox_inches="tight")
    print("\nVisualization saved to 'high_dim_latent_visualization.png'")

    print("\nObservations:")
    print("  - t-SNE better preserves local structure (clusters)")
    print("  - PCA captures global variance directions")
    print("  - Higher-dim latent spaces can have better reconstruction")


def manifold_traversal(model, test_loader, device='cpu'):
    """
    Traverse along specific directions in latent space to understand
    what each dimension represents.
    """
    print("\n" + "=" * 70)
    print("Latent Space Traversal")
    print("=" * 70)

    model.eval()

    # Get a sample image
    images, labels = next(iter(test_loader))
    sample_img = images[0:1].to(device)

    with torch.no_grad():
        if isinstance(model, VAE):
            mu, _ = model.encode(sample_img)
            base_z = mu
        else:
            _, base_z = model(sample_img)

    # Traverse each dimension
    n_steps = 11
    z_range = 3

    fig, axes = plt.subplots(2, n_steps, figsize=(15, 4))

    for dim in range(2):  # Only 2D latent space
        traversal_values = torch.linspace(-z_range, z_range, n_steps)

        for i, val in enumerate(traversal_values):
            z = base_z.clone()
            z[0, dim] = val

            with torch.no_grad():
                if isinstance(model, VAE):
                    img = model.decode(z).view(28, 28)
                else:
                    img = model.decode(z).view(28, 28)

            axes[dim, i].imshow(img.cpu().numpy(), cmap='gray')
            axes[dim, i].axis('off')
            if dim == 0:
                axes[dim, i].set_title(f'{val:.1f}', fontsize=9)

        axes[dim, 0].set_ylabel(f'z{dim+1}', fontsize=12, rotation=0, labelpad=20)

    plt.suptitle('Latent Dimension Traversal\n(Each row varies one dimension while holding the other fixed)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "latent_traversal.png"), dpi=150, bbox_inches="tight")
    print("\nTraversal saved to 'latent_traversal.png'")

    print("\nObservation:")
    print("  - Each latent dimension captures different features")
    print("  - Moving along a dimension changes specific aspects of the digit")


def uncertainty_visualization(model, test_loader, device='cpu'):
    """
    For VAE: visualize the uncertainty (variance) in the latent space.
    """
    print("\n" + "=" * 70)
    print("VAE Uncertainty Visualization")
    print("=" * 70)

    if not isinstance(model, VAE):
        print("This visualization is only for VAE models.")
        return

    model.eval()

    mu_list, log_var_list, labels_list = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            mu, log_var = model.encode(images)

            mu_list.append(mu.cpu())
            log_var_list.append(log_var.cpu())
            labels_list.append(labels)

    all_mu = torch.cat(mu_list).numpy()
    all_log_var = torch.cat(log_var_list).numpy()
    all_std = np.exp(0.5 * all_log_var)  # Convert to standard deviation
    all_labels = torch.cat(labels_list).numpy()

    # Average uncertainty per dimension
    avg_std = np.mean(all_std, axis=0)
    print(f"\nAverage uncertainty (std) per dimension: {avg_std}")

    # Visualization: show mu with uncertainty (std as size)
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use average std as point size
    point_sizes = (all_std[:, 0] + all_std[:, 1]) / 2 * 100

    scatter = ax.scatter(all_mu[:, 0], all_mu[:, 1], c=all_labels, cmap='tab10',
                         alpha=0.4, s=point_sizes)

    ax.set_xlabel('μ₁', fontsize=12)
    ax.set_ylabel('μ₂', fontsize=12)
    ax.set_title('VAE Latent Space with Uncertainty\n(Point size indicates uncertainty/variance)',
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Digit')

    plt.savefig(os.path.join(RESULTS_DIR, "vae_uncertainty.png"), dpi=150, bbox_inches="tight")
    print("\nUncertainty visualization saved to 'vae_uncertainty.png'")


def demonstrate_latent_visualization():
    """
    Main demonstration of latent space visualization techniques.
    """
    print("\n" + "=" * 70)
    print("Latent Space Visualization - Comprehensive Demo")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load data
    train_loader, test_loader = load_mnist_data(batch_size=128)

    # Train both AE and VAE
    print("\n" + "-" * 70)
    print("Training Standard Autoencoder (2D latent)")
    print("-" * 70)
    ae_model = Autoencoder(latent_dim=2)
    ae_model = train_autoencoder(ae_model, train_loader, epochs=20, device=device)

    print("\n" + "-" * 70)
    print("Training VAE (2D latent)")
    print("-" * 70)
    vae_model = VAE(latent_dim=2)
    vae_model = train_vae(vae_model, train_loader, epochs=20, device=device)

    # Visualizations
    compare_latent_spaces(ae_model, vae_model, test_loader, device=device)
    visualize_latent_with_images(vae_model, test_loader, device=device, n_images=100)
    visualize_decoder_manifold(vae_model, device=device, grid_size=20, z_range=3)
    latent_arithmetic(vae_model, test_loader, device=device)
    manifold_traversal(vae_model, test_loader, device=device)
    uncertainty_visualization(vae_model, test_loader, device=device)
    visualize_high_dim_latent(VAE(latent_dim=32), test_loader, device=device, latent_dim=32)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key takeaways:

1. LATENT SPACE STRUCTURE
   - AE: Can have irregular structure, gaps
   - VAE: Regular, continuous, centered at origin

2. VISUALIZATION TECHNIQUES
   - 2D latent: Direct scatter plot
   - High-dim: Use t-SNE or PCA

3. DECODER MANIFOLD
   - Shows what different latent regions generate
   - Smooth transitions = good manifold learning

4. LATENT ARITHMETIC
   - Vector differences encode transformations
   - Add/subtract vectors to change features

5. DIMENSION TRAVERSAL
   - Each dimension captures different features
   - Can discover interpretable directions

6. UNCERTAINTY (VAE only)
   - Variance indicates model's confidence
   - Higher variance = more uncertain encoding

Generated files:
  - latent_space_comparison.png: AE vs VAE comparison
  - latent_space_with_images.png: Embedded digit images
  - decoder_manifold.png: Full grid of decoded images
  - latent_arithmetic.png: Feature arithmetic examples
  - latent_traversal.png: Traversing each dimension
  - vae_uncertainty.png: Uncertainty visualization
  - high_dim_latent_visualization.png: t-SNE/PCA for high-dim
""")


if __name__ == "__main__":
    demonstrate_latent_visualization()
