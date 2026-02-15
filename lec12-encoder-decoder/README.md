# Lecture 12: Encoder-Decoder Architectures

This lecture covers autoencoders and variational autoencoders (VAE), with a focus on understanding and visualizing latent spaces.

## Topics

### 1. Autoencoder Basics (`01_autoencoder_basics.py`)
- What is an autoencoder and why it's useful
- Encoder-decoder architecture
- Bottleneck/latent representation
- Simple 2D example: compressing a circle to 1D

### 2. Autoencoder on MNIST (`02_autoencoder_mnist.py`)
- Building deeper autoencoders for images
- Training on MNIST digit dataset
- Visualizing reconstructions
- 2D latent space visualization
- Comparing different latent dimensions

### 3. Variational Autoencoder (`03_variational_autoencoder.py`)
- Why VAE over regular autoencoder
- Probabilistic encoding (mean and variance)
- Reparameterization trick
- VAE loss: reconstruction + KL divergence
- Generating new samples from the prior

### 4. Latent Space Visualization (`04_latent_space_visualization.py`)
- Comparing AE vs VAE latent spaces
- Embedding images in latent space plots
- Decoder manifold visualization
- Latent arithmetic (feature manipulation)
- Dimension traversal
- t-SNE/PCA for high-dimensional latent spaces

## Running the Code

Each file can be run independently:

```bash
cd lec12-encoder-decoder
python 01_autoencoder_basics.py
python 02_autoencoder_mnist.py
python 03_variational_autoencoder.py
python 04_latent_space_visualization.py
```

## Key Concepts

### Autoencoder
```
Input → [Encoder] → z (latent) → [Decoder] → Reconstruction
```
- Learns to compress and reconstruct data
- Bottleneck forces learning of essential features
- Loss: MSE between input and reconstruction

### Variational Autoencoder (VAE)
```
Input → [Encoder] → (μ, σ²) → sample z → [Decoder] → Reconstruction
```
- Encodes to a distribution, not a point
- Reparameterization: z = μ + σ·ε (ε ~ N(0,1))
- Loss: Reconstruction + β·KL(q(z|x) || N(0,1))
- Enables meaningful generation by sampling from N(0,1)

### Latent Space
- Low-dimensional representation of high-dimensional data
- Similar inputs cluster together
- VAE: smooth, continuous, regular structure
- Can interpolate, perform arithmetic, traverse dimensions

## Generated Visualizations

Running the code will generate various PNG files showing:
- Training progress
- Original vs reconstructed images
- Latent space scatter plots
- Decoder manifolds
- Interpolation between digits
- Latent arithmetic results
