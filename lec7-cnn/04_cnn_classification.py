"""
Topic 4: CNN for Classification
================================

This module demonstrates:
1. Building a complete CNN for image classification
2. Training a CNN model
3. Evaluating model performance
4. Visualizing training progress
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os


class SimpleImageDataset(Dataset):
    """
    Simple dataset for image classification.
    Generates synthetic images or loads real images.
    """
    
    def __init__(self, images, labels):
        """
        Args:
            images: Tensor of shape (N, C, H, W)
            labels: Tensor of shape (N,)
        """
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def generate_synthetic_image_data(n_samples=200, image_size=32, num_classes=3, random_seed=42):
    """
    Generate synthetic image data for classification.
    
    Args:
        n_samples: Number of images per class
        image_size: Size of images (image_size x image_size)
        num_classes: Number of classes
        random_seed: Random seed for reproducibility
    
    Returns:
        images: Tensor of shape (n_samples*num_classes, 3, image_size, image_size)
        labels: Tensor of shape (n_samples*num_classes,)
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    images = []
    labels = []
    
    for class_idx in range(num_classes):
        for _ in range(n_samples):
            # Create a simple pattern for each class
            img = torch.zeros(3, image_size, image_size)
            
            if class_idx == 0:
                # Class 0: Horizontal stripes
                for i in range(image_size):
                    if i % 4 < 2:
                        img[:, i, :] = 0.8
            elif class_idx == 1:
                # Class 1: Vertical stripes
                for j in range(image_size):
                    if j % 4 < 2:
                        img[:, :, j] = 0.8
            else:
                # Class 2: Checkerboard pattern
                for i in range(image_size):
                    for j in range(image_size):
                        if (i // 4 + j // 4) % 2 == 0:
                            img[:, i, j] = 0.8
            
            # Add some noise
            noise = torch.randn(3, image_size, image_size) * 0.1
            img = img + noise
            img = torch.clamp(img, 0, 1)
            
            images.append(img)
            labels.append(class_idx)
    
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    
    print(f"\nGenerated synthetic image dataset:")
    print(f"  - Total samples: {len(images)}")
    print(f"  - Image shape: {images[0].shape}")
    print(f"  - Number of classes: {num_classes}")
    for i in range(num_classes):
        print(f"  - Class {i}: {torch.sum(labels == i)} samples")
    
    return images, labels


def load_real_image_data(data_dir='data/2d_data/', num_classes=3):
    """
    Load real images from directory.
    
    Args:
        data_dir: Directory containing images
        num_classes: Number of classes (assumes equal distribution)
    
    Returns:
        images: Tensor of shape (N, 3, H, W)
        labels: Tensor of shape (N,)
    """
    try:
        import imageio.v3 as iio
        
        if not os.path.exists(data_dir):
            return None, None
        
        filenames = [name for name in os.listdir(data_dir)
                    if os.path.splitext(name)[-1] == '.png']
        
        if len(filenames) == 0:
            return None, None
        
        images = []
        labels = []
        
        # Assign labels based on filename order
        for idx, filename in enumerate(filenames):
            img_path = os.path.join(data_dir, filename)
            img_array = iio.imread(img_path)
            
            # Convert to tensor and normalize
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
            img_tensor = img_tensor / 255.0
            
            # Resize to 32x32 if needed
            if img_tensor.shape[1] != 32 or img_tensor.shape[2] != 32:
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=(32, 32),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            images.append(img_tensor)
            labels.append(idx % num_classes)  # Simple label assignment
        
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        
        print(f"\nLoaded real image dataset:")
        print(f"  - Total samples: {len(images)}")
        print(f"  - Image shape: {images[0].shape}")
        print(f"  - Number of classes: {num_classes}")
        
        return images, labels
    
    except ImportError:
        return None, None
    except Exception as e:
        print(f"Error loading images: {e}")
        return None, None


class CNNClassifier(nn.Module):
    """
    CNN model for image classification.
    
    Architecture:
    - Conv2d(3, 16) -> ReLU -> MaxPool2d
    - Conv2d(16, 32) -> ReLU -> MaxPool2d
    - Conv2d(32, 64) -> ReLU -> MaxPool2d
    - Flatten
    - Linear(64*4*4, 128) -> ReLU
    - Linear(128, num_classes)
    """
    
    def __init__(self, num_classes=3, image_size=32):
        super(CNNClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)      # (B, 3, 32, 32) -> (B, 16, 32, 32)
        x = F.relu(x)
        x = self.pool1(x)      # (B, 16, 32, 32) -> (B, 16, 16, 16)
        
        # Conv block 2
        x = self.conv2(x)      # (B, 16, 16, 16) -> (B, 32, 16, 16)
        x = F.relu(x)
        x = self.pool2(x)      # (B, 32, 16, 16) -> (B, 32, 8, 8)
        
        # Conv block 3
        x = self.conv3(x)      # (B, 32, 8, 8) -> (B, 64, 8, 8)
        x = F.relu(x)
        x = self.pool3(x)      # (B, 64, 8, 8) -> (B, 64, 4, 4)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (B, 64, 4, 4) -> (B, 1024)
        
        # Fully connected layers
        x = self.fc1(x)        # (B, 1024) -> (B, 128)
        x = F.relu(x)
        x = self.fc2(x)        # (B, 128) -> (B, num_classes)
        
        return x


def train_cnn(model, train_loader, epochs=20, lr=0.001, device='cpu'):
    """
    Train a CNN model.
    
    Args:
        model: CNN model
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cpu' or 'cuda')
    
    Returns:
        List of loss values and accuracies during training
    """
    print("\n" + "=" * 60)
    print("Training CNN Model")
    print("=" * 60)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    accuracies = []
    
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
    
    print("-" * 60)
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Final Accuracy: {accuracies[-1]:.2f}%")
    
    return losses, accuracies


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained CNN model
        test_loader: DataLoader for test data
        device: Device to evaluate on
    
    Returns:
        Accuracy and per-class accuracies
    """
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                if label not in class_correct:
                    class_correct[label] = 0
                    class_total[label] = 0
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
    
    accuracy = 100 * correct / total
    per_class_acc = {cls: 100 * class_correct[cls] / class_total[cls] 
                     for cls in class_correct}
    
    return accuracy, per_class_acc


def visualize_training(losses, accuracies):
    """
    Visualize training progress.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(accuracies, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training Accuracy', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("cnn_training_progress.png", dpi=150, bbox_inches="tight")
    print("\nTraining progress saved to 'cnn_training_progress.png'")


def visualize_predictions(model, test_loader, num_samples=8, device='cpu'):
    """
    Visualize model predictions on test samples.
    """
    model.eval()
    model = model.to(device)
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = F.softmax(outputs, dim=1)
    
    # Select samples to visualize
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = images[i].cpu().permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        confidence = probabilities[i][pred_label].item()
        
        axes[i].imshow(img.numpy())
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.2f}')
        axes[i].axis('off')
        
        # Color title based on correctness
        color = 'green' if true_label == pred_label else 'red'
        axes[i].title.set_color(color)
    
    plt.tight_layout()
    plt.savefig("cnn_predictions.png", dpi=150, bbox_inches="tight")
    print("Predictions visualization saved to 'cnn_predictions.png'")


def demonstrate_complete_cnn_classification():
    """
    Demonstrates the complete CNN classification pipeline.
    """
    print("\n" + "=" * 60)
    print("Complete CNN Classification Pipeline")
    print("=" * 60)
    
    # Try to load real images, otherwise use synthetic
    images, labels = load_real_image_data()
    if images is None:
        print("\nReal images not available, using synthetic data...")
        images, labels = generate_synthetic_image_data(n_samples=200, num_classes=3)
    
    # Split into train and test sets
    n_samples = len(images)
    n_train = int(0.8 * n_samples)
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_images = images[train_indices]
    train_labels = labels[train_indices]
    test_images = images[test_indices]
    test_labels = labels[test_indices]
    
    print(f"\nData split:")
    print(f"  - Training samples: {len(train_images)}")
    print(f"  - Test samples: {len(test_images)}")
    
    # Create datasets and dataloaders
    train_dataset = SimpleImageDataset(train_images, train_labels)
    test_dataset = SimpleImageDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    num_classes = len(torch.unique(labels))
    model = CNNClassifier(num_classes=num_classes)
    
    print(f"\nModel architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    losses, accuracies = train_cnn(model, train_loader, epochs=30, lr=0.001, device=device)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)
    
    test_accuracy, per_class_acc = evaluate_model(model, test_loader, device=device)
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")
    print(f"\nPer-class accuracy:")
    for cls, acc in per_class_acc.items():
        print(f"  Class {cls}: {acc:.2f}%")
    
    # Visualize training progress
    visualize_training(losses, accuracies)
    
    # Visualize predictions
    visualize_predictions(model, test_loader, num_samples=8, device=device)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. CNNs are effective for image classification")
    print("  2. Architecture: Conv layers -> Pooling -> FC layers")
    print("  3. Training uses CrossEntropyLoss for multi-class classification")
    print("  4. Batch processing with DataLoader is efficient")
    print("  5. Model.train() and model.eval() control behavior")
    print("  6. Visualizations help understand model performance")
    print("\nGenerated files:")
    print("  - cnn_training_progress.png: Training loss and accuracy")
    print("  - cnn_predictions.png: Sample predictions")


if __name__ == "__main__":
    demonstrate_complete_cnn_classification()

