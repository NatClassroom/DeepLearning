"""
Topic 1: Sequence Modeling with RNNs
=====================================

This module demonstrates:
1. Why we need recurrent neural networks for sequence data
2. How RNNs process sequences using hidden states
3. Simple RNN vs LSTM vs GRU
4. Processing variable-length sequences
5. Sequence classification example
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


def explain_why_rnns():
    """
    Explain why we need RNNs for sequence data.
    """
    print("\n" + "=" * 70)
    print("Why Do We Need Recurrent Neural Networks?")
    print("=" * 70)

    print("""
Problem with Fully Connected Networks for Sequences:
-----------------------------------------------------
Fully connected networks expect fixed-size inputs, but sequences can
have variable lengths. More importantly, the ORDER of elements matters!

Example: "The cat sat on the mat" vs "mat the on sat cat the"
  - Same words, different meanings
  - A fully connected network treats each input independently

What makes sequence data special:
1. VARIABLE LENGTH - Sentences, time series have different lengths
2. ORDER MATTERS  - "not good" vs "good not" vs "good"
3. LONG-RANGE DEPENDENCIES - Words far apart can be related
4. SHARED PATTERNS - "running" at position 3 means the same as at position 7

Solution: Recurrent Neural Networks
------------------------------------
Process one element at a time, maintaining a "memory" (hidden state)
that carries information from previous steps.

    x₁ ──> [RNN] ──> h₁
              ↓
    x₂ ──> [RNN] ──> h₂      Same weights
              ↓                at every step!
    x₃ ──> [RNN] ──> h₃
              ↓
    x₄ ──> [RNN] ──> h₄ ──> output

Key idea: h_t = f(W_hh · h_{t-1} + W_xh · x_t + b)
  - h_t: hidden state at time t (the "memory")
  - x_t: input at time t
  - W_hh, W_xh: shared weight matrices (same at every step)
""")


def explain_rnn_variants():
    """
    Explain Simple RNN vs LSTM vs GRU.
    """
    print("\n" + "=" * 70)
    print("RNN Variants: Simple RNN vs LSTM vs GRU")
    print("=" * 70)

    print("""
1. SIMPLE RNN (Elman RNN)
   h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)

   Problem: Vanishing/exploding gradients!
   - Gradients shrink exponentially over long sequences
   - Cannot learn long-range dependencies
   - Example: "The cat, which was sitting on ..., was black"
              ^--- hard to connect to "was black" 10 words later

2. LSTM (Long Short-Term Memory)
   Uses THREE gates to control information flow:

   ┌─────────────────────────────────────────────┐
   │                                             │
   │  Forget gate:  f_t = σ(W_f · [h_{t-1}, x_t])  │
   │  Input gate:   i_t = σ(W_i · [h_{t-1}, x_t])  │
   │  Output gate:  o_t = σ(W_o · [h_{t-1}, x_t])  │
   │                                             │
   │  Cell update:  c̃_t = tanh(W_c · [h_{t-1}, x_t])│
   │  Cell state:   c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t│
   │  Hidden state: h_t = o_t ⊙ tanh(c_t)       │
   │                                             │
   └─────────────────────────────────────────────┘

   - Forget gate: what to throw away from cell state
   - Input gate: what new information to store
   - Output gate: what to output from cell state
   - Cell state: "highway" for gradients (solves vanishing gradient!)

3. GRU (Gated Recurrent Unit)
   Simplified LSTM with TWO gates:

   Reset gate:  r_t = σ(W_r · [h_{t-1}, x_t])
   Update gate: z_t = σ(W_z · [h_{t-1}, x_t])
   Candidate:   h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
   Hidden:      h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

   - Fewer parameters than LSTM
   - Often similar performance
   - Update gate combines forget and input gates

Comparison:
-----------
   Model      | Gates | Parameters | Long Sequences
   Simple RNN |   0   |   Fewest   | Poor
   GRU        |   2   |   Medium   | Good
   LSTM       |   3   |   Most     | Good
""")


def demonstrate_rnn_hidden_state():
    """
    Show how RNN hidden state evolves as it processes a sequence.
    """
    print("\n" + "=" * 70)
    print("Demonstration: RNN Hidden State Evolution")
    print("=" * 70)

    # Create a simple RNN
    input_size = 1
    hidden_size = 8
    rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    # Create a simple sequence (sine wave)
    t = torch.linspace(0, 4 * np.pi, 50)
    sequence = torch.sin(t).unsqueeze(0).unsqueeze(-1)  # (1, 50, 1)

    print(f"\nInput sequence shape: {sequence.shape}")
    print(f"  - Batch size: 1")
    print(f"  - Sequence length: 50")
    print(f"  - Input features: 1 (sine wave value)")
    print(f"\nRNN hidden size: {hidden_size}")

    # Process sequence and collect hidden states
    rnn.eval()
    with torch.no_grad():
        output, h_n = rnn(sequence)

    print(f"\nOutput shape: {output.shape}")
    print(f"  - Contains hidden state at EVERY time step")
    print(f"\nh_n shape: {h_n.shape}")
    print(f"  - Contains hidden state at LAST time step only")

    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot input sequence
    axes[0].plot(t.numpy(), sequence.squeeze().numpy(), 'b-', linewidth=2)
    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].set_title('Input Sequence (Sine Wave)', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Plot hidden state evolution (first 4 dimensions)
    hidden_states = output.squeeze().numpy()  # (50, hidden_size)
    for i in range(min(4, hidden_size)):
        axes[1].plot(t.numpy(), hidden_states[:, i], linewidth=2,
                     label=f'h[{i}]', alpha=0.8)
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel('Hidden State Value', fontsize=12)
    axes[1].set_title('RNN Hidden State Evolution (First 4 Dimensions)', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rnn_hidden_states.png"),
                dpi=150, bbox_inches="tight")
    print("\nVisualization saved to 'rnn_hidden_states.png'")

    print("\nKey observation:")
    print("  - Each hidden dimension captures different aspects of the input")
    print("  - Hidden state evolves smoothly, carrying memory of past inputs")


def compare_rnn_variants():
    """
    Compare Simple RNN, LSTM, and GRU on a long-range dependency task.
    """
    print("\n" + "=" * 70)
    print("Comparison: Simple RNN vs LSTM vs GRU")
    print("=" * 70)

    print("""
Task: Remember the first element of a sequence
  - Input: sequence of random numbers, length 20-50
  - Target: the first element
  - Tests long-range memory!
""")

    torch.manual_seed(42)

    # Generate data
    def generate_memory_data(n_samples=500, seq_len=30):
        """Generate sequences where target is the first element."""
        data = torch.randn(n_samples, seq_len, 1)
        targets = data[:, 0, 0]  # First element
        return data, targets

    train_data, train_targets = generate_memory_data(500, seq_len=30)
    test_data, test_targets = generate_memory_data(100, seq_len=30)

    print(f"Training data: {train_data.shape}")
    print(f"Task: predict the first element from the last hidden state")

    # Define models
    class SequenceMemory(nn.Module):
        def __init__(self, rnn_type='RNN', hidden_size=32):
            super(SequenceMemory, self).__init__()
            if rnn_type == 'RNN':
                self.rnn = nn.RNN(1, hidden_size, batch_first=True)
            elif rnn_type == 'LSTM':
                self.rnn = nn.LSTM(1, hidden_size, batch_first=True)
            elif rnn_type == 'GRU':
                self.rnn = nn.GRU(1, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            self.rnn_type = rnn_type

        def forward(self, x):
            output, h_n = self.rnn(x)
            if self.rnn_type == 'LSTM':
                h_n = h_n[0]  # LSTM returns (h_n, c_n)
            last_hidden = h_n.squeeze(0)
            return self.fc(last_hidden).squeeze(-1)

    # Train each variant
    results = {}
    for rnn_type in ['RNN', 'LSTM', 'GRU']:
        torch.manual_seed(42)
        model = SequenceMemory(rnn_type=rnn_type, hidden_size=32)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        losses = []
        for epoch in range(200):
            pred = model(train_data)
            loss = criterion(pred, train_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_pred = model(test_data)
            test_loss = criterion(test_pred, test_targets).item()

        results[rnn_type] = {'losses': losses, 'test_loss': test_loss}

        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{rnn_type:4s} | Parameters: {n_params:5d} | "
              f"Train Loss: {losses[-1]:.4f} | Test Loss: {test_loss:.4f}")

    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'RNN': 'blue', 'LSTM': 'red', 'GRU': 'green'}
    for rnn_type, data in results.items():
        axes[0].plot(data['losses'], color=colors[rnn_type],
                     linewidth=2, label=rnn_type, alpha=0.8)

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('MSE Loss', fontsize=12)
    axes[0].set_title('Training Loss Comparison', fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Bar chart of test losses
    types = list(results.keys())
    test_losses = [results[t]['test_loss'] for t in types]
    bar_colors = [colors[t] for t in types]
    axes[1].bar(types, test_losses, color=bar_colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Test MSE Loss', fontsize=12)
    axes[1].set_title('Test Loss (Remember First Element)', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rnn_comparison.png"),
                dpi=150, bbox_inches="tight")
    print("\nComparison saved to 'rnn_comparison.png'")

    print("\nKey takeaway:")
    print("  - LSTM and GRU are much better at remembering long-range information")
    print("  - Simple RNN struggles with the vanishing gradient problem")


def demonstrate_sequence_classification():
    """
    Demonstrate sequence classification: classify a sequence pattern.
    """
    print("\n" + "=" * 70)
    print("Demonstration: Sequence Classification")
    print("=" * 70)

    print("""
Task: Classify sequences as "increasing trend" (1) or "decreasing trend" (0)
  - Input: noisy sequence of 20 values
  - Output: 0 or 1
""")

    torch.manual_seed(42)

    # Generate data
    def generate_trend_data(n_samples=500, seq_len=20):
        sequences = []
        labels = []
        for _ in range(n_samples):
            label = np.random.randint(0, 2)
            if label == 1:  # Increasing
                trend = np.linspace(0, 1, seq_len)
            else:  # Decreasing
                trend = np.linspace(1, 0, seq_len)
            noise = np.random.randn(seq_len) * 0.3
            seq = trend + noise
            sequences.append(seq)
            labels.append(label)
        return (torch.tensor(np.array(sequences), dtype=torch.float32).unsqueeze(-1),
                torch.tensor(labels, dtype=torch.float32))

    train_seqs, train_labels = generate_trend_data(800)
    test_seqs, test_labels = generate_trend_data(200)

    print(f"Training samples: {train_seqs.shape[0]}")
    print(f"Sequence length: {train_seqs.shape[1]}")

    # Model
    class SequenceClassifier(nn.Module):
        def __init__(self, hidden_size=32):
            super(SequenceClassifier, self).__init__()
            self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            output, (h_n, c_n) = self.lstm(x)
            last_hidden = h_n.squeeze(0)
            return torch.sigmoid(self.fc(last_hidden)).squeeze(-1)

    model = SequenceClassifier(hidden_size=32)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print(f"\nModel: LSTM -> Linear -> Sigmoid")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Train
    losses = []
    accuracies = []

    print(f"\nTraining for 100 epochs...")
    print("-" * 70)

    for epoch in range(100):
        model.train()
        pred = model(train_seqs)
        loss = criterion(pred, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        acc = ((pred > 0.5).float() == train_labels).float().mean().item()
        accuracies.append(acc)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/100 | Loss: {loss.item():.4f} | "
                  f"Accuracy: {acc:.2%}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_pred = model(test_seqs)
        test_acc = ((test_pred > 0.5).float() == test_labels).float().mean().item()

    print("-" * 70)
    print(f"Test Accuracy: {test_acc:.2%}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Training loss
    axes[0, 0].plot(losses, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('BCE Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)

    # Training accuracy
    axes[0, 1].plot(accuracies, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Training Accuracy', fontsize=14)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)

    # Example sequences
    for i in range(5):
        idx_inc = (test_labels == 1).nonzero(as_tuple=True)[0][i]
        idx_dec = (test_labels == 0).nonzero(as_tuple=True)[0][i]
        axes[1, 0].plot(test_seqs[idx_inc].squeeze().numpy(),
                        alpha=0.5, color='blue')
        axes[1, 1].plot(test_seqs[idx_dec].squeeze().numpy(),
                        alpha=0.5, color='red')

    axes[1, 0].set_title('Increasing Trend (Label=1)', fontsize=14)
    axes[1, 0].set_xlabel('Time Step', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].set_title('Decreasing Trend (Label=0)', fontsize=14)
    axes[1, 1].set_xlabel('Time Step', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sequence_classification.png"),
                dpi=150, bbox_inches="tight")
    print("\nVisualization saved to 'sequence_classification.png'")


def demonstrate_sequence_modeling():
    """
    Main demonstration of sequence modeling with RNNs.
    """
    # Explain concepts
    explain_why_rnns()
    explain_rnn_variants()

    # Demonstrate hidden state
    demonstrate_rnn_hidden_state()

    # Compare RNN variants
    compare_rnn_variants()

    # Sequence classification
    demonstrate_sequence_classification()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key takeaways:
  1. RNNs process sequences step-by-step with a hidden state (memory)
  2. Simple RNNs suffer from vanishing gradients on long sequences
  3. LSTM uses gates (forget, input, output) to control information flow
  4. GRU is a simpler alternative with similar performance
  5. For classification, use the last hidden state as a sequence summary

Next: Use RNNs as encoder and decoder for sequence-to-sequence tasks!
""")


if __name__ == "__main__":
    demonstrate_sequence_modeling()
