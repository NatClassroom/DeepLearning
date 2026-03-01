"""
Topic 3: Attention Mechanism
=============================

This module demonstrates:
1. The bottleneck problem with fixed context vector
2. Attention: letting the decoder "look back" at all encoder states
3. Bahdanau (additive) attention implementation
4. Attention weights visualization
5. Example: sorting sequences with attention
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

# Create results directory if it doesn't exist
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Special tokens
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2


def explain_attention_concept():
    """
    Explain why attention is needed and how it works.
    """
    print("\n" + "=" * 70)
    print("The Attention Mechanism")
    print("=" * 70)

    print("""
The Bottleneck Problem:
-----------------------
In basic seq2seq, the ENTIRE input is compressed into one vector:

    x₁ → h₁
    x₂ → h₂     ALL info must fit in
    x₃ → h₃  ─────> context (c) ────> decoder
    x₄ → h₄

For long sequences, this single vector can't capture everything!

The Attention Solution:
-----------------------
Instead of using ONE context vector, let the decoder LOOK BACK at
ALL encoder hidden states at each decoding step:

    Encoder states: h₁, h₂, h₃, h₄  (all kept!)
                     ↑   ↑   ↑   ↑
                     α₁  α₂  α₃  α₄  ← attention weights (sum to 1)
                     ↓   ↓   ↓   ↓
    Context at t:   c_t = α₁·h₁ + α₂·h₂ + α₃·h₃ + α₄·h₄

At each decoder step t, we compute:
1. SCORE: How relevant is each encoder state to current decoder state?
   score(s_t, h_i) = v^T · tanh(W_s · s_t + W_h · h_i)  (Bahdanau)

2. WEIGHTS: Normalize scores with softmax
   α_t = softmax(scores)

3. CONTEXT: Weighted sum of encoder states
   c_t = Σ α_tᵢ · h_i

4. OUTPUT: Combine context with decoder state
   y_t = f(s_t, c_t)

Why This Helps:
---------------
- Decoder can focus on relevant parts of input at each step
- No information bottleneck (all encoder states are available)
- Attention weights show WHAT the model is looking at
- For translation: aligns source and target words
""")


def explain_attention_types():
    """
    Explain different types of attention mechanisms.
    """
    print("\n" + "=" * 70)
    print("Types of Attention")
    print("=" * 70)

    print("""
1. BAHDANAU ATTENTION (Additive)
   score(s_t, h_i) = v^T · tanh(W_s · s_t + W_h · h_i)
   - Uses a small neural network to compute scores
   - Most flexible, can learn complex alignments
   - Original attention mechanism (2014)

2. LUONG ATTENTION (Multiplicative)
   score(s_t, h_i) = s_t^T · W · h_i  (general)
   score(s_t, h_i) = s_t^T · h_i       (dot product)
   - Simpler and faster than Bahdanau
   - Dot product is fastest but requires same dimensions

3. SCALED DOT-PRODUCT ATTENTION (Transformer)
   score(Q, K) = Q · K^T / √d_k
   - Used in Transformers (next lecture!)
   - Scaling prevents softmax saturation

Attention is like a "soft lookup":
  - Query (Q): what we're looking for (decoder state)
  - Keys (K): what we're matching against (encoder states)
  - Values (V): what we're retrieving (encoder states)
  - Output: weighted sum of values, weighted by query-key similarity
""")


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention mechanism.

    Computes attention scores between decoder state and all encoder states,
    then returns a weighted sum of encoder states (context vector).
    """

    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        """
        Args:
            decoder_state: Current decoder hidden state (batch, hidden_dim)
            encoder_outputs: All encoder hidden states (batch, src_len, hidden_dim)

        Returns:
            context: Weighted sum of encoder states (batch, hidden_dim)
            attention_weights: Attention distribution (batch, src_len)
        """
        # decoder_state: (batch, hidden_dim) -> (batch, 1, hidden_dim)
        decoder_state = decoder_state.unsqueeze(1)

        # Compute attention scores
        # score = v^T * tanh(W_s * s_t + W_h * h_i)
        scores = self.v(torch.tanh(
            self.W_s(decoder_state) + self.W_h(encoder_outputs)
        ))  # (batch, src_len, 1)
        scores = scores.squeeze(-1)  # (batch, src_len)

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch, src_len)

        # Context vector: weighted sum of encoder outputs
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, src_len)
            encoder_outputs                   # (batch, src_len, hidden_dim)
        ).squeeze(1)  # (batch, hidden_dim)

        return context, attention_weights


class AttentionEncoder(nn.Module):
    """
    Encoder for attention-based seq2seq.
    Same as basic encoder, but we return ALL hidden states.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(AttentionEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.lstm(embedded)
        return outputs, hidden  # outputs = ALL hidden states


class AttentionDecoder(nn.Module):
    """
    Decoder with Bahdanau attention.

    At each step:
    1. Compute attention over encoder outputs
    2. Concatenate attention context with input embedding
    3. Feed through LSTM
    4. Predict next token
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(AttentionDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(hidden_dim)
        # Input to LSTM: embedding + attention context
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward_step(self, x, hidden, encoder_outputs):
        """
        Single decoding step with attention.

        Args:
            x: Input token (batch, 1)
            hidden: Previous hidden state (h, c)
            encoder_outputs: All encoder states (batch, src_len, hidden_dim)

        Returns:
            output: Token logits (batch, 1, vocab_size)
            hidden: Updated hidden state
            attention_weights: Attention distribution (batch, src_len)
        """
        embedded = self.embedding(x)  # (batch, 1, embed_dim)

        # Compute attention using current decoder hidden state
        h_n = hidden[0].squeeze(0)  # (batch, hidden_dim)
        context, attention_weights = self.attention(h_n, encoder_outputs)

        # Concatenate embedding with attention context
        lstm_input = torch.cat([
            embedded,
            context.unsqueeze(1)  # (batch, 1, hidden_dim)
        ], dim=-1)  # (batch, 1, embed_dim + hidden_dim)

        output, hidden = self.lstm(lstm_input, hidden)
        output = self.fc(output)  # (batch, 1, vocab_size)

        return output, hidden, attention_weights


class Seq2SeqAttention(nn.Module):
    """
    Seq2Seq model with Bahdanau attention.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = AttentionEncoder(vocab_size, embed_dim, hidden_dim)
        self.decoder = AttentionDecoder(vocab_size, embed_dim, hidden_dim)
        self.vocab_size = vocab_size

    def forward(self, encoder_input, decoder_input):
        """
        Forward pass with teacher forcing.
        Returns logits and attention weights for visualization.
        """
        # Encode
        encoder_outputs, hidden = self.encoder(encoder_input)

        # Decode step by step (to collect attention weights)
        batch_size = decoder_input.size(0)
        seq_len = decoder_input.size(1)
        all_outputs = []
        all_attention = []

        for t in range(seq_len):
            x_t = decoder_input[:, t:t+1]  # (batch, 1)
            output, hidden, attn_weights = self.decoder.forward_step(
                x_t, hidden, encoder_outputs
            )
            all_outputs.append(output)
            all_attention.append(attn_weights)

        outputs = torch.cat(all_outputs, dim=1)  # (batch, seq_len, vocab_size)
        attention = torch.stack(all_attention, dim=1)  # (batch, seq_len, src_len)

        return outputs, attention

    def predict(self, encoder_input, max_len=20):
        """
        Greedy decoding with attention.
        """
        batch_size = encoder_input.size(0)

        # Encode
        encoder_outputs, hidden = self.encoder(encoder_input)

        # Start with SOS
        decoder_input = torch.full((batch_size, 1), SOS_TOKEN, dtype=torch.long)
        predictions = []
        attention_weights = []

        for _ in range(max_len):
            output, hidden, attn = self.decoder.forward_step(
                decoder_input, hidden, encoder_outputs
            )
            pred = output.argmax(dim=-1)  # (batch, 1)
            predictions.append(pred)
            attention_weights.append(attn)
            decoder_input = pred

        predictions = torch.cat(predictions, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)

        return predictions, attention_weights


def generate_sorting_data(n_samples, seq_len, vocab_size=10):
    """
    Generate data for sequence sorting task.
    Input: unsorted sequence [7, 3, 9, 1, 5]
    Target: sorted sequence [1, 3, 5, 7, 9]

    This task benefits from attention because the decoder needs to
    "search" the input to find the next smallest element.
    """
    sequences = np.random.randint(3, 3 + vocab_size, (n_samples, seq_len))

    encoder_input = torch.tensor(sequences, dtype=torch.long)

    sorted_seqs = np.sort(sequences, axis=1)

    decoder_input = torch.zeros(n_samples, seq_len + 1, dtype=torch.long)
    decoder_input[:, 0] = SOS_TOKEN
    decoder_input[:, 1:] = torch.tensor(sorted_seqs)

    decoder_target = torch.zeros(n_samples, seq_len + 1, dtype=torch.long)
    decoder_target[:, :-1] = torch.tensor(sorted_seqs)
    decoder_target[:, -1] = EOS_TOKEN

    return encoder_input, decoder_input, decoder_target


def train_attention_model(model, train_data, epochs=200, lr=0.005):
    """
    Train the attention-based seq2seq model.
    """
    print("\n" + "=" * 70)
    print("Training Seq2Seq with Attention")
    print("=" * 70)

    encoder_input, decoder_input, decoder_target = train_data

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 70)

    for epoch in range(epochs):
        model.train()

        output, attention = model(encoder_input, decoder_input)

        loss = criterion(
            output.view(-1, model.vocab_size),
            decoder_target.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 40 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                preds, _ = model.predict(encoder_input,
                                         max_len=decoder_target.size(1))
                target_no_eos = decoder_target[:, :-1]
                preds_trimmed = preds[:, :target_no_eos.size(1)]
                accuracy = (preds_trimmed == target_no_eos).float().mean().item()

            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f} | "
                  f"Token Accuracy: {accuracy:.2%}")

    print("-" * 70)
    return losses


def visualize_attention(model, encoder_input, decoder_target, n_examples=4):
    """
    Visualize attention weights as heatmaps.
    """
    print("\n" + "=" * 70)
    print("Visualizing Attention Weights")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        predictions, attention_weights = model.predict(
            encoder_input[:n_examples],
            max_len=decoder_target.size(1)
        )

    fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 5))
    if n_examples == 1:
        axes = [axes]

    for i in range(n_examples):
        src = encoder_input[i].tolist()
        tgt = [t for t in decoder_target[i].tolist() if t != EOS_TOKEN]
        attn = attention_weights[i, :len(tgt), :].numpy()

        im = axes[i].imshow(attn, cmap='YlOrRd', aspect='auto',
                            vmin=0, vmax=1)
        axes[i].set_xticks(range(len(src)))
        axes[i].set_xticklabels([str(x) for x in src], fontsize=10)
        axes[i].set_yticks(range(len(tgt)))
        axes[i].set_yticklabels([str(x) for x in tgt], fontsize=10)
        axes[i].set_xlabel('Input (unsorted)', fontsize=11)
        axes[i].set_ylabel('Output (sorted)', fontsize=11)
        axes[i].set_title(f'Example {i+1}', fontsize=12)
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.suptitle('Attention Weights: Sorting Task\n'
                 '(Bright = high attention)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "attention_weights.png"),
                dpi=150, bbox_inches="tight")
    print("\nAttention visualization saved to 'attention_weights.png'")

    print("\nKey observations:")
    print("  - Each output step attends to a specific input position")
    print("  - For sorting, the model learns to 'search' for the next smallest element")
    print("  - Attention weights reveal the model's alignment strategy")


def demonstrate_attention():
    """
    Main demonstration of attention mechanism.
    """
    # Explain concepts
    explain_attention_concept()
    explain_attention_types()

    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    seq_len = 6
    vocab_size = 13
    embed_dim = 32
    hidden_dim = 64

    print("\n" + "=" * 70)
    print(f"Task: Sort a Sequence of {seq_len} Digits (with Attention)")
    print("=" * 70)
    print(f"\nExample: [7, 3, 9, 1, 5, 4] -> [1, 3, 4, 5, 7, 9]")
    print(f"\nThis task benefits from attention because the decoder needs")
    print(f"to 'search' the input to find the next smallest element.")

    # Generate data
    train_data = generate_sorting_data(1500, seq_len)
    test_data = generate_sorting_data(300, seq_len)

    print(f"\nTraining samples: {train_data[0].shape[0]}")
    print(f"Test samples: {test_data[0].shape[0]}")

    # Create model
    model = Seq2SeqAttention(vocab_size, embed_dim, hidden_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Train
    losses = train_attention_model(model, train_data, epochs=300, lr=0.005)

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluation: Sequence Sorting")
    print("=" * 70)

    encoder_input, _, decoder_target = test_data
    model.eval()
    with torch.no_grad():
        predictions, _ = model.predict(encoder_input,
                                       max_len=decoder_target.size(1))

    n_show = 10
    print(f"\n{'Input (unsorted)':>30s} | {'Target (sorted)':>30s} | "
          f"{'Predicted':>30s} | {'OK?':>4s}")
    print("-" * 105)

    correct = 0
    for i in range(n_show):
        inp = encoder_input[i].tolist()
        tgt = [t for t in decoder_target[i].tolist() if t != EOS_TOKEN]
        pred = predictions[i].tolist()[:len(tgt)]

        is_correct = (tgt == pred)
        correct += int(is_correct)

        print(f"{str(inp):>30s} | {str(tgt):>30s} | "
              f"{str(pred):>30s} | {'Yes' if is_correct else 'No':>4s}")

    # Full test accuracy
    with torch.no_grad():
        target_no_eos = decoder_target[:, :-1]
        preds_trimmed = predictions[:, :target_no_eos.size(1)]
        token_acc = (preds_trimmed == target_no_eos).float().mean().item()
        seq_acc = (preds_trimmed == target_no_eos).all(dim=1).float().mean().item()

    print(f"\nTest token accuracy: {token_acc:.2%}")
    print(f"Test sequence accuracy: {seq_acc:.2%}")

    # Visualize attention
    visualize_attention(model, encoder_input, decoder_target, n_examples=4)

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title('Attention Seq2Seq Training Loss (Sorting)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "attention_training.png"),
                dpi=150, bbox_inches="tight")
    print("\nTraining loss saved to 'attention_training.png'")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key takeaways:
  1. Attention lets the decoder look at ALL encoder states, not just the last
  2. At each decoding step, attention computes a weighted sum of encoder states
  3. Weights are computed using a learned scoring function + softmax
  4. Attention weights are interpretable (show what the model focuses on)
  5. For sorting, attention learns to search for the next smallest element
  6. Attention solves the information bottleneck of basic seq2seq

Next: Apply attention to a practical translation task!
""")


if __name__ == "__main__":
    demonstrate_attention()
