"""
Topic 2: Seq2Seq Encoder-Decoder
=================================

This module demonstrates:
1. The seq2seq architecture: encoder RNN + decoder RNN
2. Context vector as information bottleneck
3. Teacher forcing during training
4. Greedy decoding at inference
5. Example: learning to reverse sequences of digits
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


# Special tokens
PAD_TOKEN = 0
SOS_TOKEN = 1  # Start Of Sequence
EOS_TOKEN = 2  # End Of Sequence


def explain_seq2seq_concept():
    """
    Explain the sequence-to-sequence architecture.
    """
    print("\n" + "=" * 70)
    print("Sequence-to-Sequence (Seq2Seq) Architecture")
    print("=" * 70)

    print("""
What is Seq2Seq?
----------------
A model that maps one sequence to another sequence, where the input
and output can have DIFFERENT lengths.

Applications:
  - Machine translation:  "I am a student" -> "Je suis etudiant"
  - Text summarization:   long article -> short summary
  - Chatbots:             question -> answer
  - Speech recognition:   audio frames -> text

Architecture:
                    Encoder                     Decoder
                 (reads input)             (generates output)

    x₁ → [LSTM] → h₁                     [LSTM] → y₁
              ↓                              ↑  ↓
    x₂ → [LSTM] → h₂                     [LSTM] → y₂
              ↓                              ↑  ↓
    x₃ → [LSTM] → h₃                     [LSTM] → y₃
              ↓                              ↑  ↓
    x₄ → [LSTM] → h₄ ──> context (c) ──> [LSTM] → y₄
                          (last hidden)      ↑
                                           <SOS>

Key Ideas:
----------
1. ENCODER: Reads entire input sequence, produces context vector c
   - c = final hidden state of encoder
   - Summarizes all input information into a fixed-size vector

2. DECODER: Generates output sequence one token at a time
   - Initialized with context vector c
   - At each step, takes previous output as input
   - Stops when it generates <EOS> token

3. The context vector is the ONLY connection between encoder and decoder
   - This is both a strength (clean separation) and a weakness (bottleneck)
""")


def explain_teacher_forcing():
    """
    Explain teacher forcing during training.
    """
    print("\n" + "=" * 70)
    print("Teacher Forcing")
    print("=" * 70)

    print("""
Problem: During training, the decoder generates tokens one at a time.
If it makes an error early on, all subsequent tokens will be wrong too!

Example (without teacher forcing):
  Target:  "Je suis etudiant"
  Step 1:  decoder predicts "Le" (wrong!)
  Step 2:  feeds "Le" as input -> predicts "chat" (completely off track)
  Step 3:  feeds "chat" as input -> garbage...

Solution: Teacher Forcing
  Instead of feeding the decoder's own predictions, feed the GROUND TRUTH
  from the previous step.

  Step 1:  feed <SOS>       -> predicts "Le" (wrong, but loss computed)
  Step 2:  feed "Je" (true) -> predicts "suis" (back on track!)
  Step 3:  feed "suis" (true) -> predicts "etudiant"

Training with teacher forcing:
  Decoder input:   <SOS>,  "Je",    "suis",    "etudiant"
  Decoder target:  "Je",   "suis",  "etudiant", <EOS>

At inference (no ground truth available):
  Decoder input:   <SOS>,  pred₁,   pred₂,     pred₃, ...
  Stop when:       pred_t == <EOS>

This is called "greedy decoding" because we pick the most likely
token at each step.
""")


def generate_reversal_data(n_samples, seq_len, vocab_size=10):
    """
    Generate data for sequence reversal task.
    Input: sequence of digits [3, 7, 1, 5]
    Target: reversed sequence [5, 1, 7, 3]

    Both wrapped with SOS and EOS tokens for the decoder.

    Args:
        n_samples: Number of sequences to generate
        seq_len: Length of each sequence (before adding SOS/EOS)
        vocab_size: Number of different digits (tokens 3 to vocab_size+2)

    Returns:
        encoder_input: (n_samples, seq_len) - digits to reverse
        decoder_input: (n_samples, seq_len+1) - SOS + reversed digits
        decoder_target: (n_samples, seq_len+1) - reversed digits + EOS
    """
    # Generate random digit sequences (tokens start at 3 to avoid special tokens)
    sequences = np.random.randint(3, 3 + vocab_size, (n_samples, seq_len))

    # Encoder input: original sequence
    encoder_input = torch.tensor(sequences, dtype=torch.long)

    # Reversed sequences
    reversed_seqs = sequences[:, ::-1].copy()

    # Decoder input: SOS + reversed (teacher forcing input)
    decoder_input = torch.zeros(n_samples, seq_len + 1, dtype=torch.long)
    decoder_input[:, 0] = SOS_TOKEN
    decoder_input[:, 1:] = torch.tensor(reversed_seqs)

    # Decoder target: reversed + EOS
    decoder_target = torch.zeros(n_samples, seq_len + 1, dtype=torch.long)
    decoder_target[:, :-1] = torch.tensor(reversed_seqs)
    decoder_target[:, -1] = EOS_TOKEN

    return encoder_input, decoder_input, decoder_target


class Encoder(nn.Module):
    """
    Encoder RNN: reads the input sequence and produces a context vector.

    The context vector is the final hidden state, which summarizes
    the entire input sequence into a fixed-size vector.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        """
        Args:
            x: Input sequence (batch_size, seq_len)

        Returns:
            outputs: All hidden states (batch_size, seq_len, hidden_dim)
            hidden: Final hidden state tuple (h_n, c_n) = context vector
        """
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        outputs, hidden = self.lstm(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    """
    Decoder RNN: generates the output sequence one token at a time.

    Initialized with the encoder's context vector (hidden state).
    At each step, takes the previous token and produces the next one.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        """
        Args:
            x: Decoder input (batch_size, seq_len) - SOS + previous tokens
            hidden: Initial hidden state from encoder (context vector)

        Returns:
            output: Token logits (batch_size, seq_len, vocab_size)
            hidden: Final hidden state
        """
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)  # (batch, seq_len, vocab_size)
        return output, hidden


class Seq2Seq(nn.Module):
    """
    Complete Seq2Seq model: Encoder + Decoder.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim)
        self.vocab_size = vocab_size

    def forward(self, encoder_input, decoder_input):
        """
        Forward pass with teacher forcing.

        Args:
            encoder_input: Source sequence (batch, src_len)
            decoder_input: Target input with SOS (batch, tgt_len)

        Returns:
            output: Token logits (batch, tgt_len, vocab_size)
        """
        # Encode: read input, get context vector
        _, context = self.encoder(encoder_input)

        # Decode: generate output using context + teacher forcing
        output, _ = self.decoder(decoder_input, context)

        return output

    def predict(self, encoder_input, max_len=20):
        """
        Greedy decoding at inference time (no teacher forcing).

        Args:
            encoder_input: Source sequence (batch, src_len)
            max_len: Maximum output length

        Returns:
            predictions: Predicted token IDs (batch, max_len)
        """
        batch_size = encoder_input.size(0)

        # Encode
        _, hidden = self.encoder(encoder_input)

        # Start with SOS token
        decoder_input = torch.full((batch_size, 1), SOS_TOKEN, dtype=torch.long)
        predictions = []

        for _ in range(max_len):
            output, hidden = self.decoder(decoder_input, hidden)
            # Greedy: pick the most likely token
            pred = output.argmax(dim=-1)  # (batch, 1)
            predictions.append(pred)
            decoder_input = pred  # Feed prediction as next input

        return torch.cat(predictions, dim=1)  # (batch, max_len)


def train_seq2seq(model, train_data, epochs=100, lr=0.005):
    """
    Train the seq2seq model.
    """
    print("\n" + "=" * 70)
    print("Training Seq2Seq Model")
    print("=" * 70)

    encoder_input, decoder_input, decoder_target = train_data

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 70)

    for epoch in range(epochs):
        model.train()

        # Forward pass (with teacher forcing)
        output = model(encoder_input, decoder_input)

        # Compute loss
        # Reshape for CrossEntropyLoss: (batch*seq_len, vocab_size) vs (batch*seq_len)
        loss = criterion(
            output.view(-1, model.vocab_size),
            decoder_target.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 20 == 0 or epoch == 0:
            # Check accuracy
            model.eval()
            with torch.no_grad():
                preds = model.predict(encoder_input,
                                      max_len=decoder_target.size(1))
                # Compare predictions (ignore EOS position for accuracy)
                target_no_eos = decoder_target[:, :-1]
                preds_trimmed = preds[:, :target_no_eos.size(1)]
                accuracy = (preds_trimmed == target_no_eos).float().mean().item()

            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f} | "
                  f"Token Accuracy: {accuracy:.2%}")

    print("-" * 70)
    print(f"Final Loss: {losses[-1]:.4f}")

    return losses


def evaluate_seq2seq(model, test_data, n_examples=10):
    """
    Evaluate and display seq2seq predictions.
    """
    print("\n" + "=" * 70)
    print("Evaluation: Sequence Reversal")
    print("=" * 70)

    encoder_input, decoder_input, decoder_target = test_data

    model.eval()
    with torch.no_grad():
        predictions = model.predict(encoder_input,
                                    max_len=decoder_target.size(1))

    # Display examples
    print(f"\n{'Input':>25s} | {'Target':>25s} | {'Predicted':>25s} | {'Correct?':>8s}")
    print("-" * 95)

    correct_sequences = 0
    for i in range(min(n_examples, len(encoder_input))):
        inp = encoder_input[i].tolist()
        tgt = [t for t in decoder_target[i].tolist() if t != EOS_TOKEN]
        pred = predictions[i].tolist()[:len(tgt)]

        is_correct = (tgt == pred)
        correct_sequences += int(is_correct)

        inp_str = ' '.join(str(x) for x in inp)
        tgt_str = ' '.join(str(x) for x in tgt)
        pred_str = ' '.join(str(x) for x in pred)
        status = "Yes" if is_correct else "No"

        print(f"{inp_str:>25s} | {tgt_str:>25s} | {pred_str:>25s} | {status:>8s}")

    total = min(n_examples, len(encoder_input))
    print(f"\nSequence-level accuracy: {correct_sequences}/{total} "
          f"({correct_sequences/total:.0%})")


def demonstrate_seq2seq():
    """
    Main demonstration of seq2seq encoder-decoder.
    """
    # Explain concepts
    explain_seq2seq_concept()
    explain_teacher_forcing()

    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    seq_len = 5
    vocab_size = 13  # 0=PAD, 1=SOS, 2=EOS, 3-12 = digits
    embed_dim = 32
    hidden_dim = 64

    print("\n" + "=" * 70)
    print(f"Task: Reverse a Sequence of {seq_len} Digits")
    print("=" * 70)
    print(f"\nExample: [3, 7, 1, 5, 9] -> [9, 5, 1, 7, 3]")
    print(f"\nVocab size: {vocab_size} (3 special + 10 digits)")
    print(f"Embedding dim: {embed_dim}")
    print(f"Hidden dim: {hidden_dim}")

    # Generate data
    train_data = generate_reversal_data(1000, seq_len)
    test_data = generate_reversal_data(200, seq_len)

    print(f"\nTraining samples: {train_data[0].shape[0]}")
    print(f"Test samples: {test_data[0].shape[0]}")

    # Show data format
    print(f"\nData format example:")
    print(f"  Encoder input:  {train_data[0][0].tolist()}")
    print(f"  Decoder input:  {train_data[1][0].tolist()}  (SOS + reversed)")
    print(f"  Decoder target: {train_data[2][0].tolist()}  (reversed + EOS)")

    # Create model
    model = Seq2Seq(vocab_size, embed_dim, hidden_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Train
    losses = train_seq2seq(model, train_data, epochs=200, lr=0.005)

    # Evaluate
    evaluate_seq2seq(model, test_data, n_examples=15)

    # Visualize training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title('Seq2Seq Training Loss (Sequence Reversal)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "seq2seq_training.png"),
                dpi=150, bbox_inches="tight")
    print("\nTraining loss saved to 'seq2seq_training.png'")

    # Test with longer sequences to show limitation
    print("\n" + "=" * 70)
    print("Limitation: Fixed Context Vector")
    print("=" * 70)
    print("""
The context vector (final hidden state of the encoder) must compress
the ENTIRE input sequence into a single fixed-size vector.

For short sequences (5 digits), this works well.
For longer sequences, the context vector becomes a bottleneck:
  - Information from early tokens gets "forgotten"
  - Performance degrades as sequence length increases

Solution: Attention mechanism (next topic!)
""")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key takeaways:
  1. Seq2Seq = Encoder RNN + Decoder RNN
  2. Encoder compresses input into a context vector (last hidden state)
  3. Decoder generates output one token at a time
  4. Teacher forcing: use ground truth as decoder input during training
  5. Greedy decoding: use model's own predictions at inference
  6. Limitation: fixed-size context vector is a bottleneck for long sequences

Next: Attention mechanism to solve the bottleneck problem!
""")


if __name__ == "__main__":
    demonstrate_seq2seq()
