"""
Topic 4: Seq2Seq Translation
==============================

This module demonstrates:
1. Building a simple word-level translator (English -> French)
2. Vocabulary building and tokenization
3. Encoder-decoder with attention for translation
4. BLEU score evaluation
5. Visualizing attention alignments
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

# Create results directory if it doesn't exist
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Special tokens
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


def explain_translation_pipeline():
    """
    Explain the full translation pipeline.
    """
    print("\n" + "=" * 70)
    print("Neural Machine Translation Pipeline")
    print("=" * 70)

    print("""
Building a neural translator involves several steps:

1. DATA PREPARATION
   - Collect parallel sentences (source, target pairs)
   - Build vocabulary: word -> integer index
   - Tokenize: split sentences into words
   - Add special tokens: <SOS>, <EOS>, <PAD>

2. MODEL
   - Encoder: reads source sentence, produces hidden states
   - Attention: computes relevance of each source word
   - Decoder: generates target sentence word by word

3. TRAINING
   - Teacher forcing: feed ground truth target words
   - Cross-entropy loss on predicted word probabilities
   - Optimize with Adam

4. INFERENCE
   - Greedy decoding: pick most likely word at each step
   - (Advanced: beam search for better translations)

5. EVALUATION
   - BLEU score: measures n-gram overlap with reference translation
   - Attention visualization: check word alignments

Example:
  Source:  "the cat is black"
  Target:  "le chat est noir"

  Tokenized: [the, cat, is, black] -> [le, chat, est, noir]
  Indexed:   [4, 5, 6, 7] -> [4, 5, 6, 7]  (different vocabularies!)
""")


class Vocabulary:
    """
    Maps words to integer indices and back.
    """

    def __init__(self):
        self.word2idx = {'<PAD>': PAD_IDX, '<SOS>': SOS_IDX, '<EOS>': EOS_IDX}
        self.idx2word = {PAD_IDX: '<PAD>', SOS_IDX: '<SOS>', EOS_IDX: '<EOS>'}
        self.word_count = Counter()
        self.n_words = 3  # Start after special tokens

    def add_sentence(self, sentence):
        """Add all words in a sentence to the vocabulary."""
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        """Add a single word to the vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1
        self.word_count[word] += 1

    def sentence_to_indices(self, sentence):
        """Convert a sentence string to a list of indices."""
        return [self.word2idx.get(w, PAD_IDX) for w in sentence.split()]

    def indices_to_sentence(self, indices):
        """Convert a list of indices back to a sentence string."""
        words = []
        for idx in indices:
            if idx == EOS_IDX:
                break
            if idx not in (PAD_IDX, SOS_IDX):
                words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)


def create_translation_dataset():
    """
    Create a small English-French translation dataset.
    This is a toy dataset for demonstration purposes.
    """
    print("\n" + "=" * 70)
    print("Creating Translation Dataset")
    print("=" * 70)

    # English-French sentence pairs
    # We use many combinations of subjects, verbs, and objects so the model
    # must learn word-level alignment rather than memorizing whole sentences.
    subjects_en = ["i", "he", "she", "we", "they"]
    subjects_fr = ["je", "il", "elle", "nous", "ils"]

    be_en = ["am", "is", "is", "are", "are"]
    be_fr = ["suis", "est", "est", "sommes", "sont"]

    adjectives = [
        ("happy", "heureux"), ("sad", "triste"), ("tall", "grand"),
        ("small", "petit"), ("strong", "fort"), ("tired", "fatigue"),
        ("young", "jeune"), ("old", "vieux"), ("nice", "gentil"),
        ("brave", "courageux"),
    ]

    nouns = [
        ("a student", "un etudiant"), ("a teacher", "un professeur"),
        ("a doctor", "un docteur"), ("a friend", "un ami"),
        ("a singer", "un chanteur"), ("a painter", "un peintre"),
    ]

    like_en = ["like", "likes", "likes", "like", "like"]
    like_fr = ["aime", "aime", "aime", "aimons", "aiment"]

    animals = [
        ("cats", "les chats"), ("dogs", "les chiens"),
        ("birds", "les oiseaux"), ("horses", "les chevaux"),
        ("fish", "les poissons"),
    ]

    det_nouns = [
        ("the cat", "le chat"), ("the dog", "le chien"),
        ("the house", "la maison"), ("the car", "la voiture"),
        ("the book", "le livre"), ("the bird", "le oiseau"),
        ("the horse", "le cheval"), ("the tree", "le arbre"),
    ]

    det_adjectives = [
        ("big", "grand"), ("small", "petit"), ("black", "noir"),
        ("red", "rouge"), ("old", "vieux"), ("new", "nouveau"),
        ("good", "bon"), ("bad", "mauvais"),
    ]

    pairs = []

    # Pattern 1: subject + be + adjective  (e.g. "i am happy" -> "je suis heureux")
    for si in range(len(subjects_en)):
        for adj_en, adj_fr in adjectives:
            pairs.append((
                f"{subjects_en[si]} {be_en[si]} {adj_en}",
                f"{subjects_fr[si]} {be_fr[si]} {adj_fr}"
            ))

    # Pattern 2: subject + be + noun  (e.g. "he is a teacher" -> "il est un professeur")
    for si in range(len(subjects_en)):
        for noun_en, noun_fr in nouns:
            pairs.append((
                f"{subjects_en[si]} {be_en[si]} {noun_en}",
                f"{subjects_fr[si]} {be_fr[si]} {noun_fr}"
            ))

    # Pattern 3: subject + like + animal  (e.g. "we like cats" -> "nous aimons les chats")
    for si in range(len(subjects_en)):
        for anim_en, anim_fr in animals:
            pairs.append((
                f"{subjects_en[si]} {like_en[si]} {anim_en}",
                f"{subjects_fr[si]} {like_fr[si]} {anim_fr}"
            ))

    # Pattern 4: det_noun + be + adjective  (e.g. "the cat is big" -> "le chat est grand")
    for dn_en, dn_fr in det_nouns:
        for adj_en, adj_fr in det_adjectives:
            pairs.append((
                f"{dn_en} is {adj_en}",
                f"{dn_fr} est {adj_fr}"
            ))

    # Shuffle with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    rng.shuffle(pairs)

    # Split into train and test
    split = int(0.85 * len(pairs))
    train_pairs = pairs[:split]
    test_pairs = pairs[split:]

    # Build vocabularies from ALL pairs (so test words are known)
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()

    for src, tgt in pairs:
        src_vocab.add_sentence(src)
        tgt_vocab.add_sentence(tgt)

    print(f"\nTotal sentence pairs: {len(pairs)}")
    print(f"  Training: {len(train_pairs)}")
    print(f"  Test:     {len(test_pairs)}")
    print(f"Source vocabulary size: {src_vocab.n_words}")
    print(f"Target vocabulary size: {tgt_vocab.n_words}")

    print(f"\nExample pairs:")
    for src, tgt in train_pairs[:5]:
        print(f"  EN: {src:30s} -> FR: {tgt}")

    # Convert to tensors
    def pairs_to_tensors(pair_list, src_vocab, tgt_vocab):
        max_src_len = max(len(s.split()) for s, _ in pair_list)
        max_tgt_len = max(len(t.split()) for _, t in pair_list)

        encoder_inputs = torch.zeros(len(pair_list), max_src_len, dtype=torch.long)
        decoder_inputs = torch.zeros(len(pair_list), max_tgt_len + 1, dtype=torch.long)
        decoder_targets = torch.zeros(len(pair_list), max_tgt_len + 1, dtype=torch.long)

        for i, (src, tgt) in enumerate(pair_list):
            src_ids = src_vocab.sentence_to_indices(src)
            tgt_ids = tgt_vocab.sentence_to_indices(tgt)

            encoder_inputs[i, :len(src_ids)] = torch.tensor(src_ids)

            decoder_inputs[i, 0] = SOS_IDX
            decoder_inputs[i, 1:len(tgt_ids)+1] = torch.tensor(tgt_ids)

            decoder_targets[i, :len(tgt_ids)] = torch.tensor(tgt_ids)
            decoder_targets[i, len(tgt_ids)] = EOS_IDX

        return encoder_inputs, decoder_inputs, decoder_targets

    train_tensors = pairs_to_tensors(train_pairs, src_vocab, tgt_vocab)
    test_tensors = pairs_to_tensors(test_pairs, src_vocab, tgt_vocab)

    print(f"\nTrain encoder input shape: {train_tensors[0].shape}")
    print(f"Test encoder input shape:  {test_tensors[0].shape}")

    return train_pairs, test_pairs, src_vocab, tgt_vocab, train_tensors, test_tensors


class TranslationAttention(nn.Module):
    """Bahdanau attention for translation."""

    def __init__(self, hidden_dim):
        super(TranslationAttention, self).__init__()
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        decoder_state = decoder_state.unsqueeze(1)
        scores = self.v(torch.tanh(
            self.W_s(decoder_state) + self.W_h(encoder_outputs)
        )).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class TranslationModel(nn.Module):
    """
    Complete seq2seq translation model with attention.

    Key design choices for good attention patterns:
    - Dropout on embeddings and decoder forces reliance on attention
    - Output projection uses BOTH decoder state and attention context
      so the attention pathway is directly used for word prediction
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, hidden_dim,
                 dropout=0.3):
        super(TranslationModel, self).__init__()

        # Encoder
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim,
                                          padding_idx=PAD_IDX)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Decoder
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim,
                                          padding_idx=PAD_IDX)
        self.attention = TranslationAttention(hidden_dim)
        self.decoder = nn.LSTM(embed_dim + hidden_dim, hidden_dim,
                               batch_first=True)
        # Output projection uses decoder state + attention context
        # This forces the model to route information through attention
        self.output_proj = nn.Linear(hidden_dim + hidden_dim, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.tgt_vocab_size = tgt_vocab_size

    def encode(self, src):
        embedded = self.dropout(self.src_embedding(src))
        outputs, hidden = self.encoder(embedded)
        return outputs, hidden

    def decode_step(self, x, hidden, encoder_outputs):
        embedded = self.dropout(self.tgt_embedding(x))  # (batch, 1, embed_dim)

        h_n = hidden[0].squeeze(0)
        context, attn_weights = self.attention(h_n, encoder_outputs)

        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, hidden = self.decoder(lstm_input, hidden)

        # Concatenate decoder output with attention context for prediction
        # This makes the attention context directly useful for word selection
        combined = torch.cat([output.squeeze(1), context], dim=-1)
        output = self.output_proj(combined).unsqueeze(1)

        return output, hidden, attn_weights

    def forward(self, src, tgt_input):
        encoder_outputs, hidden = self.encode(src)

        all_outputs = []
        all_attention = []

        for t in range(tgt_input.size(1)):
            x_t = tgt_input[:, t:t+1]
            output, hidden, attn = self.decode_step(x_t, hidden, encoder_outputs)
            all_outputs.append(output)
            all_attention.append(attn)

        outputs = torch.cat(all_outputs, dim=1)
        attention = torch.stack(all_attention, dim=1)

        return outputs, attention

    def translate(self, src, max_len=20):
        batch_size = src.size(0)
        encoder_outputs, hidden = self.encode(src)

        decoder_input = torch.full((batch_size, 1), SOS_IDX, dtype=torch.long)
        predictions = []
        attention_weights = []

        for _ in range(max_len):
            output, hidden, attn = self.decode_step(
                decoder_input, hidden, encoder_outputs
            )
            pred = output.argmax(dim=-1)
            predictions.append(pred)
            attention_weights.append(attn)
            decoder_input = pred

        predictions = torch.cat(predictions, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)

        return predictions, attention_weights


def compute_bleu(reference, hypothesis, max_n=4):
    """
    Compute a simplified BLEU score between reference and hypothesis.

    Args:
        reference: List of reference words
        hypothesis: List of hypothesis words
        max_n: Maximum n-gram size

    Returns:
        BLEU score (0 to 1)
    """
    if len(hypothesis) == 0:
        return 0.0

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(reference) / max(len(hypothesis), 1)))

    # N-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter()
        hyp_ngrams = Counter()

        for i in range(len(reference) - n + 1):
            ref_ngrams[tuple(reference[i:i+n])] += 1
        for i in range(len(hypothesis) - n + 1):
            hyp_ngrams[tuple(hypothesis[i:i+n])] += 1

        matches = 0
        total = 0
        for ngram, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
            total += count

        if total == 0:
            precisions.append(0)
        else:
            precisions.append(matches / total)

    # Geometric mean of precisions (with smoothing)
    log_avg = 0
    n_valid = 0
    for p in precisions:
        if p > 0:
            log_avg += np.log(p)
            n_valid += 1

    if n_valid == 0:
        return 0.0

    log_avg /= n_valid
    bleu = bp * np.exp(log_avg)

    return bleu


def train_translator(model, train_data, epochs=300, lr=0.005):
    """
    Train the translation model.
    """
    print("\n" + "=" * 70)
    print("Training Translation Model")
    print("=" * 70)

    enc_inp, dec_inp, dec_tgt = train_data

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 70)

    for epoch in range(epochs):
        model.train()

        output, _ = model(enc_inp, dec_inp)

        loss = criterion(
            output.view(-1, model.tgt_vocab_size),
            dec_tgt.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f}")

    print("-" * 70)
    print(f"Final Loss: {losses[-1]:.4f}")

    return losses


def evaluate_translator(model, pairs, src_vocab, tgt_vocab, data_tensors, label=""):
    """
    Evaluate the translator and display results.
    """
    print("\n" + "=" * 70)
    print(f"Translation Results {label}")
    print("=" * 70)

    enc_inp = data_tensors[0]

    model.eval()
    with torch.no_grad():
        predictions, attention = model.translate(enc_inp, max_len=8)

    bleu_scores = []

    n_show = min(20, len(pairs))
    print(f"\n{'English':>30s} | {'Target French':>30s} | "
          f"{'Predicted French':>30s} | {'BLEU':>6s}")
    print("-" * 105)

    for i in range(n_show):
        src_sent, tgt_sent = pairs[i]
        pred_indices = predictions[i].tolist()
        pred_sent = tgt_vocab.indices_to_sentence(pred_indices)

        ref_words = tgt_sent.split()
        hyp_words = pred_sent.split()
        bleu = compute_bleu(ref_words, hyp_words)
        bleu_scores.append(bleu)

        print(f"{src_sent:>30s} | {tgt_sent:>30s} | "
              f"{pred_sent:>30s} | {bleu:.3f}")

    # Compute BLEU for all pairs (not just displayed ones)
    for i in range(n_show, len(pairs)):
        pred_indices = predictions[i].tolist()
        pred_sent = tgt_vocab.indices_to_sentence(pred_indices)
        ref_words = pairs[i][1].split()
        hyp_words = pred_sent.split()
        bleu_scores.append(compute_bleu(ref_words, hyp_words))

    avg_bleu = np.mean(bleu_scores)
    print(f"\nAverage BLEU score ({len(pairs)} sentences): {avg_bleu:.3f}")

    return predictions, attention, bleu_scores


def visualize_translation_attention(model, pairs, src_vocab, tgt_vocab,
                                    data_tensors, n_examples=4):
    """
    Visualize attention alignments for translation.
    """
    print("\n" + "=" * 70)
    print("Visualizing Translation Attention Alignments")
    print("=" * 70)

    enc_inp = data_tensors[0]

    model.eval()
    with torch.no_grad():
        predictions, attention = model.translate(enc_inp, max_len=8)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx in range(min(n_examples, len(pairs))):
        src_sent, tgt_sent = pairs[idx]
        src_words = src_sent.split()
        pred_indices = predictions[idx].tolist()

        # Get predicted words (stop at EOS)
        pred_words = []
        for pidx in pred_indices:
            if pidx == EOS_IDX:
                break
            if pidx not in (PAD_IDX, SOS_IDX):
                word = tgt_vocab.idx2word.get(pidx, '<UNK>')
                pred_words.append(word)

        if len(pred_words) == 0:
            pred_words = ['<empty>']

        # Get attention weights
        attn = attention[idx, :len(pred_words), :len(src_words)].numpy()

        im = axes[idx].imshow(attn, cmap='YlOrRd', aspect='auto',
                              vmin=0, vmax=1)
        axes[idx].set_xticks(range(len(src_words)))
        axes[idx].set_xticklabels(src_words, fontsize=10, rotation=45)
        axes[idx].set_yticks(range(len(pred_words)))
        axes[idx].set_yticklabels(pred_words, fontsize=10)
        axes[idx].set_xlabel('English (source)', fontsize=11)
        axes[idx].set_ylabel('French (predicted)', fontsize=11)
        axes[idx].set_title(f'"{src_sent}"', fontsize=11)
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.suptitle('Translation Attention Alignments\n'
                 '(Bright = high attention on source word)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "translation_attention.png"),
                dpi=150, bbox_inches="tight")
    print("\nAttention alignments saved to 'translation_attention.png'")

    print("\nKey observations:")
    print("  - Attention reveals word-level alignments between languages")
    print("  - 'i' aligns with 'je', 'cat' with 'chat', etc.")
    print("  - Word order differences are captured by attention pattern")


def demonstrate_translation():
    """
    Main demonstration of seq2seq translation.
    """
    # Explain pipeline
    explain_translation_pipeline()

    torch.manual_seed(42)
    np.random.seed(42)

    # Create dataset
    (train_pairs, test_pairs, src_vocab, tgt_vocab,
     train_tensors, test_tensors) = create_translation_dataset()

    # Configuration - keep model small so it must rely on attention
    embed_dim = 64
    hidden_dim = 64
    dropout = 0.3

    print(f"\nModel configuration:")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Dropout: {dropout}")
    print(f"  Source vocab: {src_vocab.n_words} words")
    print(f"  Target vocab: {tgt_vocab.n_words} words")

    # Create model
    model = TranslationModel(
        src_vocab_size=src_vocab.n_words,
        tgt_vocab_size=tgt_vocab.n_words,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Train
    losses = train_translator(model, train_tensors, epochs=500, lr=0.005)

    # Evaluate on train set
    _, _, train_bleu = evaluate_translator(
        model, train_pairs, src_vocab, tgt_vocab, train_tensors,
        label="(Train Set)"
    )

    # Evaluate on test set (unseen sentences!)
    _, _, test_bleu = evaluate_translator(
        model, test_pairs, src_vocab, tgt_vocab, test_tensors,
        label="(Test Set - Unseen Sentences!)"
    )

    # Visualize attention on test set (more meaningful than train)
    visualize_translation_attention(
        model, test_pairs, src_vocab, tgt_vocab, test_tensors, n_examples=4
    )

    # Plot training loss and BLEU scores
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(losses, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12)
    axes[0].set_title('Translation Training Loss', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    x = np.arange(len(test_bleu))
    axes[1].bar(x, test_bleu, color='green', alpha=0.7)
    axes[1].set_xlabel('Test Sentence Index', fontsize=12)
    axes[1].set_ylabel('BLEU Score', fontsize=12)
    axes[1].set_title('Test Set Per-Sentence BLEU Scores', fontsize=14)
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(y=np.mean(test_bleu), color='red', linestyle='--',
                    label=f'Mean: {np.mean(test_bleu):.3f}')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "translation_results.png"),
                dpi=150, bbox_inches="tight")
    print("\nTraining and BLEU visualization saved to 'translation_results.png'")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key takeaways:
  1. Neural translation requires: tokenization, vocabulary, encoder-decoder
  2. Attention is essential for aligning source and target words
  3. Teacher forcing stabilizes training by providing ground truth inputs
  4. BLEU score measures translation quality via n-gram overlap
  5. Attention weights reveal interpretable word alignments
  6. The model generalizes to unseen word combinations (test set)

Limitations of this approach:
  - Sequential processing (cannot parallelize across time steps)
  - Still struggles with very long sequences
  - Limited vocabulary handling (no subword tokenization)
  - Greedy decoding is suboptimal (beam search is better)

What comes next:
  - Transformers replace RNNs with self-attention (fully parallel!)
  - Subword tokenization (BPE) handles rare words
  - Pre-training on large corpora (BERT, GPT, etc.)
""")


if __name__ == "__main__":
    demonstrate_translation()
