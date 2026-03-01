# Lecture 14: Sequence-to-Sequence Models

This lecture covers sequence-to-sequence (seq2seq) architectures, progressing from basic RNN sequence modeling to encoder-decoder models with attention mechanisms.

## Topics

### 1. Sequence Modeling with RNNs (`01_sequence_modeling.py`)
- Recurrent Neural Networks (RNN) and why we need them
- Hidden state as memory
- Simple RNN vs LSTM vs GRU
- Processing variable-length sequences
- Sequence classification example (sentiment-like task)

### 2. Seq2Seq Encoder-Decoder (`02_seq2seq_encoder_decoder.py`)
- The seq2seq architecture: encoder RNN + decoder RNN
- Context vector as information bottleneck
- Teacher forcing during training
- Greedy decoding at inference
- Example: learning to reverse sequences

### 3. Attention Mechanism (`03_attention_mechanism.py`)
- The bottleneck problem with fixed context vector
- Attention: letting the decoder "look back" at encoder states
- Bahdanau (additive) attention
- Attention weights visualization
- Example: sorting sequences with attention

### 4. Seq2Seq Translation (`04_seq2seq_translation.py`)
- Putting it all together: a simple word-level translator
- Vocabulary building and tokenization
- Encoder-decoder with attention for translation
- BLEU score evaluation
- Visualizing attention alignments

## Running the Code

Each file can be run independently:

```bash
cd lec14-seq2seq
python 01_sequence_modeling.py
python 02_seq2seq_encoder_decoder.py
python 03_attention_mechanism.py
python 04_seq2seq_translation.py
```

## Key Concepts

### RNN (Recurrent Neural Network)
```
x₁ → [RNN] → h₁
       ↓
x₂ → [RNN] → h₂
       ↓
x₃ → [RNN] → h₃ → output
```
- Processes sequences step by step
- Hidden state carries information across time steps
- LSTM/GRU solve the vanishing gradient problem

### Seq2Seq (Sequence-to-Sequence)
```
Encoder:  x₁,x₂,...,xₙ → [Encoder RNN] → context vector (c)
Decoder:  c → [Decoder RNN] → y₁,y₂,...,yₘ
```
- Input and output sequences can have different lengths
- Context vector summarizes the entire input
- Teacher forcing: feed ground truth as decoder input during training

### Attention Mechanism
```
Encoder:  x₁,x₂,...,xₙ → h₁,h₂,...,hₙ  (all hidden states)
                              ↓
Decoder at step t:  αₜ = softmax(score(sₜ, h₁...hₙ))
                    cₜ = Σ αₜᵢ · hᵢ  (weighted sum)
                    yₜ = f(sₜ, cₜ)
```
- Decoder attends to different parts of the input at each step
- Attention weights show which input tokens are relevant
- Solves the information bottleneck of fixed-size context vector

## Generated Visualizations

Running the code will generate various PNG files showing:
- RNN hidden state evolution
- Training loss curves
- Attention weight heatmaps
- Translation attention alignments
