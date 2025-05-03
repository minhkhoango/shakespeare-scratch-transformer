# shakespeare-scratch-transformer
# 🧠 Shakespeare Text Generator - Transformer Edition

This project implements a decoder-only Transformer model from scratch using PyTorch, trained to generate text in the style of Shakespeare. It's a clean, minimal foundation designed for educational clarity and future extensions.

---

## 📜 Description

This repository includes:

- A custom implementation of a decoder-only Transformer (no encoder stack).
- Training on a corpus of Shakespeare's plays (~5.5MB of data).
- Tokenizer, vocabulary builder, training loop, and generation function.
- Techniques like top-k and top-p sampling, temperature scaling, and padding/masking for variable-length sequences.

---

## 🏗️ Architecture Details

- **Transformer Type**: Decoder-only
- **Layers**: 6 decoder layers
- **Embedding Size**: 128 (`d_model`)
- **Feedforward Dim**: 512 (`d_ff`)
- **Heads**: 8 (`n_heads`)
- **Positional Encoding**: Sinusoidal (static)
- **Sequence Length**: 256 max
- **Special Tokens**: `<pad>`, `<sos>`, `<eos>`, `<unk>`

---

## 🧪 Results

Sample output (early epoch):
what thee do <unk> ' here here was

Still a work in progress — model quality improves with longer training, better sampling, and tuned hyperparameters.

---

## 🚀 How to Run

### 1. Install Dependencies

## How to run
pip install torch nltk

## Run training
python train.py

## Generate text
python generate.py --prompt "To be"

## Config example
config = {
    "vocab_size": len(word2idx),
    "d_model": 128,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 512,
    "max_seq_len": 256,
    "dropout": 0.1
}

🧠 Why This Project?
To really learn how Transformers work — no shortcuts, no black boxes. This codebase is a practical exercise in understanding self-attention, positional encoding, and sequence generation the hard (but rewarding) way.

🙌 Acknowledgements
Attention Is All You Need (Vaswani et al.)

Karpathy’s nanoGPT + transformer breakdowns

Andrew Ng’s Deep Learning Specialization

The 3:50 AM club

💬 Contact
Built by Khoa.
Feel free to fork, experiment, or message for collab!
