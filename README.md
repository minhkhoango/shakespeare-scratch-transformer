# Shakespeare Transformer from Scratch

This project is a decoder-only Transformer implemented from scratch using PyTorch. It was trained to generate text in Shakespearean style, using a cleaned corpus of Shakespeare's works.

## 🔥 Highlights

- Manual Transformer architecture (no `nn.Transformer`)
- Custom tokenizer and vocabulary
- Training from scratch on cloud GPUs (A4000 / H100)
- Implements:
  - Positional Encoding
  - Multi-head Attention
  - Decoder-only architecture
  - Top-k and Top-p sampling
  - Mixed-precision training
  - Custom learning rate scheduler (warm-up + decay)

## 🧠 Model Specs

- `d_model`: 128  
- `n_heads`: 8  
- `n_layers`: 6  
- `d_ff`: 512  
- `max_seq_len`: 256  
- `dropout`: 0.1  
- `vocab_size`: Based on filtered Shakespeare corpus

## 🏋️‍♂️ Training Info

- Batch size: 4500  
- Trained on: H100 (via vast.ai)  
- Epochs: 50
- Final training loss: ~2.3

## 📂 Files

- `transformer.py` – Model architecture
- `train.py` – Training loop
- `generate.py` – Text generation logic
- `vocab.pkl` – Vocabulary mapping
- `config.pkl` – Model configuration
- `transformer_state.pth` – Trained model weights (see below)

## 🛠 Usage

```bash
# Load model
with open("config.pkl", "rb") as f:
    config = pickle.load(f)

model = GPTLike(**config)
model.load_state_dict(torch.load("transformer_state.pth", map_location=device))
model.eval()
