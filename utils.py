import numpy as np
import re
from nltk import word_tokenize
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import math
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def clean_text(text): 
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[“”‘’]', "'", text)
    text = re.sub(r'[^\w\s\'-]', '', text)
    return text

def tokenize_and_build_vocab(text, min_freq=1):
    """
    Cleans, tokenizes, and builds a word‑to‑index mapping
    ordered by descending frequency (ties -> alphabetical).

    Returns
    -------
    tokens   : list[str]         # full corpus tokens
    word2idx : dict[str, int]    # token → id
    idx2word : dict[int, str]    # id   → token
    """
    # 1. clean + tokenize
    text = clean_text(text)
    tokens = [t.lower() for t in word_tokenize(text, language="english")]

    # 2. frequency count
    freq = Counter(tokens)

    # 3. filter + order by freq desc, then alpha for ties
    vocab = sorted(
        (w for w, c in freq.items() if c >= min_freq),
        key=lambda w: (-freq[w], w)   # high count first, then a‑z for stability
    )

    # 4. specials
    SPECIALS = ['<pad>', '<sos>', '<eos>', '<unk>']
    word2idx = {tok: i for i, tok in enumerate(SPECIALS, start=0)}

    # 5. populate vocab
    for w in vocab:
        if w not in word2idx:          # skip if it happens to be in SPECIALS
            word2idx[w] = len(word2idx)

    # 6. reverse map
    idx2word = {i: w for w, i in word2idx.items()}

    return tokens, word2idx, idx2word


def train_eval_split(tokens, word2idx, seq_len=32, batch_size=256, train_split=0.95):
    unk_idx  = word2idx['<unk>']
    windows = []
    for i in range(0, len(tokens) - (seq_len - 2)):
        seq = ['<sos>'] + tokens[i : i + (seq_len - 2)] + ['<eos>']
        windows.append([word2idx.get(t, unk_idx) for t in seq])
    X = torch.tensor(windows, dtype=torch.long).to(device)

    split_idx = int(len(X) * train_split)
    train_dataset = TensorDataset(X[:split_idx])
    val_dataset   = TensorDataset(X[split_idx:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
 
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0
    for batch in loader:
        batch_x = batch[0].to(device)
        x = batch_x[:, :-1]
        y = batch_x[:, 1:]

        optimizer.zero_grad()
        logits = model(x)
        assert not torch.any(torch.isnan(logits))
        assert not torch.any(torch.isinf(logits))
        loss = criterion(logits.view(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def generate(model, prompt_ids, max_new_tokens=100,
             temperature=1.2, top_k=0, top_p=0.9,
             min_len=5, eos_id=2):

    device = next(model.parameters()).device
    model.eval()
    
    generated = prompt_ids.clone().to(device)

    for step in range(max_new_tokens):
        # Ensure model input is on the same device
        input_ids = generated[:, -model.pos_enc.pe.size(1):].to(device)
        logits = model(input_ids)  # (1, t, vocab)

        next_token_logits = logits[:, -1, :] / temperature

        if step < min_len:
            next_token_logits[:, eos_id] = -1e9  # block early <eos>
            # top-k filtering
        if top_k is not None:
            topk_vals, _ = torch.topk(next_token_logits, top_k)
            next_token_logits[
                next_token_logits < topk_vals[:, [-1]]
            ] = -1e9
        
        # top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            mask = cumulative_probs > top_p
            if mask[:, 0].any():
                mask[:, 0] = False  # always keep at least 1 token

            sorted_logits[mask] = -1e9
            next_token_logits = torch.zeros_like(next_token_logits).scatter(1, sorted_idx, sorted_logits)

        probs = torch.softmax(next_token_logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_id], dim=1)

        if next_id.item() == eos_id:
            break

    return generated.squeeze(0).tolist()

def encode_prompt(prompt, word2idx):
    prompt = clean_text(prompt)
    tokens = word_tokenize(prompt, language="english")
    tokens = [t.lower() for t in tokens]
    encoded = [word2idx.get(t, word2idx['<unk>']) for t in tokens]
    return torch.tensor([encoded], device=device), tokens

