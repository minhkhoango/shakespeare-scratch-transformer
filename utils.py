import torch
import re
from nltk import word_tokenize
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    return text

def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[“”‘’]', "'", text)
    text = re.sub(r'[^\w\s\'-]', '', text)
    return text

def tokenize_and_build_vocab(text, min_freq=1):
    # 1.clean + tokenize
    text = clean_text(text)
    tokens = [t.lower() for t in word_tokenize(text, language="english")]

    # 2.frequency count
    freq = Counter(tokens)

    # 3.filter + order by freq desc, then alpha fr ties
    vocab = sorted(
        (w for w, c in freq.items() if c >= min_freq),
        key=lambda w: (-freq[w], w) 
        # high count first, then a-z for stability
        # default of sorting is increasing
    )

    # 4.Specials
    SPECIALS = ['<pad>', '<sos>', '<eos>', '<unk>']
    word2idx = {tok : i for i, tok in enumerate(SPECIALS, start=0)}

    # 5. Populate vocab
    for w in vocab:
        if w not in word2idx:
            word2idx[w] = len(word2idx) # len increasing
    
    # 6. reverse map
    idx2word = {i: w for w, i in word2idx.items()}
    return tokens, word2idx, idx2word

def train_eval_split(tokens, word2idx, seq_len=32, 
                     batch_size=256, train_split=0.95):
    unk_idx = word2idx['<unk>']
    windows = []
    for i in range(0, len(tokens) - (seq_len-2)):
        seq = ['<sos>'] + tokens[: i + (seq_len-2)] + ['<eos>']
        windows.append([word2idx.get(t, unk_idx) for t in seq])
    X = torch.tensor(windows, dtype=torch.long, device=device)

    split_idx = int(len(X) * train_split)
    train_loader = DataLoader(TensorDataset(X[:split_idx]),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[split_idx:]),
                            batch_size=batch_size)
    print(train_loader)
    return train_loader, val_loader

def get_transformer_scheduler(optimizer, d_model, warmup_steps=1000):
    def lr_lambda(step):
        step += 1 # avoid zero division
        return (d_model**-0.5) * min((step+1)**-0.5, (step+1)*warmup_steps**-1.5)
    return torch.optim.lr_scheduler(optimizer, lr_lambda)

def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, device):
    model.train()
    total  = 0.0
    for batch in loader:
        batch_x = batch[0].to(device)
        x = batch_x[:, :-1]
        y = batch_x[:, 1:]

        optimizer.zero_grad()
        with autocast('cuda'):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.reshape(-1))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scheduler.step()
        total += loss.item()
    return total / len(loader)

def generate(model, prompt_ids, max_new_tokens=200, temperature=1.2, top_k=40,
             top_p=0.9, min_len=20, eos_id=2):
    device = next(model.parameters()).device
    model.eval()

    generated = prompt_ids.clone().to(device)

    for step in range(max_new_tokens):
        # Ensure model input is on the same device
        input_ids = generated[:, -model.pos_enc.pe.size(1):].to(device)
        logits = model(input_ids) # (1, t, vocab)

        next_token_logits = logits[:, -1, :] / temperature
        if step < min_len:
            next_token_logits[:, eos_id] = -1e4 # block early <eos>
        
        if top_k is not None:
            topk_vals, _ = torch.topk(next_token_logits, top_k)
            next_token_logits[next_token_logits < topk_vals[:, [-1]]] = -1e4
        
        # top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            mask = cumulative_probs > top_p
            if mask[:, 0].any():
                mask[:, 0] = False
            
            sorted_logits[mask] = -1e4
            next_token_logits = torch.zeros_like(next_token_logits).scatter(
                1, sorted_idx, sorted_logits)
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)

            if next_id.item() == eos_id:
                break
    return generated.squeeze(0).tolist()

def encode_prompt(prompt, word2idx):
    prompt = clean_text(prompt)
    tokens = word_tokenize(prompt, language='english')
    tokens = [t.lower() for t in tokens]
    encoded = [word2idx.get(t, word2idx['<unk>']) for t in tokens]
    return torch.tensor([encoded], device=device), tokens

