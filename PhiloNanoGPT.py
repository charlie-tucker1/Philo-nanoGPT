import urllib.request
import zipfile
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import io

# Fix UTF-8 encoding for Windows output console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Hyperparameters
batch_size = 32
block_size = 192
max_iters = 12000
eval_interval = 1500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 288
dropout = 0.2
n_layers = 5
n_heads = 6

print(device)

# Download data
os.makedirs("data", exist_ok=True)
urllib.request.urlretrieve("https://philosophydata.com/phil_nlp.zip", "data/phil.zip")

with zipfile.ZipFile("data/phil.zip", 'r') as z:
    z.extractall("data/philosophy")

os.remove("data/phil.zip")

# Load philosophy corpus
df = pd.read_csv("data/philosophy/philosophy_data.csv", low_memory=False)
text = "\n".join(df['sentence_str'].astype(str))
print(f"Loaded {len(text):,} characters")

# Create character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")

# Create encoder and decoder for char-int mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode the entire text data to a Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Data shape: {data.shape}, dtype: {data.dtype}")

# Split the data into train-validation sets
n = int(.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # Generates a small batch of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a relu """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: 'communication followed by computation' """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # each token reads off logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensors of ints
        tok_emb = self.token_embedding_table(idx) # (Batch, Time, Channel)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)  # Reshape for BCE loss
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] # (B, block_size)
            # get predictions
            logits, loss = self(idx_cond)
            # focus on only the last time step
            logits = logits[:, -1, :] # becomes (B, C), plucks out the last logits value
            # softmax for probabilities:
            probs = F.softmax(logits, dim=-1) # (B,C), Softmax to find next token based solely on the last logit
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1), Gives us a simple prediction
            # append sample index to our running sequence
            idx = torch.cat((idx, idx_next), dim=1) # Creates (B, T+1)
        return idx


# Initialize model
model = BigramLanguageModel()
model = model.to(device)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    # Evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Get batch
    xb, yb = get_batch('train')

    # Evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#Save the model to .pth
torch.save(model.state_dict(), 'philo_model.pth')
print("\nModel saved to philo_model.pth")

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\nGenerated text:")
print(decode(model.generate(context, max_new_tokens=3000)[0].tolist()))
