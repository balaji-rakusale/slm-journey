import torch
import torch.nn as nn
import torch.nn.functional as F

# BUILDING A TINY GPT FROM SCRATCH
# Every concept from Day 1-4 comes together here

# Hyperparameters
batch_size = 4        # how many sentences we train on at once
block_size = 8        # how many characters we look back at
embedding_dim = 32    # size of each token embedding
num_heads = 4         # number of attention heads
num_layers = 3        # number of transformer blocks
dropout = 0.1
#Experiment 2 — Change Learning Rate
learning_rate = 1e-3  # original - sweet spot
#learning_rate = 1e-2   # too high - loss explodes?
#learning_rate = 1e-4   # too low - loss drops slowly?

#Experiment 3 — Change Model Size
# Bigger model
#embedding_dim = 64
#num_heads = 8
#num_layers = 6
# Smaller model
#embedding_dim = 16
#num_heads = 2
#num_layers = 2
#Experiment 4 — More Iterations
max_iters = 3000  # train longer - does loss keep dropping?
#max_iters = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

# Step 1 - Load and prepare data
#text = """
#To be or not to be that is the question
#Whether tis nobler in the mind to suffer
#The slings and arrows of outrageous fortune
#Or to take arms against a sea of troubles
#"""

#Experiment 1 — More Training Data

text = """
In the quiet village by the river, people lived simple lives filled with routine and tradition. Every morning, the sun rose gently over the hills, casting a golden glow across the fields where farmers worked tirelessly to grow their crops. Children would run through narrow paths, laughing and chasing each other, while elders sat beneath ancient trees sharing stories of the past.
As seasons changed, so did the rhythm of life. The monsoon brought heavy rains that nourished the land, while winter wrapped the village in a calm and peaceful silence. During festivals, the entire community gathered together, lighting lamps, singing songs, and celebrating with joy that echoed through every home.
Beyond the village, the world was vast and unknown, filled with opportunities and challenges alike. Travelers passing through would often speak of distant cities, towering buildings, and machines that could perform tasks beyond imagination. These stories sparked curiosity in the minds of the young, inspiring dreams of exploration and discovery.
Yet, despite the allure of the unknown, there was comfort in familiarity. The bonds between people, the shared struggles, and the small moments of happiness created a sense of belonging that was difficult to replace. Life moved forward steadily, shaped by both change and continuity, reminding everyone that even the simplest existence holds deep meaning.
"""

# Build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary: {''.join(chars)}")
print(f"Vocab size: {vocab_size}")

# Tokenizer
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [char_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_char[i] for i in l])

# Encode the text
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Data shape: {data.shape}")

# Split into train and validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Step 2 - Data loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Step 3 - Self Attention Head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

# Step 4 - Multi Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

# Step 5 - FeedForward
class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Step 6 - Transformer Block
class Block(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ff = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# Step 7 - Full GPT Model
class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Step 8 - Train
model = NanoGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())/1e3:.1f}K")
print("\nTraining started...")

for i in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"Step {i}: loss = {loss.item():.4f}")

# Step 9 - Generate text
print("\nGenerating text...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = decode(model.generate(context, max_new_tokens=200)[0].tolist())
print("Generated text:")
print(generated)