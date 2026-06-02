import torch
import torch.nn as nn
import json

print("=" * 50)
print("PART 1 - Why LoRA Exists")
print("=" * 50)

# Full fine tuning problem
# GPT2  = 117 Million parameters
# Llama = 7 Billion parameters
# Full fine tune = update ALL parameters = expensive

# LoRA solution
# Instead of updating W directly
# Learn two small matrices A and B
# W_new = W_original + (A x B)
# A and B have far fewer parameters

print("Full Fine Tuning vs LoRA:")
print()

# Simulate a weight matrix (like in a real transformer)
d_model = 768  # GPT2 hidden size

# Original weight matrix
W = torch.randn(d_model, d_model)
full_params = d_model * d_model
print(f"Original weight matrix: {d_model} x {d_model}")
print(f"Full fine tune params:  {full_params:,}")

# LoRA matrices with rank r
r = 8  # LoRA rank - the key hyperparameter
A = torch.randn(d_model, r)
B = torch.randn(r, d_model)
lora_params = (d_model * r) + (r * d_model)
print(f"\nLoRA rank r = {r}")
print(f"Matrix A: {d_model} x {r} = {d_model * r:,} params")
print(f"Matrix B: {r} x {d_model} = {r * d_model:,} params")
print(f"LoRA total params: {lora_params:,}")
print(f"\nParameter reduction: {full_params/lora_params:.0f}x fewer parameters!")

print("\n" + "=" * 50)
print("PART 2 - LoRA Forward Pass")
print("=" * 50)

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()

        # Original frozen weight
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features),
            requires_grad=False  # FROZEN - not updated
        )

        # LoRA matrices - only these get trained
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
        )

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

    def forward(self, x):
        # Original frozen output
        original_output = x @ self.weight.T

        # LoRA adaptation
        lora_output = x @ self.lora_A.T @ self.lora_B.T

        # Combine
        return original_output + self.scaling * lora_output

# Test LoRA layer
batch_size = 4
seq_len = 8
hidden_size = 64
rank = 4

lora_layer = LoRALayer(hidden_size, hidden_size, rank=rank)
x = torch.randn(batch_size, seq_len, hidden_size)
output = lora_layer(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")

# Count trainable vs frozen params
total_params = sum(p.numel() for p in lora_layer.parameters())
trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f"\nTotal parameters:     {total_params:,}")
print(f"Frozen parameters:    {frozen_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable %:          {trainable_params/total_params*100:.1f}%")

print("\n" + "=" * 50)
print("PART 3 - LoRA Hyperparameters Explained")
print("=" * 50)

lora_configs = [
    {"rank": 4,  "alpha": 8,  "desc": "Very efficient - small tasks"},
    {"rank": 8,  "alpha": 16, "desc": "Standard - most use cases"},
    {"rank": 16, "alpha": 32, "desc": "More capacity - complex tasks"},
    {"rank": 32, "alpha": 64, "desc": "High capacity - domain shift"},
    {"rank": 64, "alpha": 128,"desc": "Near full fine tune quality"},
]

d = 768
print(f"Base model dimension: {d}")
print(f"\n{'Rank':<6} {'Alpha':<6} {'Params':<12} {'vs Full':<12} {'Use Case'}")
print("-" * 65)
for config in lora_configs:
    r = config['rank']
    params = 2 * d * r
    ratio = (d * d) / params
    print(f"{r:<6} {config['alpha']:<6} {params:<12,} {ratio:<12.0f}x {config['desc']}")

print("\n" + "=" * 50)
print("PART 4 - Where LoRA Is Applied")
print("=" * 50)

# In transformers LoRA is applied to attention matrices
attention_matrices = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
ffn_matrices = ['gate_proj', 'up_proj', 'down_proj']

print("Attention matrices (most common LoRA targets):")
for m in attention_matrices:
    print(f"  → {m}")

print("\nFFN matrices (optional additional targets):")
for m in ffn_matrices:
    print(f"  → {m}")

print("\nMore target matrices = more trainable params = better quality but slower")

print("\n" + "=" * 50)
print("PART 5 - Load Our Dataset and Preview")
print("=" * 50)

with open('fine_tune_ready.json', 'r') as f:
    dataset = json.load(f)

print(f"Dataset loaded: {len(dataset)} samples")
print(f"\nFirst training sample:")
print(dataset[0]['text'])
print(f"\nThis dataset will feed into LoRA fine tuning tomorrow!")

print("\n" + "=" * 50)
print("LORA CHEAT SHEET")
print("=" * 50)
print("""
Key Parameters:
  r (rank)     → controls capacity (8-64)
  alpha        → scaling factor (usually 2x rank)
  target_modules → which layers to adapt
  dropout      → regularization (0.05-0.1)

Rules of Thumb:
  Small dataset  → lower rank (4-8)
  Large dataset  → higher rank (16-64)
  Simple task    → lower rank
  Complex task   → higher rank
  Always set alpha = 2 x rank
""")