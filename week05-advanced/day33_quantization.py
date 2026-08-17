import torch
import torch.nn as nn
import time
import os

print("=" * 50)
print("PART 1 - Why Quantization Matters")
print("=" * 50)

print("""
Model Size vs Precision:

Llama 3 8B:
  float32 → 32 GB  (impossible on laptop)
  float16 → 16 GB  (needs expensive GPU)
  int8    →  8 GB  (runs on free Colab)
  int4    →  4 GB  (runs on laptop GPU)
  int2    →  2 GB  (experimental)

Quality vs Size tradeoff:
  float32 → perfect quality  (baseline)
  float16 → 99.9% quality   (no visible difference)
  int8    → 99.5% quality   (barely noticeable)
  int4    → 99.0% quality   (slight degradation)
  int2    → 95.0% quality   (noticeable degradation)

Sweet spot for enterprise: int4 or int8
Best balance of quality and cost.
""")

print("=" * 50)
print("PART 2 - Quantization Types")
print("=" * 50)

quant_types = {
    "Post Training Quantization (PTQ)": {
        "when": "After training is complete",
        "data_needed": "Small calibration dataset",
        "speed": "Fast — minutes",
        "quality_loss": "Small",
        "tools": "GPTQ, AWQ, llama.cpp",
        "use_case": "Production deployment"
    },
    "Quantization Aware Training (QAT)": {
        "when": "During training",
        "data_needed": "Full training dataset",
        "speed": "Slow — hours",
        "quality_loss": "Minimal",
        "tools": "PyTorch native, BitsAndBytes",
        "use_case": "Highest quality needed"
    },
    "Dynamic Quantization": {
        "when": "At inference time",
        "data_needed": "None",
        "speed": "Instant",
        "quality_loss": "Medium",
        "tools": "PyTorch torch.quantization",
        "use_case": "Quick deployment"
    },
}

for method, details in quant_types.items():
    print(f"\n{method}:")
    for key, value in details.items():
        print(f"  {key}: {value}")

print("\n" + "=" * 50)
print("PART 3 - Implement Dynamic Quantization")
print("=" * 50)

class SimpleTransformerLayer(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.ff1 = nn.Linear(dim, dim * 4)
        self.ff2 = nn.Linear(dim * 4, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff2(torch.relu(self.ff1(x)))
        x = self.norm2(x + ff_out)
        return x

# Create model
model = SimpleTransformerLayer(dim=256)
model.eval()

# Measure original size
def get_model_size(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / 1e6

original_size = get_model_size(model)
print(f"Original model size: {original_size:.2f} MB")
print(f"Original dtype: float32")

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

quantized_size = get_model_size(quantized_model)
print(f"\nQuantized model size: {quantized_size:.2f} MB")
print(f"Quantized dtype: int8")
print(f"Size reduction: {original_size/quantized_size:.1f}x smaller")

print("\n" + "=" * 50)
print("PART 4 - Speed and Quality Test")
print("=" * 50)

batch_size = 4
seq_len = 32
dim = 256
x = torch.randn(batch_size, seq_len, dim)

runs = 10

# Original model speed
start = time.time()
with torch.no_grad():
    for _ in range(runs):
        out_original = model(x)
original_time = (time.time() - start) / runs

# Quantized model speed
start = time.time()
with torch.no_grad():
    for _ in range(runs):
        out_quantized = quantized_model(x)
quantized_time = (time.time() - start) / runs

print(f"Original model:  {original_time*1000:.2f} ms per forward pass")
print(f"Quantized model: {quantized_time*1000:.2f} ms per forward pass")
print(f"Speed improvement: {original_time/quantized_time:.2f}x faster")

# Quality check
max_diff = (out_original - out_quantized).abs().max().item()
mean_diff = (out_original - out_quantized).abs().mean().item()
print(f"\nMax output difference:  {max_diff:.6f}")
print(f"Mean output difference: {mean_diff:.6f}")
print(f"Quality preserved: {'✅ Yes' if max_diff < 0.1 else '⚠️ Check needed'}")

print("\n" + "=" * 50)
print("PART 5 - GGUF Format for Deployment")
print("=" * 50)

print("""
GGUF = GPT Generated Unified Format
Used by llama.cpp for CPU inference

Why GGUF matters:
  → Run 7B models on regular laptops
  → No GPU needed for inference
  → Perfect for enterprise on-premise deployment
  → Client doesn't need expensive hardware

GGUF quantization levels:
  Q2_K  → smallest, lowest quality
  Q4_K  → recommended for most use cases
  Q5_K  → better quality, slightly larger
  Q8_0  → near perfect quality, 8x smaller than f32
  F16   → half precision, best quality

For enterprise clients:
  Q4_K_M = best balance
  Runs 7B model on 8GB RAM laptop
  Client can use on their own machine
  No cloud costs, no data privacy concerns

This is your biggest selling point:
"Your data never leaves your laptop"
""")

print("\n" + "=" * 50)
print("PART 6 - Quantization Formats Comparison")
print("=" * 50)

formats = [
    {"format": "float32", "size_7B": "28 GB", "quality": "100%", "runs_on": "High-end GPU only"},
    {"format": "float16", "size_7B": "14 GB", "quality": "99.9%", "runs_on": "A100, 3090"},
    {"format": "int8 (GPTQ)", "size_7B": "7 GB", "quality": "99.5%", "runs_on": "T4, 3080"},
    {"format": "Q4_K_M (GGUF)", "size_7B": "4 GB", "quality": "99%", "runs_on": "Any 8GB GPU/RAM"},
    {"format": "Q2_K (GGUF)", "size_7B": "2.5 GB", "quality": "95%", "runs_on": "Any laptop"},
]

print(f"{'Format':<20} {'7B Size':<12} {'Quality':<10} {'Runs On'}")
print("-" * 65)
for f in formats:
    print(f"{f['format']:<20} {f['size_7B']:<12} {f['quality']:<10} {f['runs_on']}")

print("\n" + "=" * 50)
print("PART 7 - Enterprise Deployment Decision Tree")
print("=" * 50)

print("""
How to choose quantization for client:

Client has GPU?
  YES → Use float16 or int8 (GPTQ)
  NO  → Use Q4_K_M (GGUF) with llama.cpp

Client data is sensitive?
  YES → Deploy on-premise with GGUF
  NO  → Deploy on cloud with float16

Client needs best quality?
  YES → float16 on A100
  NO  → Q4_K_M on laptop

Client has budget?
  HIGH   → A100 GPU server + float16
  MEDIUM → T4 GPU + int8
  LOW    → Laptop + GGUF Q4_K_M

Always start with Q4_K_M for demos.
Upgrade based on client feedback.
""")

print("Day 33 Complete ✅")
print("Tomorrow: llama.cpp + running models locally")