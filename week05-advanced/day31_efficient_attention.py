import torch
import time
import math

print("=" * 50)
print("PART 1 - Standard Attention Problem")
print("=" * 50)

print("""
Standard Attention Memory Problem:

Sequence length N = 1000 tokens
Attention matrix = N x N = 1000 x 1000
Memory needed   = 1,000,000 values

Sequence length N = 10,000 tokens
Attention matrix = 10,000 x 10,000
Memory needed   = 100,000,000 values

Memory grows QUADRATICALLY with sequence length!
This is why standard GPT2 maxes out at 1024 tokens.
FlashAttention solves this with tiling.
""")

print("=" * 50)
print("PART 2 - Memory Comparison")
print("=" * 50)

def standard_attention_memory(seq_len, d_model, dtype_bytes=4):
    attention_matrix = seq_len * seq_len
    qkv = 3 * seq_len * d_model
    output = seq_len * d_model
    total = (attention_matrix + qkv + output) * dtype_bytes
    return total / 1e6

def flash_attention_memory(seq_len, d_model, block_size=64, dtype_bytes=4):
    blocks = seq_len // block_size
    block_memory = block_size * block_size
    qkv = 3 * seq_len * d_model
    output = seq_len * d_model
    total = (block_memory + qkv + output) * dtype_bytes
    return total / 1e6

seq_lengths = [512, 1024, 2048, 4096, 8192]
d_model = 768

print(f"{'Seq Len':<10} {'Standard (MB)':<20} {'FlashAttn (MB)':<20} {'Savings'}")
print("-" * 65)
for seq_len in seq_lengths:
    std_mem = standard_attention_memory(seq_len, d_model)
    flash_mem = flash_attention_memory(seq_len, d_model)
    savings = std_mem / flash_mem
    print(f"{seq_len:<10} {std_mem:<20.1f} {flash_mem:<20.1f} {savings:.0f}x less")

print("\n" + "=" * 50)
print("PART 3 - Speed Comparison")
print("=" * 50)

def standard_attention(Q, K, V, scale):
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)

def tiled_attention(Q, K, V, scale, block_size=32):
    seq_len = Q.size(0)
    d_k = Q.size(1)
    output = torch.zeros_like(Q)

    for i in range(0, seq_len, block_size):
        Q_block = Q[i:i+block_size]
        scores_block = torch.zeros(Q_block.size(0), seq_len)

        for j in range(0, seq_len, block_size):
            K_block = K[j:j+block_size]
            scores_block[:, j:j+block_size] = torch.matmul(
                Q_block, K_block.transpose(-2, -1)
            ) * scale

        weights_block = torch.softmax(scores_block, dim=-1)

        for j in range(0, seq_len, block_size):
            V_block = V[j:j+block_size]
            output[i:i+block_size] += torch.matmul(
                weights_block[:, j:j+block_size], V_block
            )

    return output

seq_len = 256
d_k = 64
scale = 1.0 / math.sqrt(d_k)

Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)
V = torch.randn(seq_len, d_k)

runs = 5

start = time.time()
for _ in range(runs):
    std_out = standard_attention(Q, K, V, scale)
std_time = (time.time() - start) / runs

start = time.time()
for _ in range(runs):
    tiled_out = tiled_attention(Q, K, V, scale)
tiled_time = (time.time() - start) / runs

print(f"Standard attention time: {std_time*1000:.2f} ms")
print(f"Tiled attention time:    {tiled_time*1000:.2f} ms")
print(f"Outputs match: {torch.allclose(std_out, tiled_out, atol=1e-5)}")

print("\n" + "=" * 50)
print("PART 4 - Modern Efficient Models")
print("=" * 50)

efficient_models = [
    {
        "model": "Mistral 7B",
        "technique": "Sliding Window Attention",
        "context": "8K tokens",
        "benefit": "Linear memory instead of quadratic"
    },
    {
        "model": "Llama 3",
        "technique": "Grouped Query Attention",
        "context": "8K-128K tokens",
        "benefit": "Fewer KV heads = less memory"
    },
    {
        "model": "Phi-3 Mini",
        "technique": "FlashAttention 2",
        "context": "4K-128K tokens",
        "benefit": "2x faster than standard attention"
    },
    {
        "model": "Gemma 2",
        "technique": "Local + Global Attention",
        "context": "8K tokens",
        "benefit": "Alternates between local and global"
    },
]

print(f"{'Model':<15} {'Technique':<30} {'Context':<15} {'Benefit'}")
print("-" * 80)
for m in efficient_models:
    print(f"{m['model']:<15} {m['technique']:<30} {m['context']:<15} {m['benefit']}")

print("\n" + "=" * 50)
print("PART 5 - Why This Matters For Enterprise")
print("=" * 50)

print("""
Standard Attention limits:
  Max context = 1024-2048 tokens
  Long documents get truncated
  Client loses important information

FlashAttention enables:
  Max context = 8K-128K tokens
  Process entire legal documents
  Process entire medical records
  Process entire codebases

Enterprise impact:
  Law firm:    Process 50 page contracts fully
  Hospital:    Process complete patient history
  Finance:     Process full annual reports

This directly increases your product value.
Longer context = better answers = higher price.
""")

print("Day 31 Complete ✅")
print("Tomorrow: Mixture of Experts + Modern Architectures")