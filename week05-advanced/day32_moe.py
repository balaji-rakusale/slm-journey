import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 50)
print("PART 1 - What is Mixture of Experts")
print("=" * 50)

print("""
Standard Dense Model:
  Every token → goes through ALL layers
  7B model → uses all 7B parameters every time
  Slow and memory hungry for large models

Mixture of Experts (MoE):
  Every token → goes through SELECTED experts only
  46B model   → uses only 12B parameters per token
  Faster inference despite bigger total size

Real Example — Mixtral 8x7B:
  Total parameters:  46B
  Active parameters: 12B per token
  8 expert FFN layers per block
  Each token uses only 2 experts
  Result: 7B quality at 2x speed!
""")

print("=" * 50)
print("PART 2 - Build Simple MoE Layer")
print("=" * 50)

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)

class Router(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        scores = self.gate(x)
        probs = F.softmax(scores, dim=-1)
        return probs

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim)
            for _ in range(num_experts)
        ])
        self.router = Router(input_dim, num_experts)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        # Router decides which experts to use
        router_probs = self.router(x_flat)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )

        # Normalize selected probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Combine expert outputs
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]
            expert_probs = top_k_probs[:, k:k+1]

            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_probs[mask] * expert_output

        return output.view(batch_size, seq_len, dim)

# Test MoE
batch_size = 2
seq_len = 4
input_dim = 64
hidden_dim = 128
num_experts = 8
top_k = 2

moe = MixtureOfExperts(input_dim, hidden_dim, num_experts, top_k)
x = torch.randn(batch_size, seq_len, input_dim)
output = moe(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")

total_params = sum(p.numel() for p in moe.parameters())
expert_params = sum(p.numel() for p in moe.experts[0].parameters())
router_params = sum(p.numel() for p in moe.router.parameters())
active_params = router_params + (top_k * expert_params)

print(f"\nTotal parameters:  {total_params:,}")
print(f"Active per token:  {active_params:,}")
print(f"Sparsity:          {(1 - active_params/total_params)*100:.1f}%")
print(f"Experts used:      {top_k}/{num_experts} per token")

print("\n" + "=" * 50)
print("PART 3 - Router Analysis")
print("=" * 50)

# Analyze which experts get selected
router_probs = moe.router(x.view(-1, input_dim))
top_k_probs, top_k_indices = torch.topk(router_probs, top_k, dim=-1)

expert_counts = torch.zeros(num_experts)
for idx in top_k_indices.view(-1):
    expert_counts[idx] += 1

print("Expert selection distribution:")
for i, count in enumerate(expert_counts):
    bar = "█" * int(count.item())
    print(f"  Expert {i}: {bar} ({count.item():.0f} times)")

print("\nIdeal distribution = equal load across all experts")
print("Unequal = expert collapse (common training problem)")

print("\n" + "=" * 50)
print("PART 4 - MoE vs Dense Comparison")
print("=" * 50)

models = [
    {
        "name": "Llama 3 8B (Dense)",
        "total_params": "8B",
        "active_params": "8B",
        "speed": "1x baseline",
        "quality": "Good"
    },
    {
        "name": "Mixtral 8x7B (MoE)",
        "total_params": "46B",
        "active_params": "12B",
        "speed": "2x faster than 46B dense",
        "quality": "Excellent"
    },
    {
        "name": "Grok 1 (MoE)",
        "total_params": "314B",
        "active_params": "86B",
        "speed": "Fast despite huge size",
        "quality": "State of art"
    },
    {
        "name": "GPT4 (rumored MoE)",
        "total_params": "1.8T",
        "active_params": "~280B",
        "speed": "Fast API responses",
        "quality": "Best available"
    },
]

print(f"{'Model':<25} {'Total':<10} {'Active':<10} {'Speed':<25} {'Quality'}")
print("-" * 85)
for m in models:
    print(f"{m['name']:<25} {m['total_params']:<10} {m['active_params']:<10} {m['speed']:<25} {m['quality']}")

print("\n" + "=" * 50)
print("PART 5 - Enterprise Relevance")
print("=" * 50)

print("""
Why MoE matters for your enterprise clients:

Cost comparison for inference:
  Dense 7B model:   $0.50 per 1M tokens
  MoE 46B model:    $0.60 per 1M tokens
  Quality diff:     MoE much better

For enterprise:
  Same cost → dramatically better quality
  Legal document analysis improves significantly
  Medical note summarization more accurate
  Financial report analysis more nuanced

When pitching to clients:
  "I use Mixtral — same cost as smaller models
   but quality of a 46B parameter model"
  This justifies premium pricing.
""")

print("Day 32 Complete ✅")
print("Tomorrow: Model quantization deep dive")