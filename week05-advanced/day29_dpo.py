import torch
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 50)
print("PART 1 - What is DPO")
print("=" * 50)

print("""
The Problem With Fine Tuning Alone:
  Fine tuning teaches model WHAT to say
  But not HOW to say it well
  Model might be accurate but unhelpful
  Or technically correct but rude

RLHF Solution (Old Way):
  Step 1 → Fine tune model
  Step 2 → Train reward model
  Step 3 → Use PPO to optimize
  Problem → Complex, unstable, expensive

DPO Solution (New Way):
  Step 1 → Fine tune model
  Step 2 → Show pairs of responses
           Good response vs Bad response
  Step 3 → Model learns to prefer good
  Benefit → Simple, stable, cheap

DPO Formula:
  Given same prompt:
  Chosen response   → what human prefers
  Rejected response → what human dislikes
  Model learns:     → always generate chosen style
""")

print("=" * 50)
print("PART 2 - Build DPO Dataset")
print("=" * 50)

# DPO dataset format
# Each sample has: prompt, chosen, rejected
dpo_data = [
    {
        "prompt": "### Instruction:\nClassify this news article.\n\n### Input:\nApple launches new iPhone today.\n\n### Response:",
        "chosen": "Category: Technology\n\nThis article is about Apple's new iPhone product launch.",
        "rejected": "I don't know what category this is. It could be anything."
    },
    {
        "prompt": "### Instruction:\nWhat is machine learning?\n\n### Response:",
        "chosen": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
        "rejected": "Machine learning is something computers do. It's complicated and hard to explain."
    },
    {
        "prompt": "### Instruction:\nSummarize this text.\n\n### Input:\nNeural networks learn patterns from large datasets.\n\n### Response:",
        "chosen": "Neural networks automatically learn patterns and representations from large amounts of training data.",
        "rejected": "The text talks about neural networks and datasets and learning stuff."
    },
    {
        "prompt": "### Instruction:\nClassify this news article.\n\n### Input:\nFederal Reserve raises interest rates.\n\n### Response:",
        "chosen": "Category: Business/Finance\n\nThis article covers monetary policy and economic news.",
        "rejected": "This is a news article about something financial I think."
    },
    {
        "prompt": "### Instruction:\nWhat is deep learning?\n\n### Response:",
        "chosen": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn complex patterns from large amounts of data.",
        "rejected": "Deep learning is like really deep machine learning with lots of layers."
    },
    {
        "prompt": "### Instruction:\nClassify this news article.\n\n### Input:\nLionel Messi wins Ballon d'Or award.\n\n### Response:",
        "chosen": "Category: Sports\n\nThis article is about football/soccer and player recognition.",
        "rejected": "Sports maybe? I'm not sure about this one."
    },
    {
        "prompt": "### Instruction:\nWhat is a transformer model?\n\n### Response:",
        "chosen": "A transformer is a neural network architecture that uses attention mechanisms to process sequential data, enabling models to understand relationships between all words simultaneously.",
        "rejected": "Transformers are models that transform things. They use attention which helps them understand text."
    },
    {
        "prompt": "### Instruction:\nExplain fine tuning.\n\n### Response:",
        "chosen": "Fine tuning is the process of taking a pretrained language model and training it further on a specific dataset to adapt it for a particular task or domain.",
        "rejected": "Fine tuning means making a model better by training it more on some data."
    },
]

print(f"DPO dataset created: {len(dpo_data)} preference pairs")
print(f"\nSample preference pair:")
print(f"Prompt:   {dpo_data[0]['prompt'][-50:]}")
print(f"Chosen:   {dpo_data[0]['chosen'][:80]}")
print(f"Rejected: {dpo_data[0]['rejected'][:80]}")

print("\n" + "=" * 50)
print("PART 3 - Analyze Dataset Quality")
print("=" * 50)

# Analyze chosen vs rejected quality
chosen_lengths = [len(d['chosen'].split()) for d in dpo_data]
rejected_lengths = [len(d['rejected'].split()) for d in dpo_data]

avg_chosen = sum(chosen_lengths) / len(chosen_lengths)
avg_rejected = sum(rejected_lengths) / len(rejected_lengths)

print(f"Average chosen length:   {avg_chosen:.1f} words")
print(f"Average rejected length: {avg_rejected:.1f} words")
print(f"\nChosen responses are {avg_chosen/avg_rejected:.1f}x longer on average")
print("Longer = more detailed = higher quality")

print("\nQuality patterns in chosen responses:")
quality_patterns = [
    "Specific category labels",
    "Clear definitions",
    "Complete sentences",
    "Domain specific terminology",
    "Structured format"
]
for pattern in quality_patterns:
    print(f"  ✅ {pattern}")

print("\nQuality issues in rejected responses:")
bad_patterns = [
    "Vague answers",
    "Uncertainty expressed",
    "Incomplete information",
    "Informal language",
    "No structure"
]
for pattern in bad_patterns:
    print(f"  ❌ {pattern}")

print("\n" + "=" * 50)
print("PART 4 - DPO Training Setup")
print("=" * 50)

# Save DPO dataset
with open('dpo_dataset.json', 'w') as f:
    json.dump(dpo_data, f, indent=2)
print("DPO dataset saved ✅")

# Show DPO training config
dpo_config = {
    "model": "gpt2 or phi-2",
    "beta": 0.1,
    "learning_rate": 1e-5,
    "batch_size": 2,
    "max_length": 256,
    "epochs": 3,
    "optimizer": "adamw"
}

print("\nDPO Training Configuration:")
for key, value in dpo_config.items():
    print(f"  {key}: {value}")

print("""
Key DPO Parameter — Beta:
  Beta = 0.1  → gentle alignment (recommended)
  Beta = 0.5  → moderate alignment
  Beta = 1.0  → strong alignment (may hurt quality)
  Start with 0.1 always
""")

print("\n" + "=" * 50)
print("PART 5 - DPO vs SFT Comparison")
print("=" * 50)

comparison = {
    "Supervised Fine Tuning (SFT)": {
        "data_needed": "instruction + output pairs",
        "teaches": "what to say",
        "result": "capable but may be unhelpful",
        "cost": "low",
        "complexity": "low"
    },
    "DPO": {
        "data_needed": "prompt + chosen + rejected triplets",
        "teaches": "how to say it well",
        "result": "helpful, harmless, honest",
        "cost": "medium",
        "complexity": "medium"
    },
    "RLHF": {
        "data_needed": "human rankings + reward model",
        "teaches": "what humans prefer",
        "result": "best alignment",
        "cost": "very high",
        "complexity": "very high"
    }
}

for method, details in comparison.items():
    print(f"\n{method}:")
    for key, value in details.items():
        print(f"  {key}: {value}")

print("\n" + "=" * 50)
print("PART 6 - Enterprise Value of DPO")
print("=" * 50)

print("""
Why Enterprise Clients Pay Extra for DPO:

Without DPO:
  Client: "What is our return policy?"
  Model:  "I think it might be 30 days or something
           but I'm not totally sure about this"

With DPO:
  Client: "What is our return policy?"
  Model:  "Our return policy allows returns within
           30 days of purchase with original receipt.
           Please contact support@company.com"

The difference:
  ❌ Uncertain, vague, unprofessional
  ✅ Confident, specific, professional

Enterprise clients notice this immediately.
DPO is what separates a demo from a product.

Add $5,000-$15,000 to your project price
when you include DPO alignment.
""")

print("\n" + "=" * 50)
print("PART 7 - Your DPO Action Plan")
print("=" * 50)

print("""
How to implement DPO for a client project:

Step 1 - Collect preference data
  → Show client 10 response pairs
  → They mark which is better
  → 50-100 pairs is enough to start

Step 2 - Run SFT first
  → Fine tune on instruction data (what we did)
  → Get a capable base model

Step 3 - Run DPO
  → Use TRL DPOTrainer
  → Train for 1-3 epochs
  → Beta = 0.1

Step 4 - Evaluate
  → Compare SFT vs DPO responses
  → Show client the difference
  → Charge premium for quality

Code for Colab (run tomorrow):
  from trl import DPOTrainer, DPOConfig
  trainer = DPOTrainer(
      model=model,
      args=DPOConfig(beta=0.1),
      train_dataset=dpo_dataset,
  )
  trainer.train()
""")

print("Day 29 Complete ✅")
print("Tomorrow: Run actual DPO training on Colab!")