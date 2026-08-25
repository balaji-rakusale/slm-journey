import torch
import json
import random
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 50)
print("PART 1 - Advanced Fine Tuning Strategies")
print("=" * 50)

print("""
Basic Fine Tuning (What we did):
  → Single dataset
  → Fixed learning rate
  → Train until loss converges
  → Hope it works!

Advanced Strategies:

1. Curriculum Learning
   → Start with easy examples
   → Gradually add harder ones
   → Model learns foundation first
   → Better final performance

2. Multi Task Fine Tuning
   → Train on multiple tasks simultaneously
   → Classification + QA + Summarization
   → More generalizable model
   → Less catastrophic forgetting

3. Continual Learning
   → Add new tasks without forgetting old ones
   → Use elastic weight consolidation
   → Critical for production models

4. Few Shot Fine Tuning
   → Fine tune on just 10-50 examples
   → Use very low learning rate
   → Works surprisingly well
   → Perfect for quick domain adaptation
""")

print("=" * 50)
print("PART 2 - Curriculum Learning")
print("=" * 50)

# Create datasets with different difficulty levels
easy_examples = [
    {"text": "### Instruction:\nWhat is AI?\n\n### Response:\nAI is artificial intelligence.", "difficulty": 1},
    {"text": "### Instruction:\nWhat is ML?\n\n### Response:\nML is machine learning.", "difficulty": 1},
    {"text": "### Instruction:\nWhat is NLP?\n\n### Response:\nNLP is natural language processing.", "difficulty": 1},
]

medium_examples = [
    {"text": "### Instruction:\nExplain machine learning.\n\n### Response:\nMachine learning enables computers to learn from data without explicit programming.", "difficulty": 2},
    {"text": "### Instruction:\nWhat is a neural network?\n\n### Response:\nA neural network is a system inspired by the human brain that learns patterns from data.", "difficulty": 2},
    {"text": "### Instruction:\nExplain fine tuning.\n\n### Response:\nFine tuning adapts a pretrained model to specific tasks using domain specific data.", "difficulty": 2},
]

hard_examples = [
    {"text": "### Instruction:\nExplain the mathematical intuition behind attention mechanisms in transformers.\n\n### Response:\nAttention computes weighted sums of values where weights are determined by query key compatibility using scaled dot product similarity.", "difficulty": 3},
    {"text": "### Instruction:\nCompare LoRA and full fine tuning in terms of parameter efficiency and performance tradeoffs.\n\n### Response:\nLoRA trains small rank decomposition matrices achieving 0.1-1% parameter updates while maintaining 95-99% of full fine tuning quality.", "difficulty": 3},
]

all_examples = easy_examples + medium_examples + hard_examples

print(f"Easy examples:   {len(easy_examples)}")
print(f"Medium examples: {len(medium_examples)}")
print(f"Hard examples:   {len(hard_examples)}")
print(f"Total examples:  {len(all_examples)}")

# Curriculum learning order
def get_curriculum_order(examples):
    return sorted(examples, key=lambda x: x['difficulty'])

curriculum_order = get_curriculum_order(all_examples)
random_order = all_examples.copy()
random.shuffle(random_order)

print("\nCurriculum order (easy first):")
for ex in curriculum_order:
    diff_label = "Easy" if ex['difficulty'] == 1 else "Medium" if ex['difficulty'] == 2 else "Hard"
    print(f"  [{diff_label}] {ex['text'][20:60]}...")

print("\n" + "=" * 50)
print("PART 3 - Multi Task Fine Tuning")
print("=" * 50)

# Different task types
multi_task_dataset = [
    # Classification task
    {
        "task": "classification",
        "text": "### Instruction:\nClassify this text: Apple launches new iPhone.\n\n### Response:\nCategory: Technology"
    },
    # QA task
    {
        "task": "qa",
        "text": "### Instruction:\nAnswer this question: What is the capital of France?\n\n### Response:\nThe capital of France is Paris."
    },
    # Summarization task
    {
        "task": "summarization",
        "text": "### Instruction:\nSummarize: Neural networks learn from data through backpropagation adjusting weights to minimize loss.\n\n### Response:\nNeural networks learn by adjusting weights through backpropagation."
    },
    # Generation task
    {
        "task": "generation",
        "text": "### Instruction:\nComplete this sentence: Machine learning is important because\n\n### Response:\nMachine learning is important because it enables computers to solve complex problems automatically."
    },
]

from collections import Counter
task_dist = Counter(ex['task'] for ex in multi_task_dataset)
print("Multi task distribution:")
for task, count in task_dist.items():
    print(f"  {task}: {count} examples")

print("\nBenefit of multi task training:")
print("  → Model learns to follow different instruction types")
print("  → More robust to varied inputs")
print("  → Better generalization")

print("\n" + "=" * 50)
print("PART 4 - Learning Rate Strategies")
print("=" * 50)

print("""
Learning Rate Schedules:

1. Constant LR (basic):
   LR = 2e-4 throughout
   Problem: too high at end, overshoots

2. Linear Decay:
   LR starts at 2e-4
   Linearly decreases to 0
   Better convergence

3. Cosine Annealing (best):
   LR follows cosine curve
   Smooth reduction
   Can restart for exploration
   Used by most SOTA models

4. Warmup + Cosine (standard):
   Steps 1-100:   LR increases linearly (warmup)
   Steps 100+:    LR decreases with cosine
   Why warmup?    Model unstable at start
                  Low LR prevents large updates
""")

# Simulate learning rate schedules
import math

def get_lr_schedule(step, total_steps, warmup_steps, max_lr, schedule_type):
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)

    if schedule_type == "linear":
        return max_lr * (1 - progress)
    elif schedule_type == "cosine":
        return max_lr * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        return max_lr

total_steps = 100
warmup_steps = 10
max_lr = 2e-4

print("\nLR at different steps (cosine schedule):")
for step in [0, 10, 25, 50, 75, 100]:
    lr = get_lr_schedule(step, total_steps, warmup_steps, max_lr, "cosine")
    bar = "█" * int(lr / max_lr * 20)
    print(f"  Step {step:3d}: {lr:.2e} {bar}")

print("\n" + "=" * 50)
print("PART 5 - Catastrophic Forgetting")
print("=" * 50)

print("""
Catastrophic Forgetting Problem:

Fine tune on legal documents →
  Model becomes great at legal tasks
  Model forgets general knowledge!

Example:
  Before fine tuning: "What is 2+2?" → "4"
  After fine tuning:  "What is 2+2?" → "The contract stipulates..."

Solutions:

1. Mix in general data (simplest):
   80% domain data + 20% general data
   Model keeps general knowledge

2. Low learning rate:
   1e-5 instead of 2e-4
   Small updates preserve old knowledge

3. Elastic Weight Consolidation (EWC):
   Identify important weights
   Penalize changing them
   Complex but effective

4. LoRA (best solution!):
   Base model weights FROZEN
   Only adapters change
   Base knowledge perfectly preserved
   This is why LoRA is so popular!
""")

print("=" * 50)
print("PART 6 - Few Shot Fine Tuning")
print("=" * 50)

# Demonstrate few shot fine tuning setup
few_shot_examples = [
    {"text": "### Instruction:\nClassify sentiment: I love this product!\n\n### Response:\nPositive"},
    {"text": "### Instruction:\nClassify sentiment: This is terrible.\n\n### Response:\nNegative"},
    {"text": "### Instruction:\nClassify sentiment: It works okay.\n\n### Response:\nNeutral"},
    {"text": "### Instruction:\nClassify sentiment: Amazing quality!\n\n### Response:\nPositive"},
    {"text": "### Instruction:\nClassify sentiment: Worst purchase ever.\n\n### Response:\nNegative"},
]

few_shot_config = {
    "num_examples": len(few_shot_examples),
    "learning_rate": 1e-5,
    "epochs": 10,
    "batch_size": 2,
    "max_steps": 25,
    "why_works": "Base model already understands sentiment — just needs format"
}

print(f"Few shot fine tuning config:")
for key, value in few_shot_config.items():
    print(f"  {key}: {value}")

print(f"\nWith only {len(few_shot_examples)} examples you can:")
print("  → Teach model a new output format")
print("  → Adapt to domain specific terminology")
print("  → Change response style completely")
print("  → Add new classification categories")

print("\n" + "=" * 50)
print("PART 7 - Your Advanced Fine Tuning Checklist")
print("=" * 50)

print("""
Before Fine Tuning:
  ✅ Analyze dataset difficulty distribution
  ✅ Plan curriculum order if needed
  ✅ Mix multiple task types
  ✅ Include 10-20% general data
  ✅ Choose appropriate learning rate

During Fine Tuning:
  ✅ Monitor train AND validation loss
  ✅ Use cosine LR schedule with warmup
  ✅ Save checkpoints every epoch
  ✅ Stop if validation loss increases (overfitting)

After Fine Tuning:
  ✅ Evaluate on held out test set
  ✅ Test catastrophic forgetting
  ✅ Compare to base model
  ✅ Run DPO alignment if needed
  ✅ Quantize for deployment
""")

# Save advanced config
advanced_config = {
    "curriculum_learning": True,
    "multi_task": True,
    "lr_schedule": "cosine_with_warmup",
    "warmup_steps": 10,
    "max_lr": 2e-4,
    "min_lr": 1e-6,
    "general_data_mix": 0.2,
    "use_lora": True,
    "lora_rank": 16,
    "dpo_after_sft": True,
    "quantize_after_dpo": True
}

with open('advanced_finetuning_config.json', 'w') as f:
    json.dump(advanced_config, f, indent=2)

print("\nAdvanced config saved ✅")
print("Day 40 Complete ✅")
print("Tomorrow: Agents and tool use!")