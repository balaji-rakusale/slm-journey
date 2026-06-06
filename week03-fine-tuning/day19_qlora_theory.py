print("Day 19 - QLoRA on GPU")
print()
print("Today's plan:")
print("  → Set up Kaggle with free GPU")
print("  → Run QLoRA fine tune on Phi-3 Mini")
print("  → Compare results with Day 16 GPT2")
print()

# Document the QLoRA config we will use
config = {
    "model": "Phi-3-mini-4k-instruct",
    "parameters": "3.8 Billion",
    "quantization": "4bit",
    "memory_needed": "~4 GB",
    "lora_rank": 16,
    "lora_alpha": 32,
    "training_steps": 60,
    "expected_time": "5-10 minutes on T4"
}

print("QLoRA Config:")
for k, v in config.items():
    print(f"  {k}: {v}")

print()
print("GPT2 vs Phi-3 comparison:")
print(f"  {'Model':<15} {'Params':<15} {'Quality'}")
print("-" * 45)
print(f"  {'GPT2':<15} {'124M':<15} {'Basic'}")
print(f"  {'Phi-3 Mini':<15} {'3.8B':<15} {'Excellent'}")
print(f"  {'Mistral 7B':<15} {'7B':<15} {'Production grade'}")