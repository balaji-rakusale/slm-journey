import torch
import json

print("=" * 50)
print("PART 1 - LoRA vs QLoRA")
print("=" * 50)

# LoRA  = freeze base model + train small matrices
# QLoRA = quantize base model to 4bit + freeze + train small matrices

# Why QLoRA is revolutionary:
# Llama 7B in float32 = 28 GB RAM  (impossible on laptop)
# Llama 7B in float16 = 14 GB RAM  (needs expensive GPU)
# Llama 7B in 4bit    =  3.5 GB RAM (runs on free Colab!)

memory_comparison = {
    'Llama 7B float32': 28,
    'Llama 7B float16': 14,
    'Llama 7B 8bit':     7,
    'Llama 7B 4bit':   3.5,
}

print("Memory requirements for Llama 7B:")
print()
for config, gb in memory_comparison.items():
    bar = "█" * int(gb * 2)
    feasible = "Feasible" if gb <= 8 else "Needs expensive GPU"
    print(f"  {config:25} {gb:5.1f} GB  {bar:30} {feasible}")

print("\n" + "=" * 50)
print("PART 2 - Quantization Explained")
print("=" * 50)

# Quantization = represent numbers with less precision
# float32 = 32 bits per number (high precision)
# float16 = 16 bits per number (medium precision)
# int8    =  8 bits per number (lower precision)
# int4    =  4 bits per number (lowest precision)

print("What quantization does to a weight:")
print()

weight = 3.14159265358979

print(f"Original float32: {weight}")
print(f"As float16:       {torch.tensor(weight, dtype=torch.float16).item():.5f}")
print(f"As int8 approx:   {round(weight * 40) / 40:.5f}")
print(f"As int4 approx:   {round(weight * 8) / 8:.5f}")

print()
print("Precision loss is tiny but memory savings are massive!")
print("4bit uses 8x less memory than float32")
print("Quality loss is less than 1% on most benchmarks")

print("\n" + "=" * 50)
print("PART 3 - QLoRA Configuration")
print("=" * 50)

# This is the exact config you will use on real models
qlora_config = {
    "model": "meta-llama/Llama-3.2-1B or mistralai/Mistral-7B",
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True
    },
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    },
    "training": {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 10,
        "max_steps": 100,
        "learning_rate": 2e-4,
        "fp16": True,
        "logging_steps": 10,
        "output_dir": "./qlora_output"
    }
}

print("QLoRA Configuration for real models:")
print()
for section, values in qlora_config.items():
    print(f"{section.upper()}:")
    if isinstance(values, dict):
        for k, v in values.items():
            print(f"  {k}: {v}")
    else:
        print(f"  {values}")
    print()

print("\n" + "=" * 50)
print("PART 4 - GPU Options For Real Training")
print("=" * 50)

gpu_options = [
    {
        "name": "Google Colab Free",
        "gpu": "T4 15GB",
        "cost": "Free",
        "max_model": "7B with QLoRA",
        "speed": "Slow"
    },
    {
        "name": "Google Colab Pro",
        "gpu": "A100 40GB",
        "cost": "$10/month",
        "max_model": "13B with QLoRA",
        "speed": "Fast"
    },
    {
        "name": "RunPod",
        "gpu": "A100 80GB",
        "cost": "$1.5/hr",
        "max_model": "70B with QLoRA",
        "speed": "Very Fast"
    },
    {
        "name": "Vast.ai",
        "gpu": "Various",
        "cost": "$0.2-0.8/hr",
        "max_model": "7B-13B",
        "speed": "Medium"
    },
    {
        "name": "Lambda Labs",
        "gpu": "A100 80GB",
        "cost": "$1.1/hr",
        "max_model": "70B with QLoRA",
        "speed": "Very Fast"
    },
]

print(f"{'Platform':<22} {'GPU':<15} {'Cost':<15} {'Max Model':<20} {'Speed'}")
print("-" * 85)
for opt in gpu_options:
    print(f"{opt['name']:<22} {opt['gpu']:<15} {opt['cost']:<15} {opt['max_model']:<20} {opt['speed']}")

print("\n" + "=" * 50)
print("PART 5 - Your Next Step Plan")
print("=" * 50)

print("""
Current state:
   Fine tuned GPT2 on CPU (small model, proof of concept)
   Understood LoRA and QLoRA theory
   Built complete data pipeline

Next steps to real enterprise model:

Step 1 - Get GPU access
  → Sign up for Google Colab (free)
  → Or rent RunPod at $0.5/hr

Step 2 - Use bigger model
  → Llama 3.2 1B (good starting point)
  → Mistral 7B   (best quality/size ratio)
  → Phi-3 Mini   (smallest, fastest)

Step 3 - Use Unsloth library
  → 2x faster than standard training
  → Less memory usage
  → Same results

Step 4 - Train on your domain data
  → Use fine_tune_ready.json we built
  → Add more domain specific samples
  → Run for 1-3 epochs

Step 5 - Evaluate and iterate
  → Use Day 17 evaluation pipeline
  → Identify weaknesses
  → Add targeted data
  → Retrain
""")

print("\n" + "=" * 50)
print("PART 6 - Prepare Colab Notebook")
print("=" * 50)

colab_code = '''
# QLoRA Fine Tuning on Google Colab
# Copy this to a new Colab notebook

# Step 1 - Install
!pip install unsloth transformers peft accelerate bitsandbytes trl datasets

# Step 2 - Load model with 4bit quantization
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-3-mini-4k-instruct",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Step 3 - Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = True,
)

print("Model ready for fine tuning!")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
'''

with open('colab_qlora_starter.py', 'w') as f:
    f.write(colab_code)

print("Saved colab_qlora_starter.py")
print("This is your starter code for real GPU training!")
print()
print("Tomorrow: Set up Colab and run your first real QLoRA fine tune!")