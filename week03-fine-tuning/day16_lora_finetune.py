import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

print("=" * 50)
print("PART 1 - Setup and Configuration")
print("=" * 50)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# We use GPT2 - small enough for CPU training
MODEL_NAME = "gpt2"
MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-4

print(f"Model:         {MODEL_NAME}")
print(f"Max length:    {MAX_LENGTH}")
print(f"Batch size:    {BATCH_SIZE}")
print(f"Epochs:        {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")

print("\n" + "=" * 50)
print("PART 2 - Load Model and Tokenizer")
print("=" * 50)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer loaded")
print(f"Vocab size: {tokenizer.vocab_size}")

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)
print(f"Base model loaded")

# Count base model parameters
base_params = sum(p.numel() for p in model.parameters())
print(f"Base model parameters: {base_params:,}")

print("\n" + "=" * 50)
print("PART 3 - Apply LoRA Configuration")
print("=" * 50)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # rank
    lora_alpha=16,          # scaling
    lora_dropout=0.05,      # dropout
    target_modules=["c_attn"],  # GPT2 attention
    bias="none"
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
print("LoRA applied to model")

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters:     {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Trainable %:          {trainable/total*100:.2f}%")
print(f"Parameter reduction:  {total/trainable:.0f}x")

print("\n" + "=" * 50)
print("PART 4 - Load and Prepare Dataset")
print("=" * 50)

# Load our fine tuning dataset
with open('fine_tune_ready.json', 'r') as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} samples")

# Tokenize dataset
def tokenize_function(examples):
    tokens = tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        return_tensors=None
    )
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

# Convert to HuggingFace dataset
texts = [item['text'] for item in raw_data]
hf_dataset = Dataset.from_dict({'text': texts})

# Split
split = hf_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split['train']
eval_dataset = split['test']

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples:  {len(eval_dataset)}")

# Tokenize
print("Tokenizing dataset...")
train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)
eval_tokenized = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)
print("Tokenization complete")

print("\n" + "=" * 50)
print("PART 5 - Training Configuration")
print("=" * 50)

training_args = TrainingArguments(
    output_dir='./lora_output',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=10,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    fp16=False,
    report_to="none"
)
# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    data_collator=data_collator
)

print("Trainer configured")

print("\n" + "=" * 50)
print("PART 6 - Train!")
print("=" * 50)

print("Starting LoRA fine tuning...")
print("This will take 5-10 minutes on CPU...")
print()

trainer.train()

print("\nTraining complete!")

print("\n" + "=" * 50)
print("PART 7 - Save and Test Model")
print("=" * 50)

# Save the LoRA weights
model.save_pretrained('./lora_weights')
tokenizer.save_pretrained('./lora_weights')
print("LoRA weights saved to ./lora_weights")

# Test the fine tuned model
print("\nTesting fine tuned model...")

model.eval()
test_prompt = """### Instruction:
Classify this news article into a category.

### Input:
Apple announced new iPhone with advanced AI features today.

### Response:"""

inputs = tokenizer(
    test_prompt,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nTest prompt: Apple announced new iPhone with AI features")
print(f"Model response: {response[len(test_prompt):]}")