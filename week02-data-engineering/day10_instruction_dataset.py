import json
import random

print("=" * 50)
print("PART 1 - Understanding Instruction Format")
print("=" * 50)

# This is the standard format used to fine tune every LLM
# Alpaca format - used by Stanford, Llama, Mistral fine tunes

sample_instruction = {
    "instruction": "Explain what artificial intelligence is",
    "input": "",
    "output": "Artificial intelligence is the capability of computational systems to perform tasks typically associated with human intelligence such as learning reasoning and problem solving."
}

print("Standard Alpaca Format:")
print(json.dumps(sample_instruction, indent=2))

print("\n" + "=" * 50)
print("PART 2 - Load Our Clean Dataset")
print("=" * 50)

# Load the clean dataset from Day 9
with open('clean_dataset.json', 'r', encoding='utf-8') as f:
    clean_data = json.load(f)

paragraphs = clean_data['data']
print(f"Loaded {len(paragraphs)} clean paragraphs")
print(f"Sample paragraph: {paragraphs[0][:150]}")

print("\n" + "=" * 50)
print("PART 3 - Auto Generate Instructions")
print("=" * 50)

# Templates to auto generate instructions from paragraphs
instruction_templates = [
    "Explain the following concept in simple terms:",
    "Summarize the following text:",
    "What is the main idea of this paragraph?",
    "Describe what this text is talking about:",
    "Explain this to a beginner:"
]

# Generate instruction dataset from our clean paragraphs
instruction_dataset = []

for i, paragraph in enumerate(paragraphs[:50]):  # use first 50
    template = random.choice(instruction_templates)

    sample = {
        "instruction": template,
        "input": paragraph[:300],  # use first 300 chars as input
        "output": paragraph[300:600] if len(paragraph) > 300 else paragraph
    }
    instruction_dataset.append(sample)

print(f"Generated {len(instruction_dataset)} instruction samples")
print(f"\nSample instruction:")
print(json.dumps(instruction_dataset[0], indent=2))

print("\n" + "=" * 50)
print("PART 4 - Build QA Dataset")
print("=" * 50)

# Question Answer pairs - better format for fine tuning
qa_templates = [
    ("What is artificial intelligence?", 0),
    ("How does machine learning work?", 5),
    ("What is deep learning?", 10),
    ("How do neural networks learn?", 15),
    ("What is natural language processing?", 20),
]

qa_dataset = []
for question, para_idx in qa_templates:
    if para_idx < len(paragraphs):
        qa_dataset.append({
            "instruction": question,
            "input": "",
            "output": paragraphs[para_idx][:400]
        })

print(f"Generated {len(qa_dataset)} QA pairs")
print(f"\nSample QA pair:")
print(json.dumps(qa_dataset[0], indent=2))

print("\n" + "=" * 50)
print("PART 5 - Combine and Save")
print("=" * 50)

# Combine both datasets
full_dataset = instruction_dataset + qa_dataset
random.shuffle(full_dataset)

print(f"Total samples: {len(full_dataset)}")

# Split into train and validation
split = int(0.9 * len(full_dataset))
train_set = full_dataset[:split]
val_set = full_dataset[split:]

print(f"Train samples: {len(train_set)}")
print(f"Val samples:   {len(val_set)}")

# Save
with open('train_instructions.json', 'w', encoding='utf-8') as f:
    json.dump(train_set, f, indent=2, ensure_ascii=False)

with open('val_instructions.json', 'w', encoding='utf-8') as f:
    json.dump(val_set, f, indent=2, ensure_ascii=False)

print("\nSaved train_instructions.json")
print("Saved val_instructions.json")

print("\n" + "=" * 50)
print("PART 6 - Dataset Quality Check")
print("=" * 50)

# Check quality of our dataset
avg_instruction_len = sum(len(s['instruction']) for s in full_dataset) / len(full_dataset)
avg_input_len = sum(len(s['input']) for s in full_dataset) / len(full_dataset)
avg_output_len = sum(len(s['output']) for s in full_dataset) / len(full_dataset)

print(f"Average instruction length: {avg_instruction_len:.0f} chars")
print(f"Average input length:       {avg_input_len:.0f} chars")
print(f"Average output length:      {avg_output_len:.0f} chars")
print(f"\nDataset ready for fine tuning!")