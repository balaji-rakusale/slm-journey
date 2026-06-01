import json
from datasets import load_from_disk
import pandas as pd
from collections import Counter

print("=" * 50)
print("WEEK 2 COMPLETE - FINAL DATA AUDIT")
print("=" * 50)

print("\n" + "=" * 50)
print("PART 1 - Load Final Pipeline Dataset")
print("=" * 50)

# Load everything we built this week
dataset = load_from_disk('final_pipeline_dataset')
train = dataset['train']
test = dataset['test']

print(f"Train samples: {len(train)}")
print(f"Test samples:  {len(test)}")
print(f"Total samples: {len(train) + len(test)}")
print(f"Features:      {train.features}")

print("\n" + "=" * 50)
print("PART 2 - Dataset Health Check")
print("=" * 50)

df = train.to_pandas()

# Check for empty fields
print("Empty field check:")
print(f"  Empty instructions: {df['instruction'].isna().sum()}")
print(f"  Empty inputs:       {df['input'].isna().sum()}")
print(f"  Empty outputs:      {df['output'].isna().sum()}")

# Length statistics
print(f"\nLength statistics:")
print(f"  Instruction - min: {df['instruction'].str.len().min()} "
      f"max: {df['instruction'].str.len().max()} "
      f"avg: {df['instruction'].str.len().mean():.0f}")
print(f"  Input       - min: {df['input'].str.len().min()} "
      f"max: {df['input'].str.len().max()} "
      f"avg: {df['input'].str.len().mean():.0f}")
print(f"  Output      - min: {df['output'].str.len().min()} "
      f"max: {df['output'].str.len().max()} "
      f"avg: {df['output'].str.len().mean():.0f}")

print("\n" + "=" * 50)
print("PART 3 - Source Distribution")
print("=" * 50)

sources = Counter(train['source'])
total = len(train)
print("Source breakdown:")
for source, count in sources.items():
    pct = count / total * 100
    bar = "█" * int(pct / 5)
    print(f"  {source:12} {bar:20} {count:3} samples ({pct:.1f}%)")

print("\n" + "=" * 50)
print("PART 4 - Sample From Each Source")
print("=" * 50)

# Show one sample from each source
shown_sources = set()
for i in range(len(train)):
    sample = train[i]
    if sample['source'] not in shown_sources:
        shown_sources.add(sample['source'])
        print(f"\nSource: {sample['source']}")
        print(f"  Instruction: {sample['instruction'][:80]}")
        print(f"  Input:       {sample['input'][:80]}")
        print(f"  Output:      {sample['output'][:80]}")
    if len(shown_sources) == 3:
        break

print("\n" + "=" * 50)
print("PART 5 - Prepare For Fine Tuning Format")
print("=" * 50)

# Convert to chat format - what fine tuning libraries expect
def to_chat_format(example):
    if example['input']:
        prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        prompt = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    return {'text': prompt, 'source': example['source']}

chat_dataset = train.map(to_chat_format)
print(f"Chat format samples: {len(chat_dataset)}")
print(f"\nSample chat format:")
print(chat_dataset[0]['text'])

print("\n" + "=" * 50)
print("PART 6 - Save Fine Tuning Ready Dataset")
print("=" * 50)

# Save as JSON
fine_tune_data = [chat_dataset[i] for i in range(len(chat_dataset))]
with open('fine_tune_ready.json', 'w', encoding='utf-8') as f:
    json.dump(fine_tune_data, f, indent=2, ensure_ascii=False)

print(f"Saved fine_tune_ready.json")
print(f"Total samples ready: {len(fine_tune_data)}")
print(f"\nThis dataset is now ready to feed into LoRA fine tuning next week!")

print("\n" + "=" * 50)
print("WEEK 2 SUMMARY")
print("=" * 50)
print("""
Day 08 → Web scraping + HuggingFace datasets
Day 09 → Data cleaning pipeline
Day 10 → Instruction dataset building
Day 11 → PII removal + quality scoring
Day 12 → HuggingFace operations mastery
Day 13 → End to end pipeline from 3 sources
Day 14 → Data audit + fine tuning format

Dataset Journey:
  Raw scraped:     188 paragraphs
  After cleaning:  175 paragraphs
  After formatting: 55 samples
  After PII:        42 samples
  After combining: 357 samples
  Fine tune ready: 321 samples

WEEK 3 PREVIEW → Fine Tuning with LoRA starts tomorrow!
""")