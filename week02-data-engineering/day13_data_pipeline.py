import json
import re
import random
from datasets import load_dataset, Dataset, concatenate_datasets

print("=" * 50)
print("PART 1 - Load Multiple Sources")
print("=" * 50)

# Source 1 - News data
news = load_dataset("ag_news", split="train[:200]")
print(f"Source 1 - News articles: {len(news)}")

# Source 2 - Q&A data
qa = load_dataset("squad", split="train[:200]")
print(f"Source 2 - QA pairs: {len(qa)}")

# Source 3 - Our production dataset
with open('production_dataset.json', 'r') as f:
    our_data = json.load(f)
print(f"Source 3 - Our dataset: {len(our_data)}")

print("\n" + "=" * 50)
print("PART 2 - Standardize All Sources")
print("=" * 50)

# Every source must become same format
# instruction, input, output

# Convert news to standard format
def news_to_standard(example):
    label_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Technology'}
    return {
        'instruction': 'Classify this news article into a category.',
        'input': example['text'][:300],
        'output': f"Category: {label_names[example['label']]}",
        'source': 'ag_news',
        'quality_score': 100
    }

# Convert QA to standard format
def qa_to_standard(example):
    return {
        'instruction': example['question'],
        'input': example['context'][:300],
        'output': example['answers']['text'][0] if example['answers']['text'] else 'No answer found',
        'source': 'squad',
        'quality_score': 100
    }

news_standard = news.map(news_to_standard)
qa_standard = qa.map(qa_to_standard)

print(f"News standardized: {len(news_standard)}")
print(f"QA standardized:   {len(qa_standard)}")

print(f"\nNews sample:")
print(f"  Instruction: {news_standard[0]['instruction']}")
print(f"  Input:       {news_standard[0]['input'][:100]}")
print(f"  Output:      {news_standard[0]['output']}")

print(f"\nQA sample:")
print(f"  Instruction: {qa_standard[0]['instruction']}")
print(f"  Input:       {qa_standard[0]['input'][:100]}")
print(f"  Output:      {qa_standard[0]['output']}")

print("\n" + "=" * 50)
print("PART 3 - Clean All Sources")
print("=" * 50)

def clean_text(text):
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def clean_sample(example):
    return {
        'instruction': clean_text(example['instruction']),
        'input': clean_text(example['input']),
        'output': clean_text(example['output']),
        'source': example['source'],
        'quality_score': example['quality_score']
    }

news_clean = news_standard.map(clean_sample)
qa_clean = qa_standard.map(clean_sample)

print(f"News cleaned: {len(news_clean)}")
print(f"QA cleaned:   {len(qa_clean)}")

print("\n" + "=" * 50)
print("PART 4 - Combine All Sources")
print("=" * 50)

# Convert our data to HuggingFace dataset
for item in our_data:
    item['source'] = 'wikipedia'

our_dataset = Dataset.from_list(our_data)

# Select only matching columns
columns = ['instruction', 'input', 'output', 'source', 'quality_score']
news_final = news_clean.select_columns(columns)
qa_final = qa_clean.select_columns(columns)
our_final = our_dataset.select_columns(columns)

# Combine all three sources
combined = concatenate_datasets([news_final, qa_final, our_final])
print(f"News:       {len(news_final)} samples")
print(f"QA:         {len(qa_final)} samples")
print(f"Wikipedia:  {len(our_final)} samples")
print(f"Combined:   {len(combined)} samples")

print("\n" + "=" * 50)
print("PART 5 - Final Quality Filter")
print("=" * 50)

# Filter out empty outputs
def has_valid_output(example):
    return (len(example['output']) > 10 and
            len(example['instruction']) > 5 and
            example['output'] != 'No answer found')

filtered = combined.filter(has_valid_output)
print(f"Before filter: {len(combined)}")
print(f"After filter:  {len(filtered)}")
print(f"Removed:       {len(combined) - len(filtered)}")

print("\n" + "=" * 50)
print("PART 6 - Final Split and Save")
print("=" * 50)

# Shuffle and split
filtered = filtered.shuffle(seed=42)
split = filtered.train_test_split(test_size=0.1)

print(f"Train: {len(split['train'])} samples")
print(f"Test:  {len(split['test'])} samples")

# Save final pipeline output
split.save_to_disk('final_pipeline_dataset')

# Also save as JSON for inspection
train_list = [split['train'][i] for i in range(min(10, len(split['train'])))]
with open('pipeline_sample.json', 'w') as f:
    json.dump(train_list, f, indent=2)

print(f"\nFinal dataset saved!")
print(f"\nSource distribution:")
from collections import Counter
sources = Counter(filtered['source'])
for source, count in sources.items():
    pct = count / len(filtered) * 100
    print(f"  {source}: {count} samples ({pct:.1f}%)")