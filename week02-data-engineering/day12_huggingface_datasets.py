from datasets import load_dataset, Dataset, DatasetDict
import json
import pandas as pd

print("=" * 50)
print("PART 1 - Load Multiple Datasets")
print("=" * 50)

# Use smaller datasets instead of Wikipedia
# ag_news we already know from Day 8
news = load_dataset("ag_news", split="train[:100]")
print(f"News samples: {len(news)}")
print(f"Features: {news.features}")
print(f"First sample text: {news[0]['text'][:200]}")

print("\n" + "=" * 50)
print("PART 2 - Dataset Operations")
print("=" * 50)

# Filter dataset - keep only long articles
long_articles = news.filter(lambda x: len(x['text']) > 200)
print(f"Articles over 200 chars: {len(long_articles)}")

# Map operation - clean text
def clean_article(example):
    text = example['text']
    text = text.replace('\n', ' ')
    text = text.strip()
    label_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Technology'}
    return {
        'text': text,
        'label_name': label_names[example['label']]
    }

cleaned_news = long_articles.map(clean_article)
print(f"Cleaned articles: {len(cleaned_news)}")
print(f"Sample cleaned: {cleaned_news[0]['text'][:200]}")

print("\n" + "=" * 50)
print("PART 3 - Convert to Instruction Format")
print("=" * 50)

# Convert news articles to instruction format
def to_instruction_format(example):
    return {
        'instruction': f"What category does this news article belong to?",
        'input': example['text'][:300],
        'output': f"This article belongs to the {example['label_name']} category."
    }

instruction_news = cleaned_news.map(to_instruction_format)
print(f"Instruction samples: {len(instruction_news)}")
print(f"\nSample:")
print(f"Instruction: {instruction_news[0]['instruction']}")
print(f"Input:       {instruction_news[0]['input'][:150]}")
print(f"Output:      {instruction_news[0]['output']}")

print("\n" + "=" * 50)
print("PART 4 - Combine With Our Dataset")
print("=" * 50)

# Load our production dataset from Day 11
with open('production_dataset.json', 'r') as f:
    our_data = json.load(f)

print(f"Our dataset:  {len(our_data)} samples")
print(f"News dataset: {len(instruction_news)} samples")

# Convert our data to HuggingFace Dataset
our_hf_dataset = Dataset.from_list(our_data)
print(f"Our HuggingFace dataset: {len(our_hf_dataset)} samples")

print("\n" + "=" * 50)
print("PART 5 - Dataset Statistics")
print("=" * 50)

# Analyze our dataset with pandas
df = our_hf_dataset.to_pandas()
print(f"Dataset shape: {df.shape}")
print(f"\nColumn stats:")
print(f"  Instruction avg length: {df['instruction'].str.len().mean():.0f} chars")
print(f"  Input avg length:       {df['input'].str.len().mean():.0f} chars")
print(f"  Output avg length:      {df['output'].str.len().mean():.0f} chars")
print(f"\nQuality score distribution:")
print(df['quality_score'].value_counts())

print("\n" + "=" * 50)
print("PART 6 - Save as HuggingFace Dataset")
print("=" * 50)

# Split into train and test
split_dataset = our_hf_dataset.train_test_split(test_size=0.1)
print(f"Train: {len(split_dataset['train'])} samples")
print(f"Test:  {len(split_dataset['test'])} samples")

# Save to disk
split_dataset.save_to_disk('my_ai_dataset')
print(f"\nDataset saved to disk!")
print(f"Ready for fine tuning!")