import re
import json
from collections import Counter

# Load the raw data we scraped yesterday
with open('raw_data.txt', 'r', encoding='utf-8') as f:
    raw_paragraphs = f.read().split('\n')

print("=" * 50)
print("PART 1 - Raw Data Inspection")
print("=" * 50)
print(f"Total raw paragraphs: {len(raw_paragraphs)}")
print(f"Total raw characters: {sum(len(p) for p in raw_paragraphs)}")
print(f"\nSample raw paragraph:")
print(raw_paragraphs[0][:200])

# Show dirty data examples
print(f"\nDirty examples:")
for p in raw_paragraphs[:20]:
    if '[' in p or '{' in p or len(p) < 50:
        print(f"  → {p[:100]}")

print("\n" + "=" * 50)
print("PART 2 - Cleaning Pipeline")
print("=" * 50)


def clean_text(text):
    # Step 1 - Remove citation markers like [1], [2], [citation needed]
    text = re.sub(r'\[[^\]]*\]', '', text)

    # Step 2 - Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)

    # Step 3 - Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Step 4 - Strip leading and trailing spaces
    text = text.strip()

    return text


def is_quality_paragraph(text):
    # Filter 1 - Too short
    if len(text) < 100:
        return False

    # Filter 2 - Too many numbers (likely a table or stats)
    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    if digit_ratio > 0.3:
        return False

    # Filter 3 - No proper sentence structure
    if '.' not in text:
        return False

    # Filter 4 - Too many special characters
    special_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / len(text)
    if special_ratio > 0.3:
        return False

    return True


# Apply cleaning pipeline
cleaned_paragraphs = []
removed_count = 0

for para in raw_paragraphs:
    # Step 1 - Clean the text
    cleaned = clean_text(para)

    # Step 2 - Quality filter
    if is_quality_paragraph(cleaned):
        cleaned_paragraphs.append(cleaned)
    else:
        removed_count += 1

print(f"Original paragraphs: {len(raw_paragraphs)}")
print(f"Removed paragraphs:  {removed_count}")
print(f"Clean paragraphs:    {len(cleaned_paragraphs)}")
print(f"Retention rate:      {len(cleaned_paragraphs) / len(raw_paragraphs) * 100:.1f}%")

print(f"\nSample cleaned paragraph:")
print(cleaned_paragraphs[0][:300])

print("\n" + "=" * 50)
print("PART 3 - Deduplication")
print("=" * 50)

# Remove exact duplicates
before_dedup = len(cleaned_paragraphs)
cleaned_paragraphs = list(set(cleaned_paragraphs))
after_dedup = len(cleaned_paragraphs)

print(f"Before deduplication: {before_dedup}")
print(f"After deduplication:  {after_dedup}")
print(f"Duplicates removed:   {before_dedup - after_dedup}")

print("\n" + "=" * 50)
print("PART 4 - Dataset Statistics")
print("=" * 50)

# Word frequency analysis
all_text = ' '.join(cleaned_paragraphs)
words = all_text.lower().split()
word_freq = Counter(words)

print(f"Total words: {len(words)}")
print(f"Unique words: {len(word_freq)}")
print(f"\nTop 10 most common words:")
for word, count in word_freq.most_common(10):
    print(f"  '{word}': {count} times")

print("\n" + "=" * 50)
print("PART 5 - Save Clean Dataset")
print("=" * 50)

# Save as JSON with metadata
dataset = {
    'metadata': {
        'source': 'Wikipedia - Artificial Intelligence',
        'total_paragraphs': len(cleaned_paragraphs),
        'total_words': len(words),
        'total_chars': len(all_text)
    },
    'data': cleaned_paragraphs
}

with open('clean_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Clean dataset saved to clean_dataset.json")
print(f"\nFinal dataset stats:")
print(f"  Paragraphs: {len(cleaned_paragraphs)}")
print(f"  Words:      {len(words)}")
print(f"  Characters: {len(all_text)}")