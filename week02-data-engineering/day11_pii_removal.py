import re
import json

print("=" * 50)
print("PART 1 - What is PII")
print("=" * 50)

# PII = Personally Identifiable Information
# Names, emails, phone numbers, addresses, credit cards
# Training on PII = legal liability for your enterprise clients
# This is why companies pay premium for clean datasets

sample_dirty_text = """
John Smith called us at john.smith@gmail.com
His phone number is 555-123-4567
He lives at 123 Main Street New York
His credit card is 4532-1234-5678-9012
His SSN is 123-45-6789
Meeting scheduled for AI discussion
Neural networks are transforming industries
"""

print("Dirty text with PII:")
print(sample_dirty_text)

print("\n" + "=" * 50)
print("PART 2 - Regex Based PII Removal")
print("=" * 50)


def remove_pii(text):
    # Remove emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

    # Remove credit card numbers
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CREDIT_CARD]', text)

    # Remove SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)

    # Remove IP addresses
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_ADDRESS]', text)

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)

    return text


cleaned = remove_pii(sample_dirty_text)
print("After PII removal:")
print(cleaned)

print("\n" + "=" * 50)
print("PART 3 - Language Detection")
print("=" * 50)


# For enterprise English only datasets
# we need to filter out non English text

def is_english(text):
    # Simple heuristic - check common English words
    english_words = {'the', 'is', 'are', 'was', 'were', 'have',
                     'has', 'had', 'will', 'would', 'could', 'should',
                     'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}

    words = set(text.lower().split())
    matches = len(words.intersection(english_words))
    return matches >= 2


test_texts = [
    "Artificial intelligence is transforming the world",
    "La inteligencia artificial está transformando el mundo",
    "人工知能は世界を変えています",
    "Machine learning models are trained on large datasets",
]

print("Language detection results:")
for text in test_texts:
    result = "English" if is_english(text) else "Non-English"
    print(f"  {result}: {text[:50]}")

print("\n" + "=" * 50)
print("PART 4 - Text Quality Scoring")
print("=" * 50)


def quality_score(text):
    score = 100  # start with perfect score

    # Penalty 1 - Too short
    if len(text) < 100:
        score -= 30

    # Penalty 2 - Too many uppercase (SHOUTING TEXT)
    upper_ratio = sum(c.isupper() for c in text) / max(len(text), 1)
    if upper_ratio > 0.3:
        score -= 20

    # Penalty 3 - Too many punctuation marks
    punct_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / max(len(text), 1)
    if punct_ratio > 0.2:
        score -= 20

    # Penalty 4 - No spaces (likely garbled)
    if ' ' not in text:
        score -= 50

    # Penalty 5 - Repetitive text
    words = text.split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            score -= 25

    return max(score, 0)


test_samples = [
    "Artificial intelligence is transforming industries worldwide through machine learning",
    "BUY NOW!!! CLICK HERE!!! AMAZING DEAL!!!",
    "the the the the the the the the the the",
    "AI",
    "Neural networks learn patterns from data through a process called backpropagation",
]

print("Quality scores:")
for sample in test_samples:
    score = quality_score(sample)
    status = "Keep" if score >= 60 else "Remove"
    print(f"  Score {score:3d} {status}: {sample[:60]}")

print("\n" + "=" * 50)
print("PART 5 - Apply Full Pipeline to Our Dataset")
print("=" * 50)

# Load our instruction dataset
with open('train_instructions.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

print(f"Loaded {len(train_data)} training samples")

# Apply full pipeline
final_dataset = []
removed = 0

for sample in train_data:
    # Step 1 - Remove PII from all fields
    sample['instruction'] = remove_pii(sample['instruction'])
    sample['input'] = remove_pii(sample['input'])
    sample['output'] = remove_pii(sample['output'])

    # Step 2 - Language check on output
    if not is_english(sample['output']):
        removed += 1
        continue

    # Step 3 - Quality score on output
    score = quality_score(sample['output'])
    if score < 60:
        removed += 1
        continue

    # Step 4 - Add quality score as metadata
    sample['quality_score'] = score
    final_dataset.append(sample)

print(f"Original samples: {len(train_data)}")
print(f"Removed samples:  {removed}")
print(f"Final samples:    {len(final_dataset)}")

# Save final production ready dataset
with open('production_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(final_dataset, f, indent=2, ensure_ascii=False)

print(f"\nProduction dataset saved!")
print(f"\nSample from final dataset:")
print(json.dumps(final_dataset[0], indent=2))