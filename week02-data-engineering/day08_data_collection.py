import requests
from bs4  import BeautifulSoup
from datasets import load_dataset
import json

# PART 1 - Scrape a webpage
print("=" *50)
print("PART 1 - Web Scraping")
print("=" * 50)

# Use headers to mimic a real browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract only paragraph text from main content
content = soup.find('div', {'id': 'mw-content-text'})
paragraphs = content.find_all('p') if content else soup.find_all('p')

clean_text = []
for p in paragraphs:
    text = p.get_text().strip()
    if len(text) > 50:
        clean_text.append(text)

print(f"Total paragraphs scraped: {len(clean_text)}")

if clean_text:
    print(f"\nFirst paragraph:")
    print(clean_text[0][:300])
    print(f"\nTotal characters collected: {sum(len(p) for p in clean_text)}")
    with open('raw_data.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(clean_text))
    print("\nRaw data saved to raw_data.txt")
else:
    print("No paragraphs found - using fallback text")
    clean_text = ["Artificial intelligence is the simulation of human intelligence by machines."]

# PART 2 - Load HuggingFace Dataset
print("\n" + "=" * 50)
print("PART 2 - HuggingFace Datasets")
print("=" * 50)

dataset = load_dataset("ag_news", split="train[:100]")
print(f"Dataset loaded successfully")
print(f"Total samples: {len(dataset)}")
print(f"\nFirst sample:")
print(dataset[0])
print(f"\nFeatures: {dataset.features}")

# PART 3 - Label distribution
print("\n" + "=" * 50)
print("PART 3 - Explore Data")
print("=" * 50)

label_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Technology'}

from collections import Counter
labels = Counter(dataset['label'])
print("Label distribution:")
for label, count in sorted(labels.items()):
    print(f"  {label_names[label]}: {count} samples")

# PART 4 - Save samples safely
print("\n" + "=" * 50)
print("PART 4 - Save Data")
print("=" * 50)

samples = []
for item in dataset:
    samples.append({
        'text': item['text'],
        'label': label_names[item['label']]
    })

with open('news_samples.json', 'w', encoding='utf-8') as f:
    json.dump(samples[:10], f, indent=2)

print(f"Saved 10 samples to news_samples.json")
print("\nFirst sample structure:")
print(json.dumps(samples[0], indent=2))