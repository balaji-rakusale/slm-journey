import json
import random
from datetime import datetime

print("=" * 50)
print("PART 1 - Why Domain Fine Tuning")
print("=" * 50)

print("""
General Model Problems:
  Ask about legal contracts → vague answers
  Ask about medical terms   → generic responses
  Ask about your codebase   → no knowledge

Domain Fine Tuning Solution:
  Train on legal documents  → legal expert
  Train on medical records  → medical assistant
  Train on your codebase    → code expert

Key Insight:
  General model knows EVERYTHING superficially
  Domain model knows ONE THING deeply

  For enterprise: deep beats broad every time
""")

print("=" * 50)
print("PART 2 - Build Domain Dataset")
print("=" * 50)

# Legal domain dataset
legal_dataset = [
    {
        "instruction": "What is a force majeure clause?",
        "output": "A force majeure clause excuses a party from performance obligations when extraordinary events beyond their control prevent fulfillment, such as natural disasters, wars, or pandemics."
    },
    {
        "instruction": "Explain indemnification in contracts.",
        "output": "Indemnification is a contractual obligation where one party agrees to compensate the other for losses, damages, or legal costs arising from specific events or breaches."
    },
    {
        "instruction": "What is liquidated damages?",
        "output": "Liquidated damages are predetermined compensation amounts specified in a contract that parties agree to pay if certain breaches occur, avoiding costly damage assessment litigation."
    },
    {
        "instruction": "Define breach of contract.",
        "output": "A breach of contract occurs when one party fails to fulfill their contractual obligations without legal justification, entitling the non-breaching party to remedies."
    },
    {
        "instruction": "What is a non-disclosure agreement?",
        "output": "An NDA is a legally binding contract establishing confidentiality between parties, prohibiting disclosure of specified confidential information to third parties."
    },
]

# Medical domain dataset
medical_dataset = [
    {
        "instruction": "What is hypertension?",
        "output": "Hypertension is persistently elevated blood pressure above 130/80 mmHg, a major risk factor for cardiovascular disease, stroke, and kidney damage requiring lifestyle changes or medication."
    },
    {
        "instruction": "Explain Type 2 diabetes.",
        "output": "Type 2 diabetes is a metabolic disorder where the body becomes resistant to insulin or produces insufficient insulin, causing elevated blood glucose levels requiring dietary management or medication."
    },
    {
        "instruction": "What is an ECG?",
        "output": "An electrocardiogram records the electrical activity of the heart through electrodes placed on the skin, used to diagnose arrhythmias, heart attacks, and other cardiac conditions."
    },
    {
        "instruction": "Define BMI.",
        "output": "Body Mass Index is a screening tool calculated by dividing weight in kilograms by height in meters squared, used to categorize underweight, normal, overweight, and obese ranges."
    },
    {
        "instruction": "What is informed consent?",
        "output": "Informed consent is a patient's voluntary agreement to medical treatment after receiving complete information about diagnosis, proposed treatment, risks, benefits, and alternatives."
    },
]

# Finance domain dataset
finance_dataset = [
    {
        "instruction": "What is EBITDA?",
        "output": "EBITDA stands for Earnings Before Interest Taxes Depreciation and Amortization, a metric measuring core operational profitability before accounting and financing decisions."
    },
    {
        "instruction": "Explain dollar cost averaging.",
        "output": "Dollar cost averaging is an investment strategy of regularly investing fixed amounts regardless of price, reducing impact of volatility by buying more shares when prices are low."
    },
    {
        "instruction": "What is a P/E ratio?",
        "output": "Price to Earnings ratio compares a company's stock price to its earnings per share, indicating how much investors pay per dollar of earnings and relative market valuation."
    },
    {
        "instruction": "Define working capital.",
        "output": "Working capital is the difference between current assets and current liabilities, measuring a company's short term liquidity and operational efficiency."
    },
    {
        "instruction": "What is compound interest?",
        "output": "Compound interest is interest calculated on both the initial principal and accumulated interest, causing exponential growth over time often called the eighth wonder of the world."
    },
]

print(f"Legal dataset:   {len(legal_dataset)} examples")
print(f"Medical dataset: {len(medical_dataset)} examples")
print(f"Finance dataset: {len(finance_dataset)} examples")

print("\n" + "=" * 50)
print("PART 3 - Format For Fine Tuning")
print("=" * 50)


def format_for_training(examples, domain):
    formatted = []
    for ex in examples:
        text = f"""### Instruction:
{ex['instruction']}

### Response:
{ex['output']}"""
        formatted.append({
            "text": text,
            "domain": domain,
            "word_count": len(ex['output'].split())
        })
    return formatted


legal_formatted = format_for_training(legal_dataset, "legal")
medical_formatted = format_for_training(medical_dataset, "medical")
finance_formatted = format_for_training(finance_dataset, "finance")

all_domain_data = legal_formatted + medical_formatted + finance_formatted
random.shuffle(all_domain_data)

print(f"Total domain samples: {len(all_domain_data)}")
print(f"\nSample formatted example:")
print(all_domain_data[0]['text'])

print("\n" + "=" * 50)
print("PART 4 - Data Quality Analysis")
print("=" * 50)

from collections import Counter

domain_counts = Counter(ex['domain'] for ex in all_domain_data)
word_counts = [ex['word_count'] for ex in all_domain_data]

print("Domain distribution:")
for domain, count in domain_counts.items():
    pct = count / len(all_domain_data) * 100
    bar = "█" * int(pct / 5)
    print(f"  {domain:<10} {bar:<20} {count} samples ({pct:.0f}%)")

print(f"\nWord count statistics:")
print(f"  Min:  {min(word_counts)} words")
print(f"  Max:  {max(word_counts)} words")
print(f"  Avg:  {sum(word_counts) / len(word_counts):.1f} words")

print("\n" + "=" * 50)
print("PART 5 - Domain Mixing Strategy")
print("=" * 50)

print("""
Domain Mixing Best Practices:

Pure Domain Fine Tuning:
  100% legal data
  ✅ Best legal performance
  ❌ Forgets general knowledge
  ❌ Fails on non-legal questions

Mixed Fine Tuning (Recommended):
  70% domain data
  30% general data
  ✅ Strong domain performance
  ✅ Keeps general knowledge
  ✅ Handles diverse questions

Multi Domain Fine Tuning:
  33% legal + 33% medical + 33% finance
  ✅ Handles all three domains
  ⚠️ Slightly weaker per domain
  ✅ Most versatile
""")


# Create mixed dataset
def create_mixed_dataset(domain_data, general_ratio=0.3):
    general_examples = [
        {
            "text": "### Instruction:\nWhat is machine learning?\n\n### Response:\nMachine learning enables computers to learn from data automatically.",
            "domain": "general"},
        {
            "text": "### Instruction:\nExplain neural networks.\n\n### Response:\nNeural networks are computational systems inspired by the human brain that learn patterns.",
            "domain": "general"},
        {
            "text": "### Instruction:\nWhat is deep learning?\n\n### Response:\nDeep learning uses neural networks with many layers to learn complex patterns from data.",
            "domain": "general"},
    ]

    n_general = max(1, int(len(domain_data) * general_ratio))
    selected_general = random.choices(general_examples, k=n_general)
    mixed = domain_data + selected_general
    random.shuffle(mixed)
    return mixed


mixed_dataset = create_mixed_dataset(all_domain_data)
print(f"Mixed dataset: {len(mixed_dataset)} total samples")
print(f"Domain samples: {len(all_domain_data)}")
print(f"General samples: {len(mixed_dataset) - len(all_domain_data)}")

print("\n" + "=" * 50)
print("PART 6 - Save Dataset")
print("=" * 50)

output = {
    "metadata": {
        "created": datetime.now().strftime("%Y-%m-%d"),
        "total_samples": len(mixed_dataset),
        "domains": list(domain_counts.keys()),
        "avg_response_words": round(sum(word_counts) / len(word_counts), 1)
    },
    "data": mixed_dataset
}

with open('domain_dataset.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Dataset saved ✅")
print(f"Total samples: {len(mixed_dataset)}")
print(f"Domains: {list(domain_counts.keys())}")
print(f"Ready for fine tuning on Colab!")

print("\nDay 44 Complete ✅")
print("Tomorrow: Evaluation and benchmarking!")