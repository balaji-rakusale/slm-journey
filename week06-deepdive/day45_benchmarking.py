import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 50)
print("PART 1 - What Are Benchmarks")
print("=" * 50)

print("""
Why Benchmarks Matter:
  "My model is better" — proves nothing
  "My model scores 72 on MMLU" — proves something

Standard LLM Benchmarks:

MMLU (Massive Multitask Language Understanding)
  → 57 subjects: math, law, medicine, history
  → Multiple choice questions
  → Tests knowledge breadth
  → GPT4: 86%, Llama3 8B: 68%

HellaSwag
  → Sentence completion tasks
  → Tests commonsense reasoning
  → GPT4: 95%, Llama3 8B: 82%

TruthfulQA
  → Tests if model gives truthful answers
  → Avoids common misconceptions
  → GPT4: 59%, Llama3 8B: 44%

HumanEval
  → Python coding tasks
  → Tests code generation quality
  → GPT4: 87%, Llama3 8B: 62%

ARC (AI2 Reasoning Challenge)
  → Science questions grade 3-9
  → Tests reasoning ability
  → GPT4: 96%, Llama3 8B: 82%
""")

print("=" * 50)
print("PART 2 - Build Custom Benchmark")
print("=" * 50)

# Custom benchmark for our domain dataset
benchmark_questions = [
    # Legal domain
    {
        "domain": "legal",
        "question": "What is a force majeure clause?",
        "reference": "A force majeure clause excuses a party from performance when extraordinary events beyond their control prevent fulfillment.",
        "keywords": ["force majeure", "extraordinary", "control", "performance"]
    },
    {
        "domain": "legal",
        "question": "Define breach of contract.",
        "reference": "A breach of contract occurs when one party fails to fulfill their contractual obligations without legal justification.",
        "keywords": ["breach", "contract", "obligations", "remedies"]
    },
    # Medical domain
    {
        "domain": "medical",
        "question": "What is hypertension?",
        "reference": "Hypertension is persistently elevated blood pressure above 130/80 mmHg.",
        "keywords": ["blood pressure", "elevated", "cardiovascular", "130/80"]
    },
    {
        "domain": "medical",
        "question": "What is informed consent?",
        "reference": "Informed consent is a patient's voluntary agreement to medical treatment after receiving complete information.",
        "keywords": ["patient", "voluntary", "agreement", "information"]
    },
    # Finance domain
    {
        "domain": "finance",
        "question": "What is EBITDA?",
        "reference": "EBITDA stands for Earnings Before Interest Taxes Depreciation and Amortization.",
        "keywords": ["earnings", "interest", "taxes", "depreciation", "amortization"]
    },
    {
        "domain": "finance",
        "question": "What is a P/E ratio?",
        "reference": "Price to Earnings ratio compares a company's stock price to its earnings per share.",
        "keywords": ["price", "earnings", "stock", "valuation"]
    },
    # General AI
    {
        "domain": "general",
        "question": "What is machine learning?",
        "reference": "Machine learning enables computers to learn from data without explicit programming.",
        "keywords": ["computers", "learn", "data", "algorithms"]
    },
    {
        "domain": "general",
        "question": "What is a transformer?",
        "reference": "Transformers use attention mechanisms to process sequences in parallel.",
        "keywords": ["attention", "sequence", "parallel", "mechanism"]
    },
]

print(f"Benchmark created: {len(benchmark_questions)} questions")
from collections import Counter
domain_dist = Counter(q['domain'] for q in benchmark_questions)
for domain, count in domain_dist.items():
    print(f"  {domain}: {count} questions")

print("\n" + "=" * 50)
print("PART 3 - Evaluation Functions")
print("=" * 50)

def keyword_score(response, keywords):
    response_lower = response.lower()
    matched = sum(1 for kw in keywords if kw.lower() in response_lower)
    return round(matched / len(keywords) * 100, 1)

def length_score(response, min_words=10, max_words=100):
    word_count = len(response.split())
    if word_count < min_words:
        return 0
    elif word_count > max_words:
        return 70
    else:
        return 100

def relevance_score(question, response):
    q_words = set(question.lower().split())
    r_words = set(response.lower().split())
    overlap = len(q_words.intersection(r_words))
    return min(100, overlap * 15)

def combined_score(response, question, keywords):
    kw = keyword_score(response, keywords)
    ln = length_score(response)
    rel = relevance_score(question, response)
    return round(kw * 0.5 + ln * 0.3 + rel * 0.2, 1)

print("Scoring functions:")
test_response = "Hypertension is elevated blood pressure above 130/80 requiring cardiovascular monitoring."
test_keywords = ["blood pressure", "elevated", "cardiovascular", "130/80"]
print(f"  Keyword score:   {keyword_score(test_response, test_keywords)}")
print(f"  Length score:    {length_score(test_response)}")
print(f"  Relevance score: {relevance_score('What is hypertension?', test_response)}")
print(f"  Combined score:  {combined_score(test_response, 'What is hypertension?', test_keywords)}")

print("\n" + "=" * 50)
print("PART 4 - Run Benchmark on GPT2")
print("=" * 50)

print("Loading GPT2...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()
print("Model loaded ✅")

def generate_answer(question, max_tokens=80):
    prompt = f"### Instruction:\n{question}\n\n### Response:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=200
    )
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3
        )
    elapsed = time.time() - start
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    return response, elapsed

print("\nRunning benchmark...")
results = []
for q in benchmark_questions:
    response, elapsed = generate_answer(q['question'])
    score = combined_score(response, q['question'], q['keywords'])
    results.append({
        "domain": q['domain'],
        "question": q['question'],
        "response": response[:100],
        "score": score,
        "time": round(elapsed, 2)
    })

print("\n" + "=" * 50)
print("PART 5 - Benchmark Results")
print("=" * 50)

print(f"\n{'Domain':<10} {'Score':<8} {'Time':<8} {'Question'}")
print("-" * 65)
for r in results:
    bar = "█" * int(r['score'] / 10)
    print(f"{r['domain']:<10} {r['score']:<8} {r['time']:<8} {r['question'][:35]}")

domain_scores = {}
for r in results:
    if r['domain'] not in domain_scores:
        domain_scores[r['domain']] = []
    domain_scores[r['domain']].append(r['score'])

print(f"\nDomain Average Scores:")
total_scores = []
for domain, scores in domain_scores.items():
    avg = sum(scores) / len(scores)
    total_scores.extend(scores)
    bar = "█" * int(avg / 5)
    print(f"  {domain:<10} {bar:<20} {avg:.1f}/100")

overall = sum(total_scores) / len(total_scores)
print(f"\n  Overall GPT2 score: {overall:.1f}/100")
print(f"  (After domain fine tuning this should improve significantly)")

print("\n" + "=" * 50)
print("PART 6 - Benchmark Summary")
print("=" * 50)

benchmark_summary = {
    "model": "gpt2",
    "date": time.strftime("%Y-%m-%d"),
    "total_questions": len(results),
    "overall_score": round(overall, 1),
    "domain_scores": {
        domain: round(sum(scores)/len(scores), 1)
        for domain, scores in domain_scores.items()
    },
    "avg_response_time": round(
        sum(r['time'] for r in results) / len(results), 2
    )
}

with open('benchmark_results.json', 'w') as f:
    json.dump(benchmark_summary, f, indent=2)

print(json.dumps(benchmark_summary, indent=2))
print("\nBenchmark saved ✅")
print("\nDay 45 Complete ✅")
print("Tomorrow: Week 6 review + planning Week 7!")