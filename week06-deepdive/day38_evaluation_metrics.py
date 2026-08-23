import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
from collections import Counter

print("=" * 50)
print("PART 1 - Why Evaluation Matters")
print("=" * 50)

print("""
The Problem:
  You fine tune a model
  Loss goes down — looks good!
  But is the model actually useful?
  Loss alone doesn't tell you.

Real Evaluation Needs:
  → Does it answer correctly?
  → Does it stay on topic?
  → Is it factually accurate?
  → Does it follow instructions?
  → Is it better than base model?

Evaluation Methods:
  1. BLEU Score    → overlap with reference
  2. ROUGE Score   → recall based overlap
  3. Perplexity    → how confused is model
  4. Human eval    → most accurate, most expensive
  5. LLM as judge  → use GPT4 to evaluate
""")

print("=" * 50)
print("PART 2 - BLEU Score")
print("=" * 50)


def compute_bleu(reference, hypothesis, max_n=4):
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i + n])
            for i in range(len(ref_tokens) - n + 1)
        )
        hyp_ngrams = Counter(
            tuple(hyp_tokens[i:i + n])
            for i in range(len(hyp_tokens) - n + 1)
        )

        matches = sum(
            min(count, ref_ngrams[gram])
            for gram, count in hyp_ngrams.items()
        )
        total = sum(hyp_ngrams.values())

        if total == 0:
            scores.append(0)
        else:
            scores.append(matches / total)

    # Brevity penalty
    bp = min(1.0, len(hyp_tokens) / len(ref_tokens))

    # Geometric mean
    if min(scores) == 0:
        return 0.0

    log_avg = sum(math.log(s) for s in scores) / len(scores)
    bleu = bp * math.exp(log_avg)
    return round(bleu * 100, 2)


# Test BLEU
reference = "Machine learning is a subset of artificial intelligence that learns from data"
hypotheses = [
    "Machine learning is a subset of artificial intelligence that learns from data",  # perfect
    "Machine learning is part of AI and uses data to learn patterns",  # good
    "Deep learning uses neural networks with multiple layers",  # off topic
    "The weather is nice today",  # completely wrong
]

print("BLEU Score Examples:")
print(f"Reference: {reference}\n")
for hyp in hypotheses:
    score = compute_bleu(reference, hyp)
    quality = "✅ Excellent" if score > 50 else "👍 Good" if score > 20 else "⚠️ Poor" if score > 5 else "❌ Bad"
    print(f"Score: {score:6.2f} {quality}")
    print(f"Hyp:   {hyp[:70]}")
    print()

print("=" * 50)
print("PART 3 - ROUGE Score")
print("=" * 50)


def compute_rouge_l(reference, hypothesis):
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    # Longest Common Subsequence
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]

    precision = lcs / n if n > 0 else 0
    recall = lcs / m if m > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return round(f1 * 100, 2)


print("ROUGE-L Score Examples:")
print(f"Reference: {reference}\n")
for hyp in hypotheses:
    score = compute_rouge_l(reference, hyp)
    quality = "✅ Excellent" if score > 60 else "👍 Good" if score > 30 else "⚠️ Poor" if score > 10 else "❌ Bad"
    print(f"Score: {score:6.2f} {quality}")
    print(f"Hyp:   {hyp[:70]}")
    print()

print("=" * 50)
print("PART 4 - Perplexity Evaluation")
print("=" * 50)

print("Loading GPT2...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()
print("Model loaded ✅")


def compute_perplexity(model, tokenizer, text, max_length=256):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    return round(math.exp(outputs.loss.item()), 2)


test_texts = [
    "Machine learning is a subset of artificial intelligence.",
    "The quick brown fox jumps over the lazy dog.",
    "Xkdj fjkds lkjds fkjsd lkjfds lkjfds.",  # gibberish
    "Neural networks learn patterns from training data.",
    "Artificial intelligence transforms how businesses operate.",
]

print("\nPerplexity scores (lower = better):")
for text in test_texts:
    ppl = compute_perplexity(model, tokenizer, text)
    quality = "✅ Natural" if ppl < 100 else "👍 Okay" if ppl < 500 else "❌ Unnatural"
    print(f"  PPL {ppl:8.2f} {quality}: {text[:60]}")

print("\n" + "=" * 50)
print("PART 5 - LLM As Judge")
print("=" * 50)

print("""
LLM As Judge — Modern Evaluation:

Instead of BLEU/ROUGE:
  Use a powerful LLM (GPT4/Claude) to evaluate
  your fine tuned model responses

Prompt template:
  "Rate this response from 1-10 for:
   - Accuracy
   - Relevance
   - Clarity
   - Completeness

   Question: {question}
   Response: {response}

   Provide score and explanation."

Why this works:
  → Captures nuance BLEU misses
  → Aligns with human judgment
  → Scalable — evaluate 1000s of responses
  → Identifies specific weaknesses

Cost:
  GPT4 evaluation: ~$0.01 per response
  1000 responses:  ~$10 total
  Worth it for enterprise projects
""")

print("=" * 50)
print("PART 6 - Build Evaluation Suite")
print("=" * 50)


# Complete evaluation function
def evaluate_model_response(question, reference_answer, model_answer):
    bleu = compute_bleu(reference_answer, model_answer)
    rouge = compute_rouge_l(reference_answer, model_answer)
    ppl = compute_perplexity(model, tokenizer, model_answer)

    # Simple keyword matching score
    ref_words = set(reference_answer.lower().split())
    ans_words = set(model_answer.lower().split())
    keyword_overlap = len(ref_words.intersection(ans_words)) / len(ref_words) * 100

    overall = (bleu * 0.3 + rouge * 0.3 + keyword_overlap * 0.4)

    return {
        "bleu": bleu,
        "rouge_l": rouge,
        "perplexity": ppl,
        "keyword_overlap": round(keyword_overlap, 2),
        "overall_score": round(overall, 2)
    }


# Test evaluation suite
eval_cases = [
    {
        "question": "What is machine learning?",
        "reference": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
        "model_answer": "Machine learning allows computers to learn patterns from data automatically using statistical algorithms."
    },
    {
        "question": "What is deep learning?",
        "reference": "Deep learning uses neural networks with multiple layers to learn complex patterns from large datasets.",
        "model_answer": "The weather today is sunny and warm with light breeze."
    },
]

print("Evaluation Suite Results:")
print()
for case in eval_cases:
    metrics = evaluate_model_response(
        case['question'],
        case['reference'],
        case['model_answer']
    )
    print(f"Question: {case['question']}")
    print(f"Reference: {case['reference'][:60]}...")
    print(f"Answer:    {case['model_answer'][:60]}...")
    print(f"Metrics:")
    print(f"  BLEU:            {metrics['bleu']}")
    print(f"  ROUGE-L:         {metrics['rouge_l']}")
    print(f"  Perplexity:      {metrics['perplexity']}")
    print(f"  Keyword Overlap: {metrics['keyword_overlap']}%")
    print(f"  Overall Score:   {metrics['overall_score']}/100")
    print()

print("Day 38 Complete ✅")
print("Tomorrow: Model serving and optimization")