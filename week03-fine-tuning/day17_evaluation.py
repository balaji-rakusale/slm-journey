import torch
import json
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("=" * 50)
print("PART 1 - Load Base and Fine Tuned Models")
print("=" * 50)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained('gpt2')
base_model.eval()
print("Base model loaded")

# Load fine tuned model
ft_model = AutoModelForCausalLM.from_pretrained('gpt2')
ft_model = PeftModel.from_pretrained(ft_model, './lora_weights')
ft_model.eval()
print("Fine tuned model loaded")

print("\n" + "=" * 50)
print("PART 2 - Perplexity Evaluation")
print("=" * 50)

# Perplexity measures how confused the model is
# Lower perplexity = better model
# Perfect model = perplexity of 1
# Random model = perplexity of vocab_size (50257 for GPT2)

def calculate_perplexity(model, tokenizer, texts, max_length=256):
    total_loss = 0
    count = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length
            )
            labels = inputs['input_ids'].clone()

            outputs = model(
                **inputs,
                labels=labels
            )
            total_loss += outputs.loss.item()
            count += 1

    avg_loss = total_loss / count
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss

# Test texts from our domain
test_texts = [
    "### Instruction:\nClassify this news article into a category.\n\n### Input:\nApple announces new MacBook with M3 chip.\n\n### Response:\nCategory: Technology",
    "### Instruction:\nClassify this news article into a category.\n\n### Input:\nFederal Reserve raises interest rates by 0.25 percent.\n\n### Response:\nCategory: Business",
    "### Instruction:\nClassify this news article into a category.\n\n### Input:\nLionel Messi wins eighth Ballon d'Or award.\n\n### Response:\nCategory: Sports",
    "### Instruction:\nWhat is artificial intelligence?\n\n### Response:\nArtificial intelligence is the simulation of human intelligence by machines.",
    "### Instruction:\nSummarize the following text:\n\n### Input:\nMachine learning is a subset of AI that learns from data.\n\n### Response:\nMachine learning enables computers to learn from data automatically.",
]

print("Calculating perplexity on test samples...")
base_ppl, base_loss = calculate_perplexity(base_model, tokenizer, test_texts)
ft_ppl, ft_loss = calculate_perplexity(ft_model, tokenizer, test_texts)

print(f"\nBase model:")
print(f"  Loss:        {base_loss:.4f}")
print(f"  Perplexity:  {base_ppl:.2f}")

print(f"\nFine tuned model:")
print(f"  Loss:        {ft_loss:.4f}")
print(f"  Perplexity:  {ft_ppl:.2f}")

improvement = (base_ppl - ft_ppl) / base_ppl * 100
print(f"\nPerplexity improvement: {improvement:.1f}%")

print("\n" + "=" * 50)
print("PART 3 - Response Quality Comparison")
print("=" * 50)

def generate_response(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=200
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(prompt):]
    return response.strip()

# Test prompts
test_prompts = [
    {
        "name": "News Classification",
        "prompt": "### Instruction:\nClassify this news article.\n\n### Input:\nNASA launches new Mars rover mission successfully.\n\n### Response:"
    },
    {
        "name": "AI Question",
        "prompt": "### Instruction:\nWhat is machine learning?\n\n### Response:"
    },
    {
        "name": "Summarization",
        "prompt": "### Instruction:\nSummarize the following text:\n\n### Input:\nDeep learning uses neural networks with many layers to learn complex patterns from large amounts of data.\n\n### Response:"
    }
]

print("Comparing base vs fine tuned responses:\n")
for test in test_prompts:
    print(f"Test: {test['name']}")
    print(f"Prompt: {test['prompt'][-80:]}")

    base_response = generate_response(base_model, tokenizer, test['prompt'])
    ft_response = generate_response(ft_model, tokenizer, test['prompt'])

    print(f"Base model:       {base_response[:100]}")
    print(f"Fine tuned model: {ft_response[:100]}")
    print()

print("\n" + "=" * 50)
print("PART 4 - Evaluation Metrics Summary")
print("=" * 50)

print(f"""
Model Comparison Report
=======================
Metric              Base GPT2    Fine Tuned    Improvement
Loss                {base_loss:.4f}       {ft_loss:.4f}        {((base_loss-ft_loss)/base_loss*100):.1f}% better
Perplexity          {base_ppl:.2f}        {ft_ppl:.2f}         {improvement:.1f}% better

What this means:
- Lower loss = model predicts tokens more accurately
- Lower perplexity = model less confused by domain text
- Fine tuned model understands your format better
""")

print("\n" + "=" * 50)
print("PART 5 - Identify Weaknesses")
print("=" * 50)

weakness_prompts = [
    "### Instruction:\nWrite a Python function to sort a list.\n\n### Response:",
    "### Instruction:\nTranslate to French: Hello world.\n\n### Response:",
    "### Instruction:\nWhat is the capital of France?\n\n### Response:",
]

print("Testing out of domain prompts (weaknesses):")
for prompt in weakness_prompts:
    ft_response = generate_response(ft_model, tokenizer, prompt)
    print(f"Prompt:   {prompt[16:60]}")
    print(f"Response: {ft_response[:100]}")
    print()

print("These weaknesses show where more training data is needed.")
print("For enterprise: add domain specific data to fix weaknesses.")