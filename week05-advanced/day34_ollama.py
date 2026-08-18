import requests
import json
import time

print("=" * 50)
print("PART 1 - What is Ollama")
print("=" * 50)

print("""
Ollama = Local LLM Runtime

Before Ollama:
  → Download model weights manually
  → Convert to GGUF format
  → Install llama.cpp
  → Configure everything manually
  → Hours of setup

With Ollama:
  → ollama pull phi3:mini
  → ollama run phi3:mini
  → Done in 5 minutes

Why enterprise clients love this:
  → Runs on their own hardware
  → Zero data leaves their network
  → No API costs
  → Works offline
  → Easy to update models
""")

print("=" * 50)
print("PART 2 - Ollama API")
print("=" * 50)

# Ollama runs a local API on port 11434
OLLAMA_URL = "http://localhost:11434"

def check_ollama():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return True, models
        return False, []
    except:
        return False, []

is_running, available_models = check_ollama()

if is_running:
    print("Ollama is running ✅")
    print(f"Available models: {len(available_models)}")
    for model in available_models:
        print(f"  → {model['name']}")
else:
    print("Ollama not running ❌")
    print("Start with: ollama serve")
    print("Pull model: ollama pull phi3:mini")

print("\n" + "=" * 50)
print("PART 3 - Generate Via API")
print("=" * 50)

def ollama_generate(prompt, model="phi3:mini", stream=False):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.7,
            "num_predict": 100,
        }
    }

    start_time = time.time()
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            end_time = time.time()
            return {
                "response": result.get('response', ''),
                "time": round(end_time - start_time, 2),
                "tokens": result.get('eval_count', 0),
                "success": True
            }
        else:
            return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

if is_running and available_models:
    model_name = available_models[0]['name']
    print(f"Testing with model: {model_name}")

    test_prompts = [
        "### Instruction:\nClassify this news: Apple launches iPhone 16.\n\n### Response:",
        "### Instruction:\nWhat is machine learning in one sentence?\n\n### Response:",
    ]

    for prompt in test_prompts:
        result = ollama_generate(prompt, model=model_name)
        if result['success']:
            print(f"\nPrompt:   {prompt[-60:]}")
            print(f"Response: {result['response'][:150]}")
            print(f"Time:     {result['time']}s")
            print(f"Tokens:   {result['tokens']}")
        else:
            print(f"Error: {result['error']}")
else:
    print("Skipping API test — Ollama not running")
    print("Install Ollama and run: ollama pull phi3:mini")

print("\n" + "=" * 50)
print("PART 4 - Ollama vs Cloud API Comparison")
print("=" * 50)

comparison = [
    {
        "option": "OpenAI GPT4",
        "cost": "$30 per 1M tokens",
        "privacy": "Data sent to OpenAI",
        "offline": "No",
        "setup": "API key only",
        "best_for": "Quick prototypes"
    },
    {
        "option": "Ollama Local",
        "cost": "$0 per token",
        "privacy": "Data stays local",
        "offline": "Yes",
        "setup": "One time install",
        "best_for": "Enterprise privacy"
    },
    {
        "option": "Self hosted GPU",
        "cost": "$0.5-2/hr GPU",
        "privacy": "Your cloud",
        "offline": "No",
        "setup": "Complex",
        "best_for": "High volume"
    },
]

for opt in comparison:
    print(f"\n{opt['option']}:")
    for key, value in opt.items():
        if key != 'option':
            print(f"  {key}: {value}")

print("\n" + "=" * 50)
print("PART 5 - Available Models in Ollama")
print("=" * 50)

ollama_models = [
    {"name": "phi3:mini", "size": "2.3 GB", "params": "3.8B", "best_for": "Fast responses, limited RAM"},
    {"name": "llama3:8b", "size": "4.7 GB", "params": "8B", "best_for": "Best open source quality"},
    {"name": "mistral:7b", "size": "4.1 GB", "params": "7B", "best_for": "Code and reasoning"},
    {"name": "gemma2:9b", "size": "5.4 GB", "params": "9B", "best_for": "Multilingual tasks"},
    {"name": "mixtral:8x7b", "size": "26 GB", "params": "46B MoE", "best_for": "Best quality locally"},
]

print(f"{'Model':<20} {'Size':<10} {'Params':<12} {'Best For'}")
print("-" * 65)
for m in ollama_models:
    print(f"{m['name']:<20} {m['size']:<10} {m['params']:<12} {m['best_for']}")

print("\n" + "=" * 50)
print("PART 6 - Enterprise Integration Code")
print("=" * 50)

enterprise_code = '''
# Drop-in replacement for OpenAI API
# Just change the URL — same code works!

# Before (OpenAI):
import openai
client = openai.OpenAI(api_key="sk-...")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

# After (Ollama — zero cost, private):
import openai
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # any string works
)
response = client.chat.completions.create(
    model="llama3:8b",
    messages=[{"role": "user", "content": prompt}]
)

# Same code. Zero cost. Complete privacy.
# This is your enterprise pitch in code.
'''

print(enterprise_code)

print("Day 34 Complete ✅")
print("Tomorrow: Build complete local AI assistant")