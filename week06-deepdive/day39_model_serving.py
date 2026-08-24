import torch
import time
import threading
import queue
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional

print("=" * 50)
print("PART 1 - Production Serving Challenges")
print("=" * 50)

print("""
Single Request (What we built so far):
  User sends request
  Model processes it
  User gets response
  Next user waits...

Problem at Scale:
  1000 users send requests simultaneously
  Each waits for all others to finish
  Response time = 1000 x single request time
  Unacceptable for production!

Solutions:
  1. Batching       → process multiple requests together
  2. Async serving  → don't block on each request
  3. KV Cache       → cache attention keys and values
  4. Continuous batching → vLLM approach
  5. Model replicas → run multiple copies
""")

print("=" * 50)
print("PART 2 - Naive vs Batched Inference")
print("=" * 50)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float32
)
model.eval()
print("Model loaded ✅")

test_prompts = [
    "What is machine learning?",
    "What is deep learning?",
    "What is natural language processing?",
    "What is a transformer model?",
]

# Naive inference - one by one
print("\nNaive inference (one by one):")
start = time.time()
naive_responses = []
for prompt in test_prompts:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=50
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    naive_responses.append(response)
naive_time = time.time() - start
print(f"Time: {naive_time:.2f}s for {len(test_prompts)} requests")
print(f"Avg per request: {naive_time/len(test_prompts):.2f}s")

# Batched inference
print("\nBatched inference (all together):")
start = time.time()
tokenizer.padding_side = 'left'
batch_inputs = tokenizer(
    test_prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=50
)
with torch.no_grad():
    batch_outputs = model.generate(
        **batch_inputs,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
batch_responses = [
    tokenizer.decode(output, skip_special_tokens=True)
    for output in batch_outputs
]
batch_time = time.time() - start
print(f"Time: {batch_time:.2f}s for {len(test_prompts)} requests")
print(f"Avg per request: {batch_time/len(test_prompts):.2f}s")
print(f"Speedup: {naive_time/batch_time:.1f}x faster!")

print("\n" + "=" * 50)
print("PART 3 - Request Queue System")
print("=" * 50)

@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    max_tokens: int = 50
    result: Optional[str] = None

class ModelServer:
    def __init__(self, model, tokenizer, batch_size=4):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.request_queue = queue.Queue()
        self.results = {}
        self.is_running = False

    def add_request(self, request: InferenceRequest):
        self.request_queue.put(request)

    def process_batch(self, requests: List[InferenceRequest]):
        prompts = [r.prompt for r in requests]

        self.tokenizer.padding_side = 'left'
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=requests[0].max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        for i, request in enumerate(requests):
            response = self.tokenizer.decode(
                outputs[i],
                skip_special_tokens=True
            )
            request.result = response
            self.results[request.request_id] = response

    def run(self):
        self.is_running = True
        batch = []
        start_wait = time.time()

        while self.is_running:
            try:
                request = self.request_queue.get(timeout=0.1)
                batch.append(request)

                # Process when batch is full or timeout
                if len(batch) >= self.batch_size:
                    self.process_batch(batch)
                    batch = []
                    start_wait = time.time()

            except queue.Empty:
                if batch and (time.time() - start_wait) > 0.5:
                    self.process_batch(batch)
                    batch = []
                    start_wait = time.time()

# Test the server
server = ModelServer(model, tokenizer, batch_size=4)
server_thread = threading.Thread(target=server.run, daemon=True)
server_thread.start()

requests = [
    InferenceRequest(f"req_{i}", prompt)
    for i, prompt in enumerate(test_prompts)
]

print("Sending requests to server...")
start = time.time()
for req in requests:
    server.add_request(req)

time.sleep(5)
server.is_running = False
elapsed = time.time() - start

print(f"Processed {len(server.results)} requests")
print(f"Total time: {elapsed:.2f}s")
print(f"\nSample response:")
if server.results:
    first_result = list(server.results.values())[0]
    print(f"  {first_result[:100]}")

print("\n" + "=" * 50)
print("PART 4 - KV Cache Explained")
print("=" * 50)

print("""
KV Cache = Key Value Cache

Without KV Cache:
  Token 1: compute attention over [token1]
  Token 2: compute attention over [token1, token2]
  Token 3: compute attention over [token1, token2, token3]
  Each new token recomputes ALL previous tokens!

With KV Cache:
  Token 1: compute K,V → store in cache
  Token 2: load K,V from cache + compute new token only
  Token 3: load K,V from cache + compute new token only
  Each new token only computes ITSELF!

Speedup: 10-100x faster for long sequences
Memory: Trades memory for speed

GPT2 has KV cache built in:
  use_cache=True (default)
  past_key_values returned by model
""")

# Demonstrate KV cache
prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors="pt")

# Without cache simulation
start = time.time()
with torch.no_grad():
    output_no_cache = model.generate(
        **inputs,
        max_new_tokens=50,
        use_cache=False,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
no_cache_time = time.time() - start

# With cache
start = time.time()
with torch.no_grad():
    output_with_cache = model.generate(
        **inputs,
        max_new_tokens=50,
        use_cache=True,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
cache_time = time.time() - start

print(f"Without KV cache: {no_cache_time:.2f}s")
print(f"With KV cache:    {cache_time:.2f}s")
print(f"Speedup:          {no_cache_time/cache_time:.1f}x faster")

print("\n" + "=" * 50)
print("PART 5 - Production Stack Comparison")
print("=" * 50)

serving_options = [
    {
        "name": "FastAPI + HuggingFace",
        "throughput": "10-50 req/s",
        "latency": "500ms-2s",
        "setup": "Easy",
        "best_for": "Small scale, demos"
    },
    {
        "name": "vLLM",
        "throughput": "100-500 req/s",
        "latency": "50-200ms",
        "setup": "Medium",
        "best_for": "Production APIs"
    },
    {
        "name": "TGI (HuggingFace)",
        "throughput": "50-200 req/s",
        "latency": "100-500ms",
        "setup": "Medium",
        "best_for": "HuggingFace ecosystem"
    },
    {
        "name": "Ollama",
        "throughput": "1-10 req/s",
        "latency": "1-5s",
        "setup": "Very Easy",
        "best_for": "Local deployment"
    },
    {
        "name": "llama.cpp server",
        "throughput": "5-20 req/s",
        "latency": "200ms-1s",
        "setup": "Medium",
        "best_for": "CPU deployment"
    },
]

print(f"{'Option':<25} {'Throughput':<20} {'Latency':<15} {'Setup':<10} {'Best For'}")
print("-" * 85)
for opt in serving_options:
    print(f"{opt['name']:<25} {opt['throughput']:<20} {opt['latency']:<15} {opt['setup']:<10} {opt['best_for']}")

print("\n" + "=" * 50)
print("PART 6 - Optimization Checklist")
print("=" * 50)

print("""
Before deploying any model check these:

Model Optimization:
  ✅ Quantize to int8 or int4
  ✅ Enable KV cache
  ✅ Use flash attention if available
  ✅ Set model to eval mode

Serving Optimization:
  ✅ Use batching
  ✅ Set max batch size
  ✅ Use async endpoints
  ✅ Add request timeout

Infrastructure:
  ✅ Use GPU if available
  ✅ Pin memory for faster transfer
  ✅ Use multiple workers
  ✅ Add health check endpoint

Monitoring:
  ✅ Log request latency
  ✅ Track token throughput
  ✅ Monitor GPU memory
  ✅ Alert on errors
""")

print("Day 39 Complete ✅")
print("Tomorrow: Advanced fine tuning strategies")