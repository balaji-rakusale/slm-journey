import requests
import json
import time
from sentence_transformers import SentenceTransformer
import chromadb

print("=" * 50)
print("PART 1 - Local AI Assistant Architecture")
print("=" * 50)

print("""
Complete Local AI Assistant Stack:

User Query
    ↓
Sentence Transformer (embed query)
    ↓
ChromaDB (semantic search)
    ↓
Retrieved Context
    ↓
Ollama (local LLM generation)
    ↓
Final Answer

100% local. Zero cloud. Zero cost.
Perfect for enterprise clients.
""")

print("=" * 50)
print("PART 2 - Setup Knowledge Base")
print("=" * 50)

# Enterprise knowledge base
knowledge_base = [
    "Our company offers three pricing tiers: Starter at $99/month, Professional at $299/month, and Enterprise at custom pricing.",
    "The refund policy allows full refunds within 30 days of purchase. After 30 days partial refunds may be considered.",
    "Technical support is available Monday to Friday 9am to 6pm IST via email at support@company.com.",
    "The product supports Windows 10+, MacOS 12+, and Ubuntu 20.04+. Mobile apps available on iOS and Android.",
    "Data is encrypted at rest using AES-256 and in transit using TLS 1.3. SOC2 Type II certified.",
    "The API rate limit is 1000 requests per minute for Professional tier and 10000 for Enterprise tier.",
    "Integration available with Slack, Microsoft Teams, Salesforce, HubSpot, and Zapier.",
    "Free trial available for 14 days with no credit card required. All features included in trial.",
    "The platform uses GPT4 and custom fine tuned models depending on the task type selected.",
    "Annual billing saves 20% compared to monthly billing. Volume discounts available above 10 seats.",
]

# Load embedding model
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded ✅")

# Build vector database
client = chromadb.Client()
collection = client.create_collection("company_kb")

embeddings = embedder.encode(knowledge_base).tolist()
collection.add(
    documents=knowledge_base,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(knowledge_base))]
)

print(f"Knowledge base built: {len(knowledge_base)} documents ✅")

print("\n" + "=" * 50)
print("PART 3 - Local RAG Pipeline")
print("=" * 50)

OLLAMA_URL = "http://localhost:11434"

def check_ollama():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = response.json().get('models', [])
        return True, models[0]['name'] if models else None
    except:
        return False, None

def semantic_search(query, top_k=3):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results['documents'][0]

def local_rag_query(query, model_name):
    # Step 1 - Retrieve context
    context_docs = semantic_search(query, top_k=3)
    context = "\n".join([f"- {doc}" for doc in context_docs])

    # Step 2 - Build prompt
    prompt = f"""You are a helpful company assistant.
Answer the question using only the context below.
If the answer is not in the context say I don't have that information.

Context:
{context}

Question: {query}

Answer:"""

    # Step 3 - Generate with Ollama
    start_time = time.time()
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 150,
        }
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=60
    )

    result = response.json()
    end_time = time.time()

    return {
        "query": query,
        "context_used": context_docs,
        "answer": result.get('response', '').strip(),
        "time": round(end_time - start_time, 2)
    }

# Test the assistant
is_running, model_name = check_ollama()

if is_running and model_name:
    print(f"Using model: {model_name} ✅")

    test_queries = [
        "What is the pricing for the Professional plan?",
        "How do I get a refund?",
        "What operating systems are supported?",
        "Is there a free trial available?",
        "How secure is my data?",
    ]

    print("\nTesting Local AI Assistant:")
    print("=" * 50)

    for query in test_queries:
        result = local_rag_query(query, model_name)
        print(f"\nQ: {result['query']}")
        print(f"A: {result['answer'][:200]}")
        print(f"⏱ {result['time']}s")
        print(f"📚 Sources: {len(result['context_used'])} docs")

else:
    print("Ollama not running — start with: ollama serve")
    print("Showing architecture only")

print("\n" + "=" * 50)
print("PART 4 - Conversation History")
print("=" * 50)

def chat_with_history(messages, model_name, max_history=5):
    # Build conversation context
    history_text = ""
    for msg in messages[-max_history:]:
        role = "User" if msg['role'] == 'user' else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful AI assistant with memory of our conversation.

Conversation history:
{history_text}

Continue the conversation naturally.
Assistant:"""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 100}
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=60
        )
        return response.json().get('response', '').strip()
    except:
        return "Error connecting to Ollama"

if is_running and model_name:
    print("Testing conversation memory:")
    conversation = [
        {"role": "user", "content": "My name is Balaji"},
        {"role": "assistant", "content": "Hello Balaji! How can I help you today?"},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is AI that learns from data."},
        {"role": "user", "content": "What was my name again?"},
    ]

    response = chat_with_history(conversation, model_name)
    print(f"Test: 'What was my name again?'")
    print(f"Response: {response[:150]}")

print("\n" + "=" * 50)
print("PART 5 - Performance Benchmarks")
print("=" * 50)

if is_running and model_name:
    print("Running performance benchmarks...")

    short_prompt = "What is AI?"
    long_prompt = "Explain the difference between machine learning, deep learning, and artificial intelligence in detail."

    results = []
    for prompt in [short_prompt, long_prompt]:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 50}
        }
        start = time.time()
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=60
        )
        elapsed = time.time() - start
        data = response.json()
        tokens = data.get('eval_count', 0)
        results.append({
            "prompt_length": len(prompt.split()),
            "tokens_generated": tokens,
            "time": round(elapsed, 2),
            "tokens_per_second": round(tokens/elapsed, 1) if elapsed > 0 else 0
        })

    print(f"\n{'Prompt Words':<15} {'Tokens Out':<12} {'Time':<10} {'Tokens/sec'}")
    print("-" * 50)
    for r in results:
        print(f"{r['prompt_length']:<15} {r['tokens_generated']:<12} {r['time']:<10} {r['tokens_per_second']}")

print("\n" + "=" * 50)
print("PART 6 - Enterprise Value Summary")
print("=" * 50)

print("""
What you built today:
✅ Local RAG system with ChromaDB
✅ Semantic search with embeddings
✅ Conversation memory
✅ Performance benchmarking
✅ Zero cloud dependency

Enterprise pitch:
  "I'll build you a private AI assistant
   that knows everything about your company.
   Runs on your own hardware.
   Your data never leaves your network.
   One time setup. Zero ongoing API costs."

Pricing:
  Setup:   $5,000 - $15,000
  Monthly: $500 - $2,000 maintenance
  ROI:     Saves 20+ hours/week staff time
""")

print("Day 35 Complete ✅")
print("Week 5 almost done — tomorrow final review!")