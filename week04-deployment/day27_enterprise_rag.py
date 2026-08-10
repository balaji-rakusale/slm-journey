import os
import json
import time
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 50)
print("PART 1 - Enterprise RAG Architecture")
print("=" * 50)

print("""
Enterprise RAG Pipeline:

Client Documents (PDFs, Word, Text)
         ↓
Document Loader
         ↓
Text Chunker (split into 500 char chunks)
         ↓
Embedding Model (text → vectors)
         ↓
Vector Database (ChromaDB)
         ↓
User Query → Semantic Search → Top K Chunks
         ↓
RAG Prompt Builder
         ↓
Language Model
         ↓
Final Answer + Sources
""")

print("=" * 50)
print("PART 2 - Document Processing Pipeline")
print("=" * 50)

# Simulate enterprise documents
enterprise_documents = {
    "company_policy.txt": """
    Our company was founded in 2010 and operates in 15 countries.
    Employee leave policy allows 25 days annual leave per year.
    Remote work policy allows 3 days work from home per week.
    All employees must complete security training annually.
    Expense claims must be submitted within 30 days of purchase.
    Travel expenses require manager approval above $500.
    """,
    "product_manual.txt": """
    Product X Version 2.0 User Manual.
    Installation requires Windows 10 or higher with 8GB RAM.
    To install run setup.exe and follow on screen instructions.
    Default username is admin and password must be changed on first login.
    Support is available Monday to Friday 9am to 5pm EST.
    For technical issues contact support@company.com.
    """,
    "faq.txt": """
    Q: How do I reset my password?
    A: Go to login page and click forgot password link.

    Q: What payment methods are accepted?
    A: We accept Visa Mastercard and PayPal.

    Q: How long does shipping take?
    A: Standard shipping takes 5-7 business days.

    Q: Can I return a product?
    A: Yes products can be returned within 30 days of purchase.

    Q: Is there a free trial?
    A: Yes we offer a 14 day free trial with no credit card required.
    """
}

def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks

# Process all documents
all_chunks = []
chunk_metadata = []

for filename, content in enterprise_documents.items():
    chunks = chunk_text(content)
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        chunk_metadata.append({
            "source": filename,
            "chunk_id": i
        })

print(f"Documents processed: {len(enterprise_documents)}")
print(f"Total chunks created: {len(all_chunks)}")
print(f"\nSample chunk:")
print(all_chunks[0])

print("\n" + "=" * 50)
print("PART 3 - Build Enterprise Vector DB")
print("=" * 50)

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded ✅")

# Create ChromaDB
client = chromadb.Client()
collection = client.create_collection("enterprise_kb")

# Embed and store all chunks
print("Embedding documents...")
embeddings = embedder.encode(all_chunks).tolist()

collection.add(
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=chunk_metadata,
    ids=[f"chunk_{i}" for i in range(len(all_chunks))]
)

print(f"Vector DB built with {len(all_chunks)} chunks ✅")

print("\n" + "=" * 50)
print("PART 4 - Enterprise Query Engine")
print("=" * 50)

print("Loading language model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()
print("Language model loaded ✅")

def enterprise_rag_query(query, top_k=3):
    start_time = time.time()

    # Step 1 - Semantic search
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    retrieved_chunks = results['documents'][0]
    retrieved_sources = [m['source'] for m in results['metadatas'][0]]

    # Step 2 - Build enterprise prompt
    context = "\n\n".join([
        f"[Source: {src}]\n{chunk}"
        for chunk, src in zip(retrieved_chunks, retrieved_sources)
    ])

    prompt = f"""You are an enterprise AI assistant.
Answer the question using only the provided context.
If the answer is not in the context say I don't know.

Context:
{context}

Question: {query}

Answer:"""

    # Step 3 - Generate answer
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_text[len(prompt):].strip()
    end_time = time.time()

    return {
        "query": query,
        "answer": answer,
        "sources": list(set(retrieved_sources)),
        "chunks_retrieved": len(retrieved_chunks),
        "time_seconds": round(end_time - start_time, 2)
    }

print("\n" + "=" * 50)
print("PART 5 - Test Enterprise Queries")
print("=" * 50)

enterprise_queries = [
    "How many days annual leave do employees get?",
    "What are the system requirements for Product X?",
    "How do I reset my password?",
    "What is the return policy?",
    "Can I work from home?",
]

print("Testing enterprise queries:\n")
for query in enterprise_queries:
    result = enterprise_rag_query(query)
    print(f"Q: {result['query']}")
    print(f"A: {result['answer'][:150]}")
    print(f"Sources: {result['sources']}")
    print(f"Time: {result['time_seconds']}s")
    print()

print("\n" + "=" * 50)
print("PART 6 - Business Value Calculator")
print("=" * 50)

print("""
Enterprise RAG Product Pricing:

What you built today:
  ✅ Document ingestion pipeline
  ✅ Automatic chunking
  ✅ Vector embeddings
  ✅ Semantic search
  ✅ Source attribution
  ✅ Enterprise prompt engineering

What clients pay for:

Tier 1 - Small Business (< 100 docs)
  Setup:    $3,000 - $5,000
  Monthly:  $500 - $1,000
  Example:  Law firm FAQ bot

Tier 2 - Medium Business (100-1000 docs)
  Setup:    $10,000 - $25,000
  Monthly:  $2,000 - $5,000
  Example:  HR policy assistant

Tier 3 - Enterprise (1000+ docs)
  Setup:    $50,000 - $200,000
  Monthly:  $10,000 - $50,000
  Example:  Bank knowledge base

Your stack cost:
  ChromaDB:            Free (open source)
  Sentence transformers: Free (open source)
  GPU rental:          $50 - $200/month
  Your margin:         90%+
""")

print("\n" + "=" * 50)
print("PART 7 - Save Pipeline Config")
print("=" * 50)

pipeline_config = {
    "embedding_model": "all-MiniLM-L6-v2",
    "language_model": "gpt2",
    "chunk_size": 200,
    "chunk_overlap": 50,
    "top_k_retrieval": 3,
    "temperature": 0.3,
    "max_new_tokens": 80,
    "vector_db": "chromadb",
    "documents_processed": len(enterprise_documents),
    "chunks_stored": len(all_chunks)
}

with open('enterprise_rag_config.json', 'w') as f:
    json.dump(pipeline_config, f, indent=2)

print("Pipeline config saved ✅")
print(json.dumps(pipeline_config, indent=2))