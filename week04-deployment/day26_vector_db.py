import json
import time
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

print("=" * 50)
print("PART 1 - Why Vector Databases")
print("=" * 50)

print("""
Keyword Search (Day 25):
  Query:    "what is deep learning"
  Searches: exact word matches
  Misses:   "neural networks" "AI training" "backprop"
  Problem:  Misses semantically similar content

Vector Search (Today):
  Query:    "what is deep learning"
  Converts: query → vector [0.2, 0.8, 0.1, ...]
  Searches: nearest vectors in space
  Finds:    "neural networks" "AI training" "backprop"
  Because:  Similar meaning = similar vectors
""")

print("=" * 50)
print("PART 2 - Load Embedding Model")
print("=" * 50)

# Embedding model converts text to vectors
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded ✅")

# Test embeddings
test_sentences = [
    "What is machine learning?",
    "How do neural networks learn?",
    "What is the weather today?",
]

embeddings = embedder.encode(test_sentences)
print(f"\nEmbedding shape: {embeddings.shape}")
print(f"Each sentence → {embeddings.shape[1]} dimensional vector")

# Show similarity
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sim_matrix = cosine_similarity(embeddings)
print(f"\nSimilarity matrix:")
print(f"{'':30} ML Question  Neural Net  Weather")
for i, sent in enumerate(test_sentences):
    print(f"{sent[:30]:30} {sim_matrix[i][0]:.3f}        {sim_matrix[i][1]:.3f}       {sim_matrix[i][2]:.3f}")

print("\nML and Neural Net are similar (high score)")
print("Weather is unrelated (low score)")

print("\n" + "=" * 50)
print("PART 3 - Build Vector Database")
print("=" * 50)

# Initialize ChromaDB
client = chromadb.Client()

# Create collection
collection = client.create_collection(
    name="slm_knowledge_base",
    metadata={"description": "AI knowledge base for RAG"}
)
print("ChromaDB collection created ✅")

# Knowledge base documents
documents = [
    "Artificial intelligence is the simulation of human intelligence by machines including learning reasoning and problem solving.",
    "Machine learning enables computers to learn from data without being explicitly programmed using statistical algorithms.",
    "Deep learning uses neural networks with many layers to learn complex patterns from large amounts of data.",
    "Natural language processing allows computers to understand and generate human language text.",
    "Fine tuning adapts a pretrained model to specific tasks using smaller domain specific datasets.",
    "LoRA is a parameter efficient fine tuning method that trains small adapter matrices instead of full model weights.",
    "Transformers use attention mechanisms to understand relationships between all words in a sequence simultaneously.",
    "Retrieval augmented generation combines document search with language model generation for accurate answers.",
    "Vector databases store embeddings and enable fast semantic similarity search across millions of documents.",
    "Enterprise AI deployment requires data privacy PII removal quality filtering and domain specific fine tuning.",
]

# Add documents to ChromaDB
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))],
    embeddings=embedder.encode(documents).tolist()
)

print(f"Added {len(documents)} documents to vector DB ✅")

print("\n" + "=" * 50)
print("PART 4 - Semantic Search")
print("=" * 50)

def semantic_search(query, top_k=3):
    # Convert query to vector
    query_embedding = embedder.encode([query]).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return results['documents'][0]

# Test queries
test_queries = [
    "How do I train a language model efficiently?",
    "What makes transformers special?",
    "How to deploy AI for enterprise clients?",
]

print("Semantic search results:")
for query in test_queries:
    print(f"\nQuery: {query}")
    results = semantic_search(query, top_k=2)
    for i, result in enumerate(results):
        print(f"  Result {i+1}: {result[:80]}...")

print("\n" + "=" * 50)
print("PART 5 - Advanced RAG Pipeline")
print("=" * 50)

from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading language model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()
print("Language model loaded ✅")

def advanced_rag(query, top_k=3, max_tokens=100):
    start_time = time.time()

    # Step 1 - Semantic search
    retrieved_docs = semantic_search(query, top_k=top_k)

    # Step 2 - Build context
    context = "\n".join([
        f"[{i+1}] {doc}"
        for i, doc in enumerate(retrieved_docs)
    ])

    # Step 3 - Build prompt
    prompt = f"""Context:
{context}

Question: {query}

Answer based on the context:"""

    # Step 4 - Generate
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=400
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_text[len(prompt):].strip()
    end_time = time.time()

    return {
        "query": query,
        "retrieved_docs": retrieved_docs,
        "answer": answer,
        "time": round(end_time - start_time, 2)
    }

# Test advanced RAG
print("\nTesting Advanced RAG:")
test_questions = [
    "How does LoRA make fine tuning efficient?",
    "What is the role of attention in transformers?",
]

for question in test_questions:
    result = advanced_rag(question)
    print(f"\nQuestion: {result['query']}")
    print(f"Retrieved: {len(result['retrieved_docs'])} docs")
    print(f"Answer:    {result['answer'][:200]}")
    print(f"Time:      {result['time']}s")

print("\n" + "=" * 50)
print("PART 6 - Comparison: Keyword vs Semantic")
print("=" * 50)

# Keyword search from Day 25
def keyword_search(query, docs, top_k=2):
    query_words = set(query.lower().split())
    scores = []
    for doc in docs:
        doc_words = set(doc.lower().split())
        overlap = len(query_words.intersection(doc_words))
        scores.append((overlap, doc))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scores[:top_k]]

comparison_query = "How do models learn patterns from training examples?"
print(f"Query: {comparison_query}\n")

keyword_results = keyword_search(comparison_query, documents, top_k=2)
semantic_results = semantic_search(comparison_query, top_k=2)

print("Keyword Search results:")
for r in keyword_results:
    print(f"  → {r[:80]}...")

print("\nSemantic Search results:")
for r in semantic_results:
    print(f"  → {r[:80]}...")

print("\nSemantic search finds conceptually related docs")
print("even when exact words don't match!")

print("\n" + "=" * 50)
print("DAY 26 COMPLETE")
print("=" * 50)
print("""
What you built today:
✅ Vector embeddings with sentence-transformers
✅ ChromaDB vector database
✅ Semantic search pipeline
✅ Advanced RAG with vector search
✅ Keyword vs semantic comparison

Enterprise Value:
  Client gives you 10,000 documents
  You embed all documents → store in ChromaDB
  User asks question → semantic search → RAG answer
  Model answers from client's own documents
  That's a $20K-$50K product!
""")