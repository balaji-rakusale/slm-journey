import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import time

print("=" * 50)
print("PART 1 - Why Basic RAG Fails")
print("=" * 50)

print("""
Basic RAG Problems:

Problem 1 — Semantic search finds wrong chunks
  Query:  "What is the refund policy?"
  Finds:  "Policy document header — Page 1"
  Misses: "Refunds allowed within 30 days"

Problem 2 — Retrieved chunks lack context
  Chunk:  "It applies to all purchases made after Jan 2024"
  Missing: What is "it"? Need surrounding context.

Problem 3 — Top K is too rigid
  Always retrieves exactly 3 chunks
  Sometimes 1 is enough, sometimes 10 needed

Solutions:
  → Reranking (fix problem 1)
  → Parent document retrieval (fix problem 2)
  → Dynamic K selection (fix problem 3)
""")

print("=" * 50)
print("PART 2 - Load Models")
print("=" * 50)

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded ✅")

print("Loading reranker model...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Reranker model loaded ✅")

print("\n" + "=" * 50)
print("PART 3 - Build Knowledge Base")
print("=" * 50)

# Detailed knowledge base with parent-child chunks
documents = [
    # Parent document 1
    "Our refund policy is designed to ensure customer satisfaction. All purchases are eligible for a full refund within 30 days of the purchase date. After 30 days, partial refunds may be considered on a case by case basis. Digital products are non-refundable once downloaded.",
    # Parent document 2
    "Technical support is available to all customers regardless of their subscription tier. Basic support includes email assistance with 48 hour response time. Professional tier customers receive priority support with 4 hour response time. Enterprise customers have access to 24/7 phone support.",
    # Parent document 3
    "Our pricing structure offers three tiers to accommodate different needs. The Starter plan at $99 per month includes up to 5 users and 10GB storage. The Professional plan at $299 per month supports up to 25 users and 100GB storage. Enterprise pricing is custom based on requirements.",
    # Parent document 4
    "Data security is our top priority. All customer data is encrypted using AES-256 encryption at rest. Data in transit is protected using TLS 1.3 protocol. We conduct annual SOC2 Type II audits. Customer data is never shared with third parties without explicit consent.",
    # Parent document 5
    "The onboarding process is designed to get you productive quickly. After signup you receive a welcome email with setup instructions. A dedicated onboarding specialist contacts you within 24 hours. The typical onboarding takes 3 to 5 business days to complete.",
]

# Create smaller chunks from documents
def create_chunks(documents, chunk_size=100):
    chunks = []
    chunk_to_parent = {}
    chunk_id = 0

    for parent_id, doc in enumerate(documents):
        words = doc.split()
        for i in range(0, len(words), chunk_size//2):
            chunk = ' '.join(words[i:i+chunk_size])
            if len(chunk) > 20:
                chunks.append(chunk)
                chunk_to_parent[chunk_id] = parent_id
                chunk_id += 1

    return chunks, chunk_to_parent

chunks, chunk_to_parent = create_chunks(documents)
print(f"Documents: {len(documents)}")
print(f"Chunks created: {len(chunks)}")

# Build vector DB
client = chromadb.Client()
collection = client.create_collection("advanced_kb")

embeddings = embedder.encode(chunks).tolist()
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print("Vector DB built ✅")

print("\n" + "=" * 50)
print("PART 4 - Basic vs Advanced RAG")
print("=" * 50)

def basic_rag(query, top_k=3):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results['documents'][0]

def advanced_rag_with_reranking(query, initial_k=10, final_k=3):
    # Step 1 - Retrieve more candidates
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(initial_k, len(chunks))
    )
    candidates = results['documents'][0]

    # Step 2 - Rerank using cross encoder
    pairs = [[query, doc] for doc in candidates]
    rerank_scores = reranker.predict(pairs)

    # Step 3 - Sort by rerank score
    ranked = sorted(
        zip(rerank_scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )

    # Step 4 - Return top final_k
    return [doc for _, doc in ranked[:final_k]]

def parent_document_retrieval(query, top_k=2):
    # Retrieve child chunks
    child_chunks = basic_rag(query, top_k=top_k)

    # Find parent documents
    parent_ids = set()
    for chunk in child_chunks:
        for chunk_id, parent_id in chunk_to_parent.items():
            if chunks[chunk_id] in chunk or chunk in chunks[chunk_id]:
                parent_ids.add(parent_id)

    # Return full parent documents
    return [documents[pid] for pid in parent_ids]

# Test all three approaches
test_queries = [
    "Can I get my money back?",
    "How fast will support respond?",
    "Is my data safe?",
]

print("Comparing RAG approaches:\n")
for query in test_queries:
    print(f"Query: {query}")

    basic = basic_rag(query, top_k=2)
    advanced = advanced_rag_with_reranking(query, initial_k=8, final_k=2)
    parent = parent_document_retrieval(query, top_k=2)

    print(f"Basic RAG:    {basic[0][:80]}...")
    print(f"Reranked:     {advanced[0][:80]}...")
    print(f"Parent doc:   {parent[0][:80] if parent else 'None'}...")
    print()

print("\n" + "=" * 50)
print("PART 5 - Reranking Score Analysis")
print("=" * 50)

query = "What is the refund policy?"
candidates = basic_rag(query, top_k=5)
pairs = [[query, doc] for doc in candidates]
scores = reranker.predict(pairs)

print(f"Query: {query}")
print(f"\nReranking scores:")
ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
for score, doc in ranked:
    relevance = "✅ Relevant" if score > 0 else "❌ Not relevant"
    print(f"  Score {score:6.2f} {relevance}: {doc[:60]}...")

print("\n" + "=" * 50)
print("PART 6 - Speed Comparison")
print("=" * 50)

query = "What are the pricing plans?"
runs = 3

start = time.time()
for _ in range(runs):
    basic_rag(query, top_k=3)
basic_time = (time.time() - start) / runs

start = time.time()
for _ in range(runs):
    advanced_rag_with_reranking(query, initial_k=8, final_k=3)
advanced_time = (time.time() - start) / runs

print(f"Basic RAG time:    {basic_time*1000:.0f} ms")
print(f"Advanced RAG time: {advanced_time*1000:.0f} ms")
print(f"Overhead:          {advanced_time/basic_time:.1f}x slower")
print(f"Quality gain:      Significantly better relevance")
print(f"Verdict:           Worth the overhead for enterprise!")

print("\nDay 37 Complete ✅")
print("Tomorrow: Fine tuning evaluation metrics")