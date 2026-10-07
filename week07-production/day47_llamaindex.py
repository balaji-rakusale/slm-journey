import json
import time
from datetime import datetime

print("=" * 50)
print("PART 1 - What is LlamaIndex")
print("=" * 50)

print("""
What we built manually (Days 25-27):
  → Load documents manually
  → Chunk text manually
  → Embed with sentence transformers
  → Store in ChromaDB manually
  → Query manually
  → Build prompt manually
  → Call LLM manually
  Total: 200+ lines of code

LlamaIndex does all of this:
  → 10-20 lines of code
  → Production grade pipeline
  → Many connectors built in
  → Handles edge cases automatically

LlamaIndex Components:
  Documents  → load from PDF, Word, web, DB
  Nodes      → chunked pieces of documents
  Index      → stores and retrieves nodes
  Query Engine → handles user questions
  LLM        → generates final answer
""")

print("=" * 50)
print("PART 2 - Manual RAG vs LlamaIndex")
print("=" * 50)

manual_rag_code = """
# Manual RAG (what we built - ~200 lines)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()
collection = client.create_collection("kb")
chunks = chunk_text(documents)
embeddings = embedder.encode(chunks).tolist()
collection.add(documents=chunks, embeddings=embeddings, ids=[...])

def query(question):
    q_emb = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    context = "\\n".join(results['documents'][0])
    prompt = f"Context: {context}\\nQuestion: {question}\\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0])
"""

llamaindex_code = """
# LlamaIndex RAG (~10 lines)
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is the refund policy?")
print(response)
"""

print("Manual RAG code:")
print(manual_rag_code)
print("\nLlamaIndex code:")
print(llamaindex_code)

print("=" * 50)
print("PART 3 - Build LlamaIndex Pipeline")
print("=" * 50)

# Check if LlamaIndex is installed
try:
    from llama_index.core import (
        VectorStoreIndex,
        Document,
        Settings,
        StorageContext,
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    llamaindex_available = True
    print("LlamaIndex installed ✅")
except ImportError:
    llamaindex_available = False
    print("LlamaIndex not installed")
    print("Run: pip install llama-index llama-index-embeddings-huggingface")

if llamaindex_available:
    # Configure embedding model
    print("Setting up embedding model...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    Settings.chunk_size = 256
    Settings.chunk_overlap = 50
    print("Embedding model configured ✅")

    # Create documents
    documents = [
        Document(
            text="""Our refund policy allows full refunds within 30 days of purchase.
            After 30 days partial refunds may be considered case by case.
            Digital products are non-refundable once downloaded.""",
            metadata={"source": "refund_policy", "domain": "policy"}
        ),
        Document(
            text="""Technical support is available Monday to Friday 9am to 6pm IST.
            Email support at support@company.com for all queries.
            Enterprise customers get 24/7 dedicated support.""",
            metadata={"source": "support_policy", "domain": "support"}
        ),
        Document(
            text="""Pricing plans: Starter at $99/month for 5 users.
            Professional at $299/month for 25 users.
            Enterprise pricing is custom based on requirements.""",
            metadata={"source": "pricing", "domain": "pricing"}
        ),
        Document(
            text="""Machine learning is a subset of AI that enables computers
            to learn from data without explicit programming using algorithms.""",
            metadata={"source": "ml_basics", "domain": "education"}
        ),
        Document(
            text="""Deep learning uses neural networks with multiple layers
            to learn complex patterns from large amounts of training data.""",
            metadata={"source": "dl_basics", "domain": "education"}
        ),
    ]

    print(f"\nDocuments created: {len(documents)}")

    # Build index
    print("Building vector index...")
    start = time.time()
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=False
    )
    index_time = time.time() - start
    print(f"Index built in {index_time:.2f}s ✅")

    print("\n" + "=" * 50)
    print("PART 4 - Query the Index")
    print("=" * 50)

    # Use simple retriever without LLM
    retriever = index.as_retriever(similarity_top_k=3)

    test_queries = [
        "What is the refund policy?",
        "How do I contact support?",
        "What are the pricing plans?",
        "What is machine learning?",
    ]

    print("Testing retrieval:")
    for query in test_queries:
        start = time.time()
        nodes = retriever.retrieve(query)
        elapsed = time.time() - start

        print(f"\nQuery: {query}")
        print(f"Retrieved {len(nodes)} nodes in {elapsed:.3f}s")
        for i, node in enumerate(nodes[:2]):
            score = node.score if node.score else 0
            print(f"  Node {i+1} (score: {score:.3f}): {node.text[:80]}...")

    print("\n" + "=" * 50)
    print("PART 5 - Metadata Filtering")
    print("=" * 50)

    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

    # Filter by domain
    filters = MetadataFilters(filters=[
        MetadataFilter(key="domain", value="education")
    ])

    education_retriever = index.as_retriever(
        similarity_top_k=3,
        filters=filters
    )

    print("Filtering by domain='education':")
    edu_nodes = education_retriever.retrieve("explain neural networks")
    for node in edu_nodes:
        print(f"  Source: {node.metadata.get('source', 'unknown')}")
        print(f"  Text: {node.text[:100]}...")

    print("\n" + "=" * 50)
    print("PART 6 - Persist and Load Index")
    print("=" * 50)

    # Save index to disk
    index.storage_context.persist(persist_dir="./llamaindex_storage")
    print("Index persisted to disk ✅")

    # Load index from disk
    from llama_index.core import load_index_from_storage

    loaded_storage = StorageContext.from_defaults(
        persist_dir="./llamaindex_storage"
    )
    loaded_index = load_index_from_storage(loaded_storage)
    print("Index loaded from disk ✅")

    # Verify loaded index works
    loaded_retriever = loaded_index.as_retriever(similarity_top_k=2)
    test_nodes = loaded_retriever.retrieve("refund policy")
    print(f"Loaded index retrieved {len(test_nodes)} nodes ✅")

else:
    print("\nRunning architecture demo without LlamaIndex...")

print("\n" + "=" * 50)
print("PART 7 - LlamaIndex vs Manual RAG")
print("=" * 50)

comparison = {
    "Feature": ["Lines of code", "Document loaders", "Index types",
                "Persistence", "Metadata filtering", "Production ready"],
    "Manual RAG": ["200+", "Manual only", "ChromaDB only",
                   "Manual", "Manual", "Needs work"],
    "LlamaIndex": ["10-20", "PDF Word Web DB API", "Vector Graph KG",
                   "Built in", "Built in", "Yes out of box"],
}

print(f"\n{'Feature':<25} {'Manual RAG':<25} {'LlamaIndex'}")
print("-" * 70)
for i in range(len(comparison['Feature'])):
    feat = comparison['Feature'][i]
    manual = comparison['Manual RAG'][i]
    llama = comparison['LlamaIndex'][i]
    print(f"{feat:<25} {manual:<25} {llama}")

summary = {
    "date": datetime.now().strftime("%Y-%m-%d"),
    "tool": "LlamaIndex",
    "documents_indexed": 5,
    "index_time": round(index_time, 2) if llamaindex_available else 0,
    "queries_tested": 4,
}

with open('llamaindex_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nDay 47 Complete ✅")
print("Tomorrow: Streaming responses + async APIs!")