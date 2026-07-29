import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import time
from datetime import datetime

print("Loading model...")
MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)
model.eval()
print("Model loaded ✅")

print("=" * 50)
print("PART 1 - What is RAG")
print("=" * 50)

rag_explanation = {
    "Problem": "LLMs have knowledge cutoff — don't know recent events",
    "Solution": "RAG = Retrieval Augmented Generation",
    "How": "Search relevant documents → add to prompt → model answers",
    "Example": "Ask about your company → search company docs → answer",
    "Benefit": "Model answers from YOUR data not just training data"
}

for key, value in rag_explanation.items():
    print(f"  {key:10} → {value}")

print("\n" + "=" * 50)
print("PART 2 - Simple RAG Implementation")
print("=" * 50)

# Simple document store — in production use vector DB
documents = [
    {
        "id": 1,
        "title": "What is AI",
        "content": "Artificial intelligence is the simulation of human intelligence by machines. It includes machine learning, deep learning and natural language processing."
    },
    {
        "id": 2,
        "title": "Machine Learning",
        "content": "Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed. It uses algorithms to find patterns."
    },
    {
        "id": 3,
        "title": "Deep Learning",
        "content": "Deep learning uses neural networks with many layers to learn complex patterns from large amounts of data. It powers image recognition and language models."
    },
    {
        "id": 4,
        "title": "Fine Tuning",
        "content": "Fine tuning adapts a pretrained model to specific tasks using smaller datasets. LoRA and QLoRA are efficient fine tuning methods."
    },
    {
        "id": 5,
        "title": "Transformers",
        "content": "Transformers use attention mechanisms to process sequences. They are the foundation of modern language models like GPT and BERT."
    },
]

def simple_retrieval(query, top_k=2):
    query_words = set(query.lower().split())
    scores = []
    for doc in documents:
        doc_words = set(doc['content'].lower().split())
        overlap = len(query_words.intersection(doc_words))
        scores.append((overlap, doc))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scores[:top_k]]

# Test retrieval
test_query = "What is machine learning?"
retrieved = simple_retrieval(test_query)
print(f"Query: {test_query}")
print(f"Retrieved {len(retrieved)} relevant documents:")
for doc in retrieved:
    print(f"  → {doc['title']}: {doc['content'][:80]}...")

print("\n" + "=" * 50)
print("PART 3 - RAG Pipeline")
print("=" * 50)

def rag_generate(query, max_tokens=100):
    # Step 1 - Retrieve relevant documents
    retrieved_docs = simple_retrieval(query, top_k=2)

    # Step 2 - Build context from retrieved docs
    context = "\n".join([
        f"Document {i+1}: {doc['content']}"
        for i, doc in enumerate(retrieved_docs)
    ])

    # Step 3 - Build RAG prompt
    prompt = f"""### Context:
{context}

### Instruction:
Answer the question using the context above.

### Question:
{query}

### Answer:"""

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

    return {
        "query": query,
        "retrieved_docs": [d['title'] for d in retrieved_docs],
        "answer": answer
    }

# Test RAG
print("Testing RAG pipeline:")
result = rag_generate("What is deep learning?")
print(f"Query:     {result['query']}")
print(f"Retrieved: {result['retrieved_docs']}")
print(f"Answer:    {result['answer'][:200]}")

print("\n" + "=" * 50)
print("PART 4 - Chat UI With History")
print("=" * 50)

# Chat history store
chat_history = []

def chat(message, history, use_rag):
    start_time = time.time()

    if use_rag:
        result = rag_generate(message, max_tokens=80)
        response = f"[RAG: used docs: {result['retrieved_docs']}]\n{result['answer']}"
    else:
        prompt = f"### Instruction:\n{message}\n\n### Response:"
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3
            )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()

    end_time = time.time()
    response += f"\n\n⏱️ {end_time-start_time:.1f}s"

    # Save to history
    chat_history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "message": message,
        "response": response,
        "used_rag": use_rag
    })

    return response

# Build Chat UI
with gr.Blocks(title="SLM Chat") as demo:
    gr.Markdown("""
    # 🤖 SLM Chat Interface
    ### Your personal AI — built from scratch!
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=chat,
                additional_inputs=[
                    gr.Checkbox(
                        label="Use RAG (search documents)",
                        value=False
                    )
                ],
                examples=[
                    ["What is artificial intelligence?", False],
                    ["What is deep learning?", True],
                    ["Explain machine learning", True],
                    ["What are transformers?", True],
                ],
                title="Chat with your SLM"
            )

        with gr.Column(scale=1):
            gr.Markdown("### 📚 Knowledge Base")
            for doc in documents:
                gr.Markdown(f"**{doc['title']}**\n{doc['content'][:80]}...")

if __name__ == "__main__":
    print("\nStarting Chat UI...")
    print("Open: http://localhost:7860")
    demo.launch(share=True)