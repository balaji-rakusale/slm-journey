import torch
import torch.nn.functional as F



# SELF ATTENTION FROM SCRATCH
# This is the heart of every LLM including GPT4

# Step 1 - Create a simple sentence as vectors
# Imagine 4 words, each represented as 8 numbers
torch.manual_seed(42)
sentence_length = 4  # 4 words
embedding_dim = 8    # each word = 8 numbers

# Random word embeddings (in real model these are learned)
x = torch.randn(sentence_length, embedding_dim)
print("Input shape:", x.shape)
print("Our sentence as vectors:")
print(x)

# Step 2 - Create Q, K, V weight matrices
# These are learned during training
W_q = torch.randn(embedding_dim, embedding_dim)
W_k = torch.randn(embedding_dim, embedding_dim)
W_v = torch.randn(embedding_dim, embedding_dim)

# Step 3 - Calculate Q, K, V
Q = x @ W_q  # Query - what am I looking for?
K = x @ W_k  # Key   - what do I contain?
V = x @ W_v  # Value - what do I actually give?


print("\nQ shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

# Step 4 - Calculate attention scores
scores = Q @ K.T  # how much should each word attend to every other word
print("\nRaw attention scores:")
print(scores)

# Step 5 - Scale the scores
# Without scaling gradients explode
scaled_scores = scores / (embedding_dim ** 0.5)
print("\nScaled scores:")
print(scaled_scores)

# Step 6 - Softmax to get probabilities
attention_weights = F.softmax(scaled_scores, dim=-1)
print("\nAttention weights (each row sums to 1):")
print(attention_weights)
print("\nRow sums (should all be 1.0):")
print(attention_weights.sum(dim=-1))

# Step 7 - Final output
output = attention_weights @ V
print("\nFinal attention output shape:", output.shape)
print("Output:")
print(output)