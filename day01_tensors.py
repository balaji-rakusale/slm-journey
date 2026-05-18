import torch
import torch.nn.functional as F
# Matrix multiply
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
z = torch.matmul(x, y)
print("Matrix multiply result:")
print(z)

# Softmax by hand
scores = torch.tensor([2.0, 1.0, 0.5])
probs = F.softmax(scores, dim=0)
print("\nSoftmax probabilities:")
print(probs)