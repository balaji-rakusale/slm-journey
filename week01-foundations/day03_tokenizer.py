# Understanding tokenization from scratch
# Step 1 - Simple word tokenizer
sentence = "I am building my own Small Language Model"

# Split by space - most basic tokenizer
word_tokens = sentence.split()
print("Word Tokens:", word_tokens)
print(f"Total Tokens: {len(word_tokens)}")

# Step 2 - Character tokenizer
char_tokens=list(sentence)
print("\nChar Tokens:")
print(char_tokens)
print(f"Total Tokens: {len(char_tokens)}")

# Step 3 - Build a vocabulary
vocab=sorted(set(char_tokens))
print("\nVocabulary:")
print(vocab)
print(f"Vocab size: {len(vocab)}")

# Step 4 - Encode (text to numbers)
char_to_int = {}
for i, ch in enumerate(vocab):
    char_to_int[ch] = i

print("\nChar to int mapping:")
print(char_to_int)

# NOW encode the sentence using the fixed mapping
encoded = [char_to_int[ch] for ch in sentence]
print("\nEncoded sentence:")
print(encoded)

# Step 5 - Decode (numbers back to text)
int_to_char = {i: ch for ch, i in char_to_int.items()}
decoded = ''.join([int_to_char[i] for i in encoded])
print("\nDecoded sentence:")
print(decoded)
# Step 6 - Now use HuggingFace tokenizer and compare
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokens = tokenizer.encode(sentence)
print("\nGPT2 tokens:")
print(gpt2_tokens)
print(f"Total GPT2 tokens: {len(gpt2_tokens)}")
print("\nGPT2 decoded:")
print(tokenizer.decode(gpt2_tokens))