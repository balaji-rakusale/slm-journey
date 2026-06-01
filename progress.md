# Progress Log

## Week 1 Summary
- Built autograd engine from scratch
- Built tokenizer from scratch
- Built self attention from scratch
- Trained first language model — 39.7K parameters
- Loss went from 3.57 to 0.52
- Model generated Shakespeare style text

## Key Learnings
- Learning rate 1e-3 is sweet spot for small models
- Model size must match data size (Chinchilla law)
- Tokenization is the bridge between text and numbers
- Attention lets words understand context of other words

## Week 2 Goal
Master data engineering pipeline —
collection, cleaning, deduplication, quality filtering

## Week 2 Summary — Data Engineering 
- Built web scraper from scratch
- Built full data cleaning pipeline
- Built instruction dataset in Alpaca format
- Implemented PII removal for enterprise safety
- Mastered HuggingFace datasets operations
- Combined 3 sources into one pipeline
- 321 samples ready for fine tuning

## Week 3 Goal
Fine tune a real LLM using LoRA and QLoRA