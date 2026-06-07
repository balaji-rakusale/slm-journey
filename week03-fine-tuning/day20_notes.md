# Day 20 — QLoRA GPU Fine Tuning on Kaggle

## Setup
- Platform: Kaggle with Tesla T4 GPU
- Model: microsoft/phi-2 (1.5B parameters)
- Quantization: 4bit QLoRA

## Results
- Training time: 131 seconds
- Steps: 60
- Loss: 3.021 → 2.052

## Key Learning
- GPU is 18x faster than CPU
- 1.5B model fits in 0.59GB with 4bit quantization
- Same pipeline works on any model size