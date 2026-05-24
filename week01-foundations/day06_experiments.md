# Day 06 — Hyperparameter Experiments

## Objective
Understand how different hyperparameters affect model training and output quality by running systematic experiments on nanoGPT.

---

## Experiments

| Experiment      | Loss Start | Loss End | Generated Quality                        |
|-----------------|------------|----------|------------------------------------------|
| Original        | 3.57       | 0.52     | Good — clear words and phrases           |
| Higher LR       | 3.73       | 2.00     | Poor — jumbled words, unstable           |
| Lower LR        | 3.61       | 2.52     | Weak — too slow to learn enough          |
| Bigger model    | 3.87       | 1.96     | Decent — needs more data to shine        |
| Smaller model   | 3.77       | 2.45     | Weak — too small to capture patterns     |
| More iterations | 3.88       | 1.49     | Better — more coherent sentences         |

---

## What I Learned

### 1. Higher Learning Rate (1e-2) — WORSE
- Loss stuck at 2.00
- Model takes giant steps and overshoots the optimal point
- Like jumping 10 meters trying to find the bottom of a valley

### 2. Lower Learning Rate (1e-4) — WORSE
- Loss stuck at 2.52
- Model learns but incredibly slowly
- Would need 10,000+ iterations to match original
- Learning rate 1e-3 is the sweet spot for this dataset

### 3. Bigger Model (embedding=64, heads=8, layers=6) — WORSE
- Loss stuck at 1.96
- Bigger model needs more data to shine
- Our dataset is only 168 characters — too small
- Chinchilla law confirmed: model size and data size must scale together

### 4. Smaller Model (embedding=16, heads=2, layers=2) — WORSE
- Loss stuck at 2.45
- Too small to capture language patterns
- Not enough capacity to hold information

### 5. More Iterations (3000 steps) — BETTER
- Loss dropped to 1.49
- More training time always helps up to a point
- After that model overfits — memorizes instead of learning

---

## Key Insights

### The Goldilocks Problem
Every hyperparameter has a sweet spot:
- Not too high, not too low learning rate
- Not too big, not too small model
- Not too few, not too many iterations

### Chinchilla Scaling Law (Confirmed Today)
> Bigger model + less data = BAD
> Smaller model + more data = GOOD
> Matched model + matched data = BEST

### Learning Rate Is King
Most sensitive hyperparameter in deep learning.
Small change = massive difference in results.

---

## Hyperparameter Cheat Sheet

| Hyperparameter | Too High         | Too Low           | Sweet Spot |
|----------------|------------------|-------------------|------------|
| Learning Rate  | Loss explodes    | Learns too slowly | 1e-3       |
| Model Size     | Needs more data  | Underfits         | Match data |
| Iterations     | Overfits         | Underfits         | Monitor loss |
| Batch Size     | Memory issues    | Noisy gradients   | 4-32       |

---

## Resources
- [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556)
- [Karpathy nanoGPT](https://github.com/karpathy/nanoGPT)

---
