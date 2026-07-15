# Joint Phase-1 Extended Balanced Training

## Overview

This training stage teaches the model **three tasks at once**, using a single forward pass per sample. Each task has its own loss function, and gradients from all three are summed before each weight update.

---

## The Three Tasks

### 1. Time Series Forecasting (Chronos)
- **Data**: ~525k windows drawn from 8 real-world time-series corpora (electricity demand, weather, exchange rates, Wikipedia traffic, etc.)
- **Objective**: Given a history window of patches, predict the next patch
- **Loss**: MSE — the model is penalized for the numerical distance between its prediction and the true next patch

### 2. Time Series QA (TSQA)
- **Data**: ~38k multiple-choice questions about time series (e.g. "what is the trend?", "which period has the highest value?")
- **Objective**: Select the correct answer from a set of options
- **Loss**: Cross-entropy — standard next-token prediction on the answer text

### 3. Time Series Captioning (M4)
- **Data**: ~80k samples from the M4 forecasting competition, each paired with a textual description of the series
- **Objective**: Generate a natural language caption describing the time series
- **Loss**: Cross-entropy — autoregressive token generation

---

## The Problem This Stage Solves

A naive joint training approach assigns batch proportions proportional to dataset size. Since Chronos is ~6× larger than TSQA and ~7× larger than M4, roughly **80% of every epoch** is MSE batches. This dilutes the language signal so much that **TSQA accuracy dropped from 96% → 93%** when Chronos was added at its natural proportion.

---

## The Solution: Task-Balanced Sampling

Instead of sampling proportional to dataset size, a custom `TaskWeightedDistributedBatchSampler` enforces **equal batch counts across all three tasks (1:1:1)**:

```
Each epoch:
  ├── 1/3 of batches → Chronos (MSE)   ← undersampled
  ├── 1/3 of batches → TSQA (CE)       ← oversampled
  └── 1/3 of batches → M4 (CE)         ← oversampled
```

This means the model sees the same amount of language supervision as forecasting supervision per epoch, preventing the MSE signal from dominating.

---

## Architecture Flow (per sample)

```
Input: time series patches + text prompt
        │
        ▼
   Patch Encoder  (MLP or Transformer)
        │
        ▼
  Projector → LLM (Gemma-3-270M)
        │
   ┌────┴────┐
   │         │
  MSE       CE
 (Chronos) (TSQA / M4)
```

The encoder and LLM are **shared across all three tasks**. Only the loss head differs. This forces the shared representations to be useful for both continuous forecasting and discrete language tasks simultaneously.

---

## Why This Matters

| Property | Without balancing | With balancing |
|---|---|---|
| TSQA accuracy | 93% | ~96% |
| CE signal proportion | ~20% | ~67% |
| Chronos coverage/epoch | 100% | ~33% (rest via oversampling of language) |

The tradeoff is that Chronos sees fewer unique windows per epoch, but the model retains strong language grounding — which is the entire point of building on a pre-trained LLM backbone.
