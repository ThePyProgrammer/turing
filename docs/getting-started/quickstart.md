---
title: Quick Start
description: Go from zero to autonomous ML experiments in five steps.
---

# Quick Start

## The five-step loop

### Step 1: Scaffold a project

```
/turing:init
```

Turing asks three questions:

1. **Project name** -- directory and namespace for this experiment (e.g., `sentiment`)
2. **Primary metric** -- what you are optimizing (e.g., `f1_weighted`, `accuracy`, `rmse`)
3. **Data source** -- path to your dataset (e.g., `data/reviews.csv`)

This creates the full experiment harness: `config.yaml`, `train.py`, `evaluate.py`, `prepare.py`, feature templates, and the hypothesis queue. Everything lands in a self-contained directory.

### Step 2: Run the experiment loop

```
/turing:train
```

The `@ml-researcher` agent takes over. It reads the config, trains a baseline model, evaluates it against the immutable evaluation harness, then iterates -- modifying hyperparameters, trying feature engineering ideas, swapping model architectures. Each experiment is logged to `experiments/log.jsonl`.

A typical first run produces 5-10 experiments before convergence triggers the patience threshold.

### Step 3: Read what happened

```
/turing:brief
```

The `@ml-evaluator` agent (read-only -- it cannot touch your code) generates an intelligence report:

```
Research Brief: sentiment
=========================

Experiments run: 8
Best model: exp-006 (XGBoost, f1_weighted=0.847)
Baseline: exp-001 (XGBoost, f1_weighted=0.721)
Improvement: +17.5% relative

Key findings:
  - TF-IDF n-gram range (1,3) outperformed (1,1) by 9.2%
  - max_depth=6 with learning_rate=0.05 dominated default config
  - Random forest (exp-004) competitive but 2.1% behind best XGBoost

Convergence: patience exhausted after 3 non-improving iterations
Hypothesis queue: 2 untested ideas remaining

Recommended next steps:
  1. Try LightGBM (registered in model_registry.yaml)
  2. Engineer interaction features from top-5 TF-IDF terms
  3. Run /turing:seed to check result stability
```

### Step 4: Inject your taste

The agent is good at systematic search. You are good at insight. Combine them:

```
/turing:try "use character-level n-grams alongside word n-grams for subword signal"
```

This adds your hypothesis to the queue with priority. The next training run picks it up first.

### Step 5: Train again

```
/turing:train
```

The agent tests your hypothesis, compares it against the current best, and keeps whichever wins. Repeat steps 3-5 until you are satisfied.

## Hands-off mode

If you want Turing to keep iterating while you do other work:

```
/loop 5m /turing:train
```

This runs the training loop every 5 minutes. Each invocation picks up where the last left off -- new hypotheses, updated convergence state, the full context.

## What to do next

| Goal | Command |
|------|---------|
| Check current standings | `/turing:status` |
| Compare two experiments | `/turing:compare exp-003 exp-006` |
| Run a hyperparameter sweep | `/turing:sweep` |
| Verify result stability | `/turing:seed` |
| Check reproducibility | `/turing:reproduce exp-006` |
| Get model suggestions from literature | `/turing:suggest` |
