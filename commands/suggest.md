---
name: suggest
description: Literature-grounded model selection. Reads the ML task context, searches recent literature, and suggests model architectures worth trying — with citations. Suggestions are auto-queued as hypotheses.
disable-model-invocation: true
argument-hint: "[task description override]"
allowed-tools: Read, Write, Bash(python scripts/*:*, source .venv/bin/activate:*), Grep, Glob, WebSearch, WebFetch
---

Suggest model architectures for the current ML task, grounded in recent literature. Hypotheses backed by papers, not vibes.

## Steps

### 1. Understand the Task

Read the project config and recent experiment history to understand the task:

```bash
cat config.yaml
```

```bash
source .venv/bin/activate && python scripts/show_metrics.py --last 10 2>/dev/null || echo "No experiments yet"
```

If `$ARGUMENTS` is provided, use that as the task description. Otherwise, infer from `config.yaml` (model type, primary metric, data source, target column).

From the config and any task description, identify the key task properties:
- Data type (tabular, time series, image, text, etc.)
- Objective (classification, regression, generation, etc.)
- Special constraints (imbalanced classes, small dataset, real-time, interpretability, etc.)
- Current model family and what's been tried

### 2. Search Literature

Use `WebSearch` to find recent papers and benchmark results. Run 3-5 searches targeting:

1. **Model comparison for this task type:** e.g., "best models for tabular classification benchmark 2024"
2. **Current model alternatives:** e.g., "LightGBM vs XGBoost vs CatBoost tabular data"
3. **Task-specific techniques:** e.g., "handling class imbalance gradient boosting"

For each search, use `WebFetch` on the top 1-2 results to extract specific model recommendations, benchmark numbers, and methodology.

Focus on:
- Recent work (2023-2026) with empirical comparisons
- Benchmark studies and surveys
- arXiv papers or reputable ML blogs with concrete results

### 3. Synthesize Suggestions

From the literature, synthesize **3-5 concrete model architecture suggestions**. Each must include:

- **Model architecture:** specific (e.g., "LightGBM with GOSS sampling", not "try a different model")
- **Why:** one-sentence rationale grounded in what the literature says
- **Citation:** paper or source that supports this
- **Expected impact:** high/medium/low based on how well it fits this task
- **Implementation hint:** what to change in `train.py` (one concrete line)

### 4. Queue as Hypotheses

For each suggestion, add to the hypothesis queue:

```bash
source .venv/bin/activate && python scripts/manage_hypotheses.py add "<model>: <rationale> (source: <citation>)" --priority medium --source literature
```

### 5. Show Results

```
Literature-Grounded Model Suggestions
======================================

Task: <task description>
Current: <current model> (<current metric>=<value>)
Sources consulted: <N papers/articles>

1. [HIGH] <technique>
   Why: <one-sentence rationale with citation>
   Source: <URL>
   Change: <specific train.py change>
   → Queued as hyp-NNN

2. [MEDIUM] ...

Queued N hypotheses. Run /turing:train to test them.
```

## Fallback

If web search returns insufficient results, suggest model families from `config/taxonomy.toml` based on what hasn't been tried yet. Note that suggestions are taxonomy-based, not literature-backed, and queue with `--source taxonomy`.

## Integration

- Suggestions feed into `hypotheses.yaml` — the next `/turing:train` picks them up
- `/turing:brief` shows queued literature-sourced hypotheses
- Human can override priority: `/turing:try` always takes precedence
