---
name: suggest
description: Literature-grounded model selection. Classifies the ML task, searches recent literature via web search, and suggests model architectures worth trying — with citations. Suggestions are auto-queued as hypotheses.
disable-model-invocation: true
argument-hint: "[task description override]"
allowed-tools: Read, Write, Bash(python scripts/*:*, source .venv/bin/activate:*), Grep, Glob, WebSearch, WebFetch
---

Suggest model architectures for the current ML task, grounded in recent literature. This is the evidence-backed complement to the agent's own reasoning — hypotheses backed by papers, not vibes.

## Steps

### 1. Classify the Task

Run the task classifier to detect which ML task properties apply and generate search queries:

```bash
source .venv/bin/activate && python scripts/classify_task.py --config config.yaml --format json
```

If `$ARGUMENTS` is provided, use it as the task description override:
```bash
source .venv/bin/activate && python scripts/classify_task.py --description "$ARGUMENTS" --format json
```

### 2. Search Literature

Use `WebSearch` to find recent papers and articles for each search query from step 1 (take the top 3-5 queries). For each query:

1. Run `WebSearch` with the query
2. Read the top 2-3 results with `WebFetch` to extract model architecture recommendations
3. Note the source URL, paper title/author, and specific model recommendations

Focus on:
- Recent papers (2023-2026) comparing model architectures for this task type
- Benchmark studies and surveys
- arXiv papers with empirical results

### 3. Synthesize Suggestions

From the literature found, synthesize **3-5 concrete model architecture suggestions**. Each suggestion must include:

- **Model architecture:** specific model type (e.g., "LightGBM with leaf-wise growth", not just "try a different model")
- **Why:** one-sentence rationale grounded in what the literature says
- **Citation:** paper/source that supports this suggestion
- **Expected impact:** high/medium/low based on how well the task properties match
- **Implementation hint:** what to change in `train.py` (one line)

### 4. Queue as Hypotheses

For each suggestion, add it to the hypothesis queue:

```bash
source .venv/bin/activate && python scripts/manage_hypotheses.py add "<model>: <rationale> (source: <citation>)" --priority medium --source literature
```

### 5. Show Results

Display the suggestions in a structured format:

```
Literature-Grounded Model Suggestions
======================================

Task: <detected task description>
Categories: <detected categories>
Sources consulted: <N papers/articles>

1. [HIGH] LightGBM with GOSS sampling
   Why: Smith et al. (2024) showed 12% improvement over XGBoost on imbalanced tabular data
   Source: arxiv.org/abs/2024.XXXXX
   Change: Replace XGBClassifier with LGBMClassifier(boosting_type='goss')
   → Queued as hyp-NNN

2. [MEDIUM] CatBoost with ordered boosting
   ...

Queued N hypotheses. Run /turing:train to test them.
```

## When No Literature is Found

If web search returns insufficient results:
1. Fall back to the task taxonomy's search_terms to suggest model families from `config/taxonomy.toml`
2. Note that suggestions are taxonomy-based, not literature-backed
3. Still queue as hypotheses but with `--source taxonomy` instead of `--source literature`

## Integration

- Suggestions feed into the hypothesis queue (`hypotheses.yaml`)
- The next `/turing:train` iteration will prioritize these
- `/turing:brief` will show queued literature-sourced hypotheses
- Human can override priority: `/turing:try` always takes precedence
