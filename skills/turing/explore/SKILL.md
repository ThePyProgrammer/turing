---
name: explore
description: Tree-search-guided hypothesis exploration using AB-MCTS. Explores the space of experiment ideas as a search tree, scored by the critique engine. Discovers non-obvious refinement chains that linear suggestion cannot find.
disable-model-invocation: true
argument-hint: "[ml/project] [--iterations N] [--top N] [--strategy abmcts-a|abmcts-m|greedy]"
allowed-tools: Read, Write, Bash(python scripts/*:*, source .venv/bin/activate:*), Grep, Glob
---

Explore the hypothesis space using tree search. Instead of suggesting independent ideas, this builds and searches a tree of refinement chains — each node is a hypothesis scored by novelty, feasibility, and expected impact.

## Project Detection

0. **Detect project directory:**
   - If `$ARGUMENTS` contains a path (e.g., `ml/coding`), use that as the project directory
   - Else if cwd contains `config.yaml` and `train.py`, use cwd
   - Else search for `ml/*/` subdirectories containing `config.yaml`
     - If exactly one found, use it
     - If multiple found, list them and ask the user which to target
   - All subsequent commands run from the detected project directory

## Parse Options

Extract from `$ARGUMENTS`:
- `--iterations N` — search depth (default: 30)
- `--top N` — number of results to return (default: 5)
- `--strategy` — algorithm choice: `abmcts-a` (default), `abmcts-m` (Bayesian), or `greedy` (no TreeQuest needed)
- `--seeds-only` — just show generated seeds without running search
- `--json` — output as JSON for programmatic use

## Steps

### 1. Assess Current State

```bash
source .venv/bin/activate && python scripts/show_metrics.py --last 10 2>/dev/null || echo "No experiments yet"
```

Read `config.yaml` to understand the current model and metric.

### 2. Run Tree Search

```bash
source .venv/bin/activate && python scripts/treequest_suggest.py \
    --log experiments/log.jsonl \
    --config config.yaml \
    --top <N> \
    --iterations <N> \
    --strategy <strategy>
```

The script will:
- Generate seed hypotheses from config and experiment history
- Run AB-MCTS (or greedy fallback) over the hypothesis tree
- Score each node using the critique engine
- Return top-K ranked, deduplicated hypotheses

### 3. Queue Best Hypotheses

For each result, add to the hypothesis queue:

```bash
source .venv/bin/activate && python scripts/manage_hypotheses.py add "<description>" \
    --priority medium --source treequest
```

### 4. Show Results

Display the search output and confirm queuing:

```
TreeQuest Hypothesis Exploration (AB-MCTS-A)
============================================
Nodes explored: 35
Top 5 hypotheses by critique score:

  1. [PROCEED] (score: 7.8/10)
     Switch to LightGBM with dart boosting; additionally add polynomial features
     Novelty: 8  Feasibility: 9  Impact: 7
     -> Queued as hyp-NNN

  2. [PROCEED] (score: 7.2/10)
     Use low learning rate (0.01) with 2000 estimators; additionally add L2 regularization
     Novelty: 7  Feasibility: 8  Impact: 7
     Depth: 1 (refined from parent)
     -> Queued as hyp-NNN

  ...

Queued N hypotheses. Run /turing:train to test them.
```

## How It Differs From /turing:suggest

| | `/turing:suggest` | `/turing:explore` |
|---|---|---|
| **Source** | Web literature search | Tree search over critique scores |
| **Strategy** | Independent suggestions | Refinement chains (parent -> child) |
| **Requires internet** | Yes | No |
| **Discovers** | What papers recommend | What combinations score well |
| **Best for** | Early-stage exploration | Mid-experiment optimization |

## Integration

- Results feed into `hypotheses.yaml` — the next `/turing:train` picks them up
- `/turing:brief` shows queued treequest-sourced hypotheses
- `/turing:suggest --strategy treequest` is an alias for this command
- Human can override priority: `/turing:try` always takes precedence
