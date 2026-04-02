---
title: "Experiment Intelligence"
description: "Seven commands for deep experiment analysis: error diagnosis, ablation studies, Pareto frontiers, model cards, logbooks, reports, and posters."
---

# Experiment Intelligence

These commands go beyond aggregate metrics to understand *why* models behave the way they do and communicate findings effectively. From failure-mode clustering to publication-ready posters, this category covers analysis and reporting.

---

### `/turing:diagnose` -- Error Analysis

Analyze where and why the model fails, beyond aggregate metrics. Clusters failure cases, identifies systematic failure modes, and suggests targeted fixes with auto-queued hypotheses.

For classification tasks, the diagnosis includes confusion matrices, most-confused pairs, and per-class precision/recall/F1. For regression, it covers residual statistics, P90/P95 errors, feature-range bias, and systematic bias detection. Failure modes are ranked by impact.

**Syntax:** `/turing:diagnose [exp-id] [--auto-queue] [--top 5]`

- Defaults to the best experiment if no ID is provided.
- `--auto-queue` generates hypotheses targeting identified weaknesses and adds them to `hypotheses.yaml`.
- `--top N` limits the number of failure modes reported (default: 5).

**Examples:**

```
/turing:diagnose
# Analyze best experiment's failure modes

/turing:diagnose exp-042
# Diagnose a specific experiment

/turing:diagnose --auto-queue
# Diagnose and auto-queue fix hypotheses for the next /turing:train

/turing:diagnose --top 10
# Show top 10 failure modes instead of default 5
```

!!! tip
    Run `/turing:diagnose --auto-queue` when the agent has plateaued. The auto-generated hypotheses target specific weaknesses rather than generic "try something different" suggestions.

---

### `/turing:ablate` -- Ablation Study

Run a systematic ablation study: remove components one at a time, measure impact, and produce a publication-ready table. Flags dead-weight components whose removal actually improves the metric.

Each ablation can be run with multiple seeds for statistical robustness. Output supports both markdown and LaTeX table formats.

**Syntax:** `/turing:ablate [exp-id] [--components "X,Y,Z"] [--seeds 3] [--latex]`

- Defaults to the best experiment if no ID is provided.
- `--components "dropout,feature_X,regularization"` specifies which components to ablate.
- `--seeds 3` runs each ablation multiple times for robustness.
- `--latex` outputs a LaTeX-formatted table for direct paper inclusion.

**Examples:**

```
/turing:ablate
# Auto-detect ablatable components from the best experiment

/turing:ablate exp-042
# Ablate a specific experiment's components

/turing:ablate --components "dropout,subsample"
# Ablate specific named components

/turing:ablate --seeds 3 --latex
# Multi-seed ablation with LaTeX table output
```

!!! tip
    Dead-weight detection is one of the most valuable outputs. If removing a component *improves* the metric, you have found unnecessary complexity. Simplify the model and pocket the free improvement.

---

### `/turing:frontier` -- Pareto Frontier Visualization

Visualize the Pareto frontier across multiple objectives. Answers "which model is actually best?" when there are tradeoffs between accuracy, speed, and model size.

Identifies Pareto-optimal experiments (no other experiment is better on all metrics simultaneously) and dominated experiments (strictly worse than at least one Pareto-optimal point).

**Syntax:** `/turing:frontier [--metrics "accuracy,train_seconds,n_params"] [--ascii]`

- `--metrics` specifies which metrics to analyze (default: primary metric + train_seconds).
- `--ascii` generates an ASCII scatter plot with 2D projection.

**Examples:**

```
/turing:frontier
# Default: primary metric vs training time

/turing:frontier --metrics "accuracy,train_seconds"
# 2D Pareto frontier

/turing:frontier --metrics "accuracy,train_seconds,n_params"
# 3D frontier across accuracy, speed, and model size

/turing:frontier --ascii
# Include an ASCII scatter plot (* for Pareto, . for dominated)
```

!!! tip
    Use the Pareto frontier when choosing a model for production. The "best" model on your primary metric might be 10x slower than a model that is only 0.5% worse. The frontier makes these tradeoffs explicit.

---

### `/turing:card` -- Model Card Generation

Generate a standardized model card documenting the trained model: type, performance, training data, limitations, intended use, and artifact contract. Follows the model card format for responsible ML documentation.

**Syntax:** `/turing:card`

No arguments. Reads from `config.yaml`, `experiments/log.jsonl`, and `model_contract.md` to populate the card.

**Examples:**

```
/turing:card
# Generate MODEL_CARD.md from experiment logs and config

/turing:card
# If no experiments exist, generates a skeleton card
```

!!! tip
    After generating the card, review the Ethical Considerations and Intended Use sections manually. The generator fills in what it can from experiment data, but bias, fairness, and "what the model is NOT intended for" require human judgment.

---

### `/turing:logbook` -- Research Logbook

Generate a research logbook showing the full experiment narrative: hypotheses proposed, experiments run, decisions made, and progress over time. Outputs HTML (with interactive Chart.js trajectory chart) or markdown.

The logbook captures the complete story of the research campaign, not just the final result.

**Syntax:** `/turing:logbook [--since YYYY-MM-DD] [--format html|markdown] [--output path]`

- `--since` filters to events after a specific date.
- `--format` selects HTML (default, with interactive chart) or markdown.
- `--output` writes to a file instead of stdout.

**Examples:**

```
/turing:logbook --output logbook.html
# Full HTML logbook with interactive trajectory chart

/turing:logbook --format markdown --output logbook.md
# Markdown format for embedding in documentation

/turing:logbook --since 2026-03-24 --output logbook.html
# Last week's activity only
```

!!! tip
    The HTML logbook with the interactive trajectory chart is excellent for sharing with collaborators. Open `logbook.html` in a browser to see the metric progression and experiment-by-experiment narrative.

---

### `/turing:report` -- Research Report

Generate a structured markdown research report from experiment history. More detailed than a brief, less visual than a poster. Includes executive summary, methodology, key findings, model comparison, and recommendations.

The report enhances the raw logbook data with synthesized analysis: which model families outperformed others, what hyperparameter ranges work, surprising results, and failure patterns.

**Syntax:** `/turing:report [--since YYYY-MM-DD] [--output path]`

- `--since` filters to events after a specific date.
- `--output` writes to a file instead of stdout.

**Examples:**

```
/turing:report
# Display the full report inline

/turing:report --output reports/campaign-v1.md
# Save to file for archiving

/turing:report --since 2026-03-15 --output reports/week-12.md
# Report covering a specific time window
```

!!! tip
    Use `/turing:report` at the end of a research campaign or before a team review. It provides the narrative context that raw metrics tables lack: *why* certain approaches worked, *what* patterns emerged, and *where* to go next.

---

### `/turing:poster` -- Research Poster

Generate a single-page HTML research poster summarizing the experiment campaign: best result, trajectory chart, key findings, and methodology. Self-contained HTML file with no build step -- works when opened as `file://`.

Poster layout uses a card-based design with colored top borders, clean typography, and a Chart.js trajectory visualization. Print-optimized CSS supports direct PDF export from the browser.

**Syntax:** `/turing:poster [title override]`

- Optional argument overrides the poster title (defaults to the task description from `config.yaml`).

**Examples:**

```
/turing:poster
# Generate poster/index.html with default title from config

/turing:poster "Customer Churn Prediction: An Autonomous ML Study"
# Override the poster title
```

!!! tip
    Open `poster/index.html` in your browser and use Ctrl+P / Cmd+P to save as PDF. The poster is formatted for A1 landscape by default. This is an effective way to share campaign results in a meeting or on a lab wall.
