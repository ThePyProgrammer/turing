---
title: "Research Workflow"
description: "Three commands for the broader research workflow: literature search, paper drafting, and model export to production formats."
---

# Research Workflow

These commands connect the experiment loop to the broader research and deployment lifecycle. `lit` searches the literature without leaving the terminal, `paper` drafts publication sections directly from experiment logs, and `export` packages models for production with equivalence verification.

---

### `/turing:lit`: Literature Search

Search the literature scoped to the current experiment domain. Find papers, SOTA baselines, and related work without leaving the terminal. Results can auto-queue hypotheses into the experiment pipeline.

Supports three modes: free query for general search, baseline mode for SOTA comparison against your best result, and related mode for finding papers that use similar methods to a specific experiment.

**Syntax:** `/turing:lit <query> | --baseline | --related <exp-id> [--auto-queue] [--limit N]`

- Free query searches Semantic Scholar for papers matching the query string.
- `--baseline` finds SOTA results for the current task and compares against your best experiment.
- `--related exp-042` finds papers using similar methods to a specific experiment.
- `--auto-queue` auto-queues hypotheses from literature findings with `source: "literature"`.
- `--limit N` caps the number of results.

**Examples:**

```
/turing:lit "gradient boosting missing values"
# Search for papers on a specific technique

/turing:lit --baseline
# Find SOTA benchmarks and compare against your current best

/turing:lit --related exp-042
# Find papers using similar methods to experiment 042

/turing:lit --auto-queue "ensemble methods"
# Search and auto-queue hypotheses from findings
```

!!! tip
    Use `--baseline` early in a project to understand where the state of the art is. If your best result is already within 1% of SOTA, you know diminishing returns are ahead. If there is a 10% gap, the literature often reveals which techniques close it.

---

### `/turing:paper`: Draft Paper Sections

Draft mechanical paper sections directly from experiment logs: setup, results, ablation tables, and hyperparameter appendices. Supports both LaTeX and markdown output. Numbers are pulled directly from experiment logs with no manual transcription.

Available sections: `setup` (experimental setup prose), `results` (comparison table with best bolded and seed study stats), `ablation` (from `/turing:ablate` results), `hyperparameters` (appendix-style parameter table per model).

**Syntax:** `/turing:paper [--sections setup,results,ablation,hyperparameters] [--format latex|markdown]`

- `--sections` selects which sections to draft (default: all).
- `--format` selects LaTeX (default) or markdown output.

**Examples:**

```
/turing:paper
# Draft all sections in LaTeX format

/turing:paper --format markdown
# All sections in markdown for README or documentation

/turing:paper --sections setup,results
# Just the experimental setup and results table

/turing:paper --sections ablation --format latex
# Just the ablation table in LaTeX for copy-paste into a paper
```

!!! tip
    The results table automatically bolds the best model and includes seed study statistics if available. Run `/turing:ablate` and `/turing:seed` before `/turing:paper` to get the most complete output. Sections are saved to `paper/sections/`.

---

### `/turing:export`: Export Model to Production

Export a trained model to a production-ready format with equivalence verification, latency benchmarking, and a deployment model card. Ensures the exported model produces the same predictions as the original.

Supported formats: `joblib`, `xgboost_json`, `onnx`, `torchscript`, `tflite`. The format is auto-detected from the model type if not specified.

**Syntax:** `/turing:export [exp-id] [--format joblib|xgboost_json|onnx|torchscript|tflite] [--skip-equivalence] [--skip-latency] [--samples N]`

- Defaults to the best experiment if no ID is provided.
- `--format` overrides auto-detection.
- `--skip-equivalence` skips the inference equivalence check.
- `--skip-latency` skips the latency benchmark.
- `--samples 100` sets the number of test samples for verification (default varies by format).

The export pipeline verifies:
- **Equivalence:** exported model produces the same predictions as the original (equivalent, approximately_equivalent, or divergent).
- **Latency:** p50/p95/p99 inference times with speedup vs the original.
- **Deployment card:** metrics, seed study results, equivalence verdict, latency, and dependency list.

**Examples:**

```
/turing:export
# Export best experiment in auto-detected format

/turing:export exp-042
# Export a specific experiment

/turing:export --format xgboost_json
# Export as native XGBoost JSON

/turing:export --format onnx
# Export to ONNX for cross-framework deployment

/turing:export --skip-equivalence --skip-latency
# Fast export without verification checks
```

!!! tip
    Always run the full export (with equivalence and latency checks) at least once. ONNX conversion in particular can introduce subtle numerical differences. The equivalence check catches these before they reach production. Exports are saved to `exports/exp-NNN/`.
