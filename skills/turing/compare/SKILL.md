---
name: compare
description: Compare two ML experiment runs side-by-side — metrics, configuration deltas, and a verdict on which approach is more promising.
argument-hint: "<exp-id-1> <exp-id-2>"
allowed-tools: Read, Bash(*), Grep, Glob
---

Compare two ML experiment runs side-by-side to understand what changed and why one performed better.

## Steps

1. **Run comparison:**
   ```bash
   uv run python scripts/compare_runs.py $0 $1
   ```

2. **Analyze the delta:**
   - **Metric differences:** all configured metrics for both runs
   - **Configuration delta:** what changed (model type, hyperparameters, features)
   - **Causal analysis:** which changes likely caused the metric difference
   - **Verdict:** which approach is more promising for future experiments

3. **If either ID is missing:** report the error and suggest `/turing:status` to see available experiment IDs.
