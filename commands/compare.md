---
name: compare
description: Compare two ML experiment runs side-by-side.
disable-model-invocation: true
argument-hint: "<exp-id-1> <exp-id-2>"
---

Compare two ML experiment runs side-by-side to understand what changed and why one performed better.

## Steps

1. Run the comparison command with the two experiment IDs provided:
   ```bash
   source .venv/bin/activate && python scripts/compare_runs.py $0 $1
   ```

2. Present the comparison with analysis:
   - **Metric differences:** All metrics from config.yaml `evaluation.metrics` for both runs
   - **What changed:** model type, hyperparameters, features, configuration
   - **Why one performed better:** interpret the metric deltas in context of the changes made
   - **Recommendation:** which approach is more promising for future experiments

3. If either experiment ID is not found, report the error and suggest running `/helios:status` to see available experiment IDs.
