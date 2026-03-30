---
name: status
description: Show current ML experiment status — best model, recent experiments, convergence state, and trend analysis. Delegates to @ml-evaluator for read-only safety.
disable-model-invocation: true
---

Show the current state of the ML training pipeline. This is an observation-only operation — no code is modified.

## Steps

1. **Run metrics display:**
   ```bash
   source .venv/bin/activate && python scripts/show_metrics.py --last 10
   ```

2. **Summarize for the user:**
   - **Best model:** type, key metrics, experiment ID
   - **Total experiments:** count from the log
   - **Convergence state:** consecutive non-improvements vs patience threshold
   - **Trend:** improving, plateauing, or regressing?
   - **Recommendation:** continue training, try a different approach, or declare convergence

3. **If no experiments exist:** report that the pipeline is ready but untrained. Suggest `/turing:train`.
