---
name: status
description: Show current ML experiment status -- best model, recent experiments, convergence state.
disable-model-invocation: true
---

Show the current state of the ML training pipeline.

## Steps

1. Run the status command:
   ```bash
   source .venv/bin/activate && python scripts/show_metrics.py --last 10
   ```

2. Summarize the results for the user:
   - **Current best model:** type, key metrics (as defined in config.yaml `evaluation.metrics`)
   - **Number of experiments run:** total from the experiment log
   - **Convergence status:** how many consecutive non-improvements, whether convergence threshold has been reached
   - **Recent trend:** are experiments improving, plateauing, or regressing?

3. If no experiments exist yet, report that the training pipeline is ready but no experiments have been run. Suggest running `/helios:train` to start.
