---
name: frontier
description: Visualize Pareto frontier across multiple objectives — answers "which model is actually best?" when there are tradeoffs.
argument-hint: "[--metrics \"accuracy,train_seconds,n_params\"] [--ascii]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Visualize the Pareto frontier across multiple objectives from experiment history.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - `--metrics "accuracy,train_seconds,n_params"` specifies metrics to analyze
   - Without `--metrics`, uses primary metric + train_seconds from config
   - `--ascii` generates an ASCII scatter plot (2D projection)

3. **Run Pareto analysis:**
   ```bash
   python scripts/pareto_frontier.py $ARGUMENTS
   ```

4. **Report results:**
   - **Pareto-optimal experiments:** table with all metrics and what each is best at
   - **Dominated experiments:** with their nearest Pareto neighbor
   - **ASCII scatter plot** (if `--ascii`): 2D projection with * for Pareto, · for dominated
   - Summary: "N Pareto-optimal of M experiments across K metrics"

5. **Saved output:** results written to `experiments/frontiers/frontier-YYYY-MM-DD.yaml`

6. **If no experiments have all requested metrics:** suggest which metrics are available.

## Examples

```
/turing:frontier                                              # Default: metric vs time
/turing:frontier --metrics "accuracy,train_seconds"           # 2D frontier
/turing:frontier --metrics "accuracy,train_seconds,n_params"  # 3D frontier
/turing:frontier --ascii                                      # With scatter plot
```
