---
name: diagnose
description: Error analysis — cluster failure cases, identify systematic failure modes, and suggest targeted fixes with auto-queued hypotheses.
disable-model-invocation: true
argument-hint: "[exp-id] [--auto-queue] [--top 5]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Analyze where and why the model fails, beyond aggregate metrics.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Generate predictions if needed:**
   Check if `experiments/predictions/exp-NNN-preds.yaml` exists. If not, run:
   ```bash
   python train.py --predict-only --output experiments/predictions/
   ```
   The predictions file must contain `y_true`, `y_pred`, `task_type`, and optionally `features`.

3. **Parse arguments from `$ARGUMENTS`:**
   - First argument can be an experiment ID (e.g., `exp-042`); defaults to best
   - `--auto-queue` auto-queues hypotheses from failure modes into `hypotheses.yaml`
   - `--top 5` limits to top N failure modes (default 5)

4. **Run error analysis:**
   ```bash
   python scripts/diagnose_errors.py $ARGUMENTS
   ```

5. **Report results:**
   - **Classification:** confusion matrix, most-confused pairs, per-class P/R/F1, low-recall classes
   - **Regression:** residual stats, P90/P95 errors, feature-range bias, systematic bias
   - **Failure modes:** ranked by impact, with suggested fixes
   - **Auto-hypotheses:** if `--auto-queue`, shows queued hypotheses targeting weaknesses

6. **Saved output:** report written to `experiments/diagnoses/exp-NNN-diagnosis.yaml`

7. **If no predictions file exists:** instruct user to run the model on validation set first.

## Examples

```
/turing:diagnose                    # Analyze best experiment
/turing:diagnose exp-042            # Specific experiment
/turing:diagnose --auto-queue       # Queue fix hypotheses
/turing:diagnose --top 10           # Top 10 failure modes
```
