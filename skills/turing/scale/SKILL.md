---
name: scale
description: Scaling law estimator — run small experiments at different sizes, fit a power law, and predict full-scale performance before committing compute.
argument-hint: "[--axis data|compute|params] [--points 4] [--analyze results.yaml]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Predict full-scale performance from a handful of small experiments. Answers "is it worth training on the full dataset?" in 30 minutes instead of 3 days.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - `--axis data|compute|params` — scaling axis (default: data)
   - `--points 4` — number of scale points (default: 4)
   - `--analyze results.yaml` — analyze existing results instead of planning
   - `--plot` — include ASCII scaling plot
   - `--json` — raw JSON output

3. **Plan or analyze:**
   - **Plan mode (default):** generates scale point configs to run
     ```bash
     python scripts/scaling_estimator.py --axis data --points 4
     ```
   - **Analyze mode:** fits power law to completed results
     ```bash
     python scripts/scaling_estimator.py --analyze experiments/scaling/results.yaml
     ```

4. **Scaling axes:**
   - **data:** train on 10%, 25%, 50%, 75% of dataset
   - **compute:** train for 10%, 25%, 50%, 75% of max epochs
   - **params:** scale model size (fewer estimators, shallower depth)

5. **After planning:** run each scale point experiment, record results in YAML, then use `--analyze` to fit the curve

6. **Report includes:**
   - Power law fit: `metric = a × n^b` with R²
   - Predictions for 100%, 150%, 200% scale
   - Verdict: DIMINISHING RETURNS / MARGINAL GAINS / WORTH SCALING

7. **Saved output:** report written to `experiments/scaling/scale-YYYY-MM-DD.yaml`

## Examples

```
/turing:scale                                  # Plan: data axis, 4 points
/turing:scale --axis compute --points 3        # Plan: compute axis, 3 points
/turing:scale --analyze results.yaml --plot    # Analyze with ASCII plot
```
