---
name: ensemble
description: Automated ensemble construction — combines top-K models via voting, stacking, and blending for zero-cost improvement.
argument-hint: "[--top-k 5] [--methods voting,stacking,blending]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Build ensembles from your best experiments automatically. Often yields 1-3% improvement with zero additional training.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - `--top-k 5` — number of top models to include (default: 5)
   - `--methods voting,stacking,blending` — ensemble methods to try
   - `--predictions-dir experiments/predictions` — directory with saved predictions
   - `--json` — raw JSON output

3. **Run ensemble construction:**
   ```bash
   python scripts/build_ensemble.py $ARGUMENTS
   ```

4. **Report results:**
   - Table of all ensemble methods tried with metric deltas vs best single model
   - Best ensemble method highlighted with improvement amount
   - Diversity analysis: prediction correlation matrix, diversity assessment
   - Base model summary: which experiments were combined

5. **Ensemble methods:**
   - **Voting:** majority vote (classification) or mean (regression)
   - **Weighted voting:** weights proportional to individual model performance
   - **Stacking:** cross-validated meta-learner (ridge/logistic) on out-of-fold predictions
   - **Blending:** holdout-based meta-learner (simpler, less data-efficient)

6. **Prerequisites:** experiments must have saved predictions in `experiments/predictions/`. Each experiment needs `<exp-id>-predictions.npy` and a shared `labels.npy`.

7. **If no predictions exist:** suggest saving predictions during training by adding prediction logging to `evaluate.py`.

8. **Saved output:** report written to `experiments/ensembles/ensemble-*.yaml`

## Examples

```
/turing:ensemble                              # Default: top-5, all methods
/turing:ensemble --top-k 3                    # Top-3 models only
/turing:ensemble --methods voting,stacking    # Specific methods
/turing:ensemble --json                       # Machine-readable output
```
