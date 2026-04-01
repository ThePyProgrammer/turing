---
name: update
description: Incremental model update — add new data without full retraining, with forgetting detection.
disable-model-invocation: true
argument-hint: "<exp-id> --new-data <path> [--replay-ratio 0.1] [--tolerance 0.005]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Add new data to an existing model without starting from scratch. Detects catastrophic forgetting.

## Steps
1. `source .venv/bin/activate`
2. `python scripts/incremental_update.py $ARGUMENTS`
3. **Saved:** `experiments/updates/`

## Model-specific strategies
- **XGBoost/LightGBM:** continued boosting with additional rounds
- **Neural networks:** fine-tune with reduced LR + replay buffer from old data
- **scikit-learn:** partial_fit() or warm_start=True

## Examples
```
/turing:update exp-089 --new-data data/new_batch.csv
/turing:update exp-089 --new-data data/new.csv --replay-ratio 0.2
/turing:update exp-089 --new-data data/new.csv --tolerance 0.01
/turing:update exp-089 --new-data data/new.csv --json
```
