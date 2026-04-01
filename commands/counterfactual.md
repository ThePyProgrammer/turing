---
name: counterfactual
description: Input-level counterfactual explanations — find the smallest input change to flip a prediction.
disable-model-invocation: true
argument-hint: "<exp-id> --sample <index> [--target <class>]"
allowed-tools: Read, Bash(*), Grep, Glob
---

What would need to change to flip this prediction? Minimum-change counterfactual for individual predictions.

## Steps
1. `source .venv/bin/activate`
2. `python scripts/counterfactual_explanation.py $ARGUMENTS`
3. **Saved:** `experiments/counterfactuals/`

## Methods
- **Greedy perturbation:** change one feature at a time, find minimum flip
- **Prototype-based:** find nearest training sample from target class
- Both methods run and the best (smallest distance) is selected

## Examples
```
/turing:counterfactual exp-042 --sample 1247
/turing:counterfactual exp-042 --sample 1247 --target 0
/turing:counterfactual exp-042 --batch-misclassified
/turing:counterfactual exp-042 --sample 500 --json
```
