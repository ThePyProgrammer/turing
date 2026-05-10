---
name: merge
description: Model merging — average weights from multiple checkpoints into a single model (soups, TIES, DARE). Free accuracy, zero latency cost.
disable-model-invocation: true
argument-hint: "<exp-ids...> [--method uniform|greedy|ties|dare]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Combine model weights (not predictions) into a single, better model with no latency overhead.

## Steps

1. **Activate environment:** `source .venv/bin/activate`
2. **Run:** `python scripts/model_merger.py $ARGUMENTS`
3. **Methods:** uniform soup (simple average), greedy soup (include only if improves), TIES (trim+elect+merge), DARE (drop+rescale)
4. **Report:** compatibility check, per-model metrics, method comparison, improvement delta
5. **Saved output:** `experiments/merges/merge-*.yaml`

## Examples

```
/turing:merge exp-042 exp-053 exp-067              # All methods
/turing:merge exp-042 exp-053 --method greedy      # Greedy soup only
```
