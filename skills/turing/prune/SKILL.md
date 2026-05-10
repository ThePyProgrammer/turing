---
name: prune
description: Weight pruning — measure accuracy at different sparsity levels, find the knee point, produce a smaller/faster model.
argument-hint: "<exp-id> [--sparsity 0.5,0.75,0.9] [--method magnitude|structured|lottery]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Remove redundant weights for faster inference and smaller models.

## Steps

1. **Sync environment:** `uv sync`
2. **Run:** `uv run python scripts/model_pruning.py $ARGUMENTS`
3. **Methods:** magnitude (zero small weights), structured (remove neurons), lottery (iterative with rewind)
4. **For tree models:** progressively reduces n_estimators
5. **Report:** sparsity sweep table, knee point, recommended sparsity
6. **Saved output:** `experiments/pruning/<exp-id>-pruning.yaml`

## Examples

```
/turing:prune exp-042                              # Default: magnitude, 5 levels
/turing:prune exp-042 --method structured          # Remove entire neurons
/turing:prune exp-042 --sparsity 0.5,0.75,0.9     # Custom levels
```
