---
name: surgery
description: Architecture modification — add/remove layers, widen/narrow, swap activations, inject skip connections. Specify what to change, system handles how.
disable-model-invocation: true
argument-hint: "<exp-id> --op <operation> [args...]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Programmatic architecture changes with auto warm-start from existing weights.

## Steps

1. **Activate environment:** `source .venv/bin/activate`
2. **Run:** `python scripts/architecture_surgery.py $ARGUMENTS`
3. **Operations:** add-layer, remove-layer, widen, narrow, swap-activation, add-skip, add-norm, deepen, swap-objective
4. **For tree models:** deepen (increase max_depth), widen (more estimators), swap-objective
5. **Report:** operation details, config changes, parameter count delta, warm-start source
6. **Saved output:** `experiments/surgery/<exp-id>-<op>.yaml`

## Examples

```
/turing:surgery exp-042 --op widen 2             # 2x wider hidden layers
/turing:surgery exp-042 --op add-layer           # Insert a layer
/turing:surgery exp-042 --op swap-activation relu gelu  # ReLU → GELU
/turing:surgery exp-042 --op deepen              # Deeper trees
```
