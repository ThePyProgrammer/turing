---
name: xray
description: Internal model diagnostics — gradient flow, dead neurons, activation stats, weight distributions, tree depth analysis.
argument-hint: "[exp-id] [--layer encoder.layer.2] [--compare exp-a exp-b]"
allowed-tools: Read, Bash(*), Grep, Glob
---

See inside the model. When it underperforms, the fix depends on *why*.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - Optional experiment ID
   - `--layer "name"` — focus on specific layer
   - `--compare exp-a exp-b` — side-by-side diagnostics
   - `--json` — raw JSON output

3. **Run model diagnostics:**
   ```bash
   python scripts/model_xray.py $ARGUMENTS
   ```

4. **Diagnostics by model type:**
   - **Neural networks:** gradient magnitudes, activation stats, dead neuron %, weight distributions, gradient-to-weight ratio
   - **Tree models:** depth utilization, leaf purity, feature split dominance
   - **scikit-learn:** coefficient magnitudes, feature importance concentration

5. **Issues detected:** dead gradients, vanishing/exploding gradients, dead neurons, sparse weights, feature dominance, overfitting risk

6. **Saved output:** report in `experiments/xrays/<exp-id>-xray.yaml`

## Examples

```
/turing:xray exp-042              # Full diagnostics
/turing:xray                      # Best experiment
```
