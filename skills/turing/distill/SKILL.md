---
name: distill
description: Model compression via distillation — train a smaller student model to match a larger teacher's predictions.
argument-hint: "<teacher-exp-id> [--compression 4] [--method soft-labels]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Compress a large model into a smaller, faster one for production. Measures the accuracy/size/latency tradeoff.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - First argument is teacher experiment ID (required)
   - `--compression 4` — compression ratio (default: 4x)
   - `--method soft_labels|feature_matching|dataset_distillation` — distillation method
   - `--target-latency 5` — auto-adjust compression to meet latency target (ms)
   - `--json` — raw JSON output

3. **Run distillation planner:**
   ```bash
   python scripts/model_distiller.py $ARGUMENTS
   ```

4. **Report includes:**
   - Teacher model metrics
   - Auto-selected student architecture (fewer trees/layers/width)
   - Estimated size reduction and latency improvement
   - Distillation configuration (temperature, alpha, loss function)
   - Verdict: EXCELLENT / ACCEPTABLE / MARGINAL / TOO MUCH LOSS

5. **Student selection by model type:**
   - **Tree models:** fewer estimators, shallower depth
   - **Neural networks:** fewer layers, narrower hidden dims
   - **scikit-learn:** simpler model family (RandomForest → DecisionTree)

6. **Distillation methods:**
   - **soft_labels:** train on teacher's probability outputs with temperature scaling
   - **feature_matching:** align intermediate representations (neural only)
   - **dataset_distillation:** train on teacher-labeled synthetic data

7. **Saved output:** report written to `experiments/distillations/distill-<exp-id>.yaml`

## Examples

```
/turing:distill exp-042                              # 4x compression, soft labels
/turing:distill exp-042 --compression 8              # Aggressive compression
/turing:distill exp-042 --method feature_matching    # Neural feature alignment
/turing:distill exp-042 --target-latency 5           # Meet 5ms latency target
```
