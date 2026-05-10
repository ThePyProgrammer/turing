---
name: warm
description: Warm-start from a prior model — load checkpoint, optionally freeze layers, adjust learning rate, and continue training.
argument-hint: "<exp-id> [--freeze-layers encoder] [--unfreeze-after 5]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Take a trained checkpoint and use it as initialization for a new experiment. Automates the "start from here but change X" pattern.

## Steps

1. **Sync environment:**
   ```bash
   uv sync
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - First argument is the source experiment ID (required)
   - `--freeze-layers encoder decoder` — layer names to freeze (neural only)
   - `--unfreeze-after 5` — unfreeze all layers after N epochs (gradual unfreezing)
   - `--lr-factor 0.1` — learning rate reduction factor (default: 0.1x)
   - `--json` — raw JSON output

3. **Run warm-start planner:**
   ```bash
   uv run python scripts/warm_start.py $ARGUMENTS
   ```

4. **Report results:**
   - Model type detection (tree, neural, sklearn)
   - Strategy: continue_boosting, load_weights, or warm_start_param
   - Numbered step-by-step instructions
   - Config changes to apply
   - Checkpoint info (path, format, size)

5. **Strategies by model type:**
   - **Tree models (XGBoost/LightGBM):** continue boosting from existing trees with more estimators
   - **Neural networks:** load weights, optionally freeze layers, reset optimizer, reduce LR
   - **scikit-learn:** use `warm_start=True` parameter for incremental learning

6. **If no checkpoint found:** plan is still generated, but warns that checkpoint is needed

7. **Saved output:** report written to `experiments/warm_starts/warm-<exp-id>.yaml`

## Examples

```
/turing:warm exp-042                                   # Auto-detect strategy
/turing:warm exp-042 --freeze-layers encoder           # Freeze encoder layers
/turing:warm exp-042 --freeze-layers encoder --unfreeze-after 5  # Gradual unfreezing
/turing:warm exp-042 --lr-factor 0.01                  # Very small fine-tuning LR
```
