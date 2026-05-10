---
name: preflight
description: Pre-flight resource check — estimates VRAM, RAM, and disk requirements before running ML training. Compares against available system resources and issues PASS/WARN/FAIL verdict. Use before training to catch OOM errors before they happen.
disable-model-invocation: true
argument-hint: "[--model-type torch] [--params 10M] [--batch-size 32]"
allowed-tools: Read, Bash(python scripts/*:*, source .venv/bin/activate:*, nvidia-smi:*), Grep, Glob
---

Check whether the current system has enough resources to run the planned experiment.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Run preflight check:**

   If `$ARGUMENTS` is empty (auto-detect from config.yaml):
   ```bash
   python scripts/preflight.py
   ```

   If `$ARGUMENTS` contains flags:
   ```bash
   python scripts/preflight.py $ARGUMENTS
   ```

3. **Interpret the verdict:**

   - **PASS** — system has sufficient resources. Proceed with training.
   - **WARN** — resources are tight. Training may succeed but could be slow or unstable. Present warnings to the user and ask whether to proceed.
   - **FAIL** — training will likely fail (OOM, disk full, no GPU for GPU-required model). Present the specific resource gap and suggest mitigations:
     - RAM too low: reduce dataset size, use chunked loading, or add swap
     - VRAM too low: reduce batch size, use fp16/bf16, enable gradient checkpointing, or use a smaller model
     - Disk too low: clean up old models/checkpoints
     - No GPU: switch to a CPU-friendly model (XGBoost, LightGBM, sklearn)

4. **If running before `/turing:train`:** report the verdict so the human can decide whether to proceed, adjust config, or choose a different model type.

## Examples

```bash
# Auto-detect from config.yaml (works for Turing projects)
/turing:preflight

# Check for a specific model type
/turing:preflight --model-type transformer --params 350M --batch-size 16 --precision fp16

# Check with a specific dataset
/turing:preflight --model-type xgboost --dataset data/train.csv

# JSON output for scripting
/turing:preflight --json
```

## What It Checks

| Resource | How estimated | Warning threshold |
|----------|--------------|-------------------|
| **RAM** | Dataset size (4x CSV on disk) + model memory (tree nodes or param count) | >90% of available |
| **VRAM** | Model params + gradients + optimizer state + activations | >80% of largest GPU |
| **Disk** | Model artifacts + dataset + checkpoints | >50% of free space |
| **GPU presence** | torch.cuda or nvidia-smi | Required for neural nets >1GB VRAM |

## Model-Specific Estimates

| Model Type | RAM | VRAM | GPU Required? |
|-----------|-----|------|---------------|
| XGBoost/LightGBM | Trees + data (typically <4GB) | 0 | No |
| Random Forest | Trees + data (can be large) | 0 | No |
| Linear/Logistic | 2x data | 0 | No |
| MLP (small) | Data + params | Params x 4 (Adam) | If >1GB VRAM |
| Transformer | Data + params | Params x 4 + activations | Yes |
