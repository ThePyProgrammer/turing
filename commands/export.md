---
name: export
description: Export model to production format with equivalence verification, latency benchmarking, and deployment model card.
argument-hint: "[exp-id] [--format joblib|xgboost_json|onnx|torchscript|tflite]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Export a trained model to a production-ready format.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - First argument can be an experiment ID (e.g., `exp-042`); defaults to best
   - `--format joblib|xgboost_json|onnx|torchscript|tflite` specifies export format (auto-detected if omitted)
   - `--skip-equivalence` skips inference equivalence check
   - `--skip-latency` skips latency benchmark
   - `--samples 100` sets test sample count

3. **Run export pipeline:**
   ```bash
   python scripts/export_model.py $ARGUMENTS
   ```

4. **Report results:**
   - **Export:** format, file size, output path, dependencies
   - **Equivalence:** verdict (equivalent/approximately_equivalent/divergent), max delta
   - **Latency:** p50/p95/p99 ms, speedup vs original
   - **Model Card:** metrics, seed study, equivalence, latency, dependencies

5. **Output:** exported model + model_card.yaml written to `exports/exp-NNN/`

6. **If model file not found:** suggest checking models/best/ directory.

## Examples

```
/turing:export                                     # Best experiment, default format
/turing:export exp-042                             # Specific experiment
/turing:export --format xgboost_json               # Native XGBoost JSON
/turing:export --format onnx                       # ONNX format
/turing:export --skip-equivalence --skip-latency   # Fast export
```
