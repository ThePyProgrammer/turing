---
name: quantize
description: Post-training quantization — FP32→INT8/FP16, measure accuracy loss, 2-4x speedup with <0.5% accuracy loss.
argument-hint: "<exp-id> [--precision int8|fp16|dynamic]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Quantize for production. Lowest-effort optimization: 2-4x speedup, 2-4x memory reduction.

## Steps

1. **Activate environment:** `source .venv/bin/activate`
2. **Run:** `python scripts/model_quantization.py $ARGUMENTS`
3. **Precision levels:** FP32 (baseline), FP16 (GPU), INT8 dynamic (simplest), INT8 static (best accuracy)
4. **Report:** precision comparison table, recommended level, QAT suggestion if needed
5. **Saved output:** `experiments/quantization/<exp-id>-quantization.yaml`

## Examples

```
/turing:quantize exp-042                    # Compare all precision levels
/turing:quantize exp-042 --precision int8   # INT8 specifically
```
