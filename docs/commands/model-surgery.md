---
title: "Model Surgery"
description: "Weight pruning, post-training quantization, model merging, and programmatic architecture modification."
---

# Model Surgery

Commands for modifying models after training: removing redundant weights, reducing numerical precision, merging weights from multiple checkpoints, and making structural architecture changes.

---

### `/turing:prune`: Weight pruning

Remove redundant weights for faster inference and smaller models. Measures accuracy at different sparsity levels to find the knee point where further pruning degrades performance. Methods include magnitude pruning (zero small weights), structured pruning (remove entire neurons), and lottery ticket (iterative pruning with weight rewinding). For tree models, progressively reduces `n_estimators`.

**Syntax:** `/turing:prune <exp-id> [--sparsity 0.5,0.75,0.9] [--method magnitude|structured|lottery]`

**Examples:**
```
/turing:prune exp-042                              # Default: magnitude, 5 levels
/turing:prune exp-042 --method structured          # Remove entire neurons
/turing:prune exp-042 --sparsity 0.5,0.75,0.9     # Custom levels
```

---

### `/turing:quantize`: Post-training quantization

Quantize for production with minimal effort. Achieves 2-4x speedup and 2-4x memory reduction with typically less than 0.5% accuracy loss. Compares FP32 (baseline), FP16 (GPU), INT8 dynamic (simplest), and INT8 static (best accuracy) precision levels. Suggests quantization-aware training if post-training quantization causes unacceptable degradation.

**Syntax:** `/turing:quantize <exp-id> [--precision int8|fp16|dynamic]`

**Examples:**
```
/turing:quantize exp-042                    # Compare all precision levels
/turing:quantize exp-042 --precision int8   # INT8 specifically
```

---

### `/turing:merge`: Model weight merging

Combine model weights (not predictions) into a single, better model with no latency overhead. Methods include uniform soup (simple average), greedy soup (include only if it improves), TIES (trim, elect, merge), and DARE (drop and rescale). Unlike ensembling, the merged model has the same inference cost as a single model.

**Syntax:** `/turing:merge <exp-ids...> [--method uniform|greedy|ties|dare]`

**Examples:**
```
/turing:merge exp-042 exp-053 exp-067              # All methods
/turing:merge exp-042 exp-053 --method greedy      # Greedy soup only
```

---

### `/turing:surgery`: Architecture modification

Programmatic architecture changes with automatic warm-start from existing weights. Specify what to change and the system handles how. Operations include add-layer, remove-layer, widen, narrow, swap-activation, add-skip, add-norm, deepen, and swap-objective. For tree models: deepen (increase `max_depth`), widen (more estimators), and swap-objective.

**Syntax:** `/turing:surgery <exp-id> --op <operation> [args...]`

**Examples:**
```
/turing:surgery exp-042 --op widen 2             # 2x wider hidden layers
/turing:surgery exp-042 --op add-layer           # Insert a layer
/turing:surgery exp-042 --op swap-activation relu gelu  # ReLU -> GELU
/turing:surgery exp-042 --op deepen              # Deeper trees
```
