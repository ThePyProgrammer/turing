---
title: "Model Composition"
description: "Combine models via ensembling, compose pipelines from swappable stages, and warm-start new experiments from existing checkpoints."
---

# Model Composition

Commands for building on top of existing work: combining multiple models into ensembles, decomposing pipelines into reusable stages, and initializing new experiments from prior checkpoints.

---

### `/turing:ensemble`: Automated ensemble construction

Combines top-K models via voting, stacking, and blending for zero-cost improvement. Often yields 1-3% improvement with no additional training. The ensemble builder evaluates prediction diversity, tries multiple combination methods, and reports the best one with improvement deltas against the best single model.

**Syntax:** `/turing:ensemble [--top-k 5] [--methods voting,stacking,blending] [--predictions-dir experiments/predictions] [--json]`

**Examples:**
```
/turing:ensemble                              # Default: top-5, all methods
/turing:ensemble --top-k 3                    # Top-3 models only
/turing:ensemble --methods voting,stacking    # Specific methods
/turing:ensemble --json                       # Machine-readable output
```

---

### `/turing:stitch`: Pipeline composition

Decompose ML pipelines into swappable stages that can be independently varied, cached, and reused across experiments. When only the model stage changes, preprocessing and feature engineering are skipped. Stages include preprocess, features, model, and postprocess (configurable in `config.yaml` under `pipeline.stages`).

**Syntax:** `/turing:stitch <show|swap|cache|run> [stage] [--from exp-id]`

**Examples:**
```
/turing:stitch show                          # Display pipeline stages
/turing:stitch swap model --from exp-031     # Keep features, swap model
/turing:stitch cache                         # Cache intermediate outputs
/turing:stitch run                           # Run with cached stages
```

---

### `/turing:warm`: Warm-start from checkpoint

Take a trained checkpoint and use it as initialization for a new experiment. Automates the "start from here but change X" pattern with model-type-aware strategies: continued boosting for tree models, weight loading with optional layer freezing for neural networks, and `warm_start=True` for scikit-learn. Supports gradual unfreezing and learning rate reduction.

**Syntax:** `/turing:warm <exp-id> [--freeze-layers encoder] [--unfreeze-after 5] [--lr-factor 0.1] [--json]`

**Examples:**
```
/turing:warm exp-042                                   # Auto-detect strategy
/turing:warm exp-042 --freeze-layers encoder           # Freeze encoder layers
/turing:warm exp-042 --freeze-layers encoder --unfreeze-after 5  # Gradual unfreezing
/turing:warm exp-042 --lr-factor 0.01                  # Very small fine-tuning LR
```
