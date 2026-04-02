---
title: "Scaling & Efficiency"
description: "Predict full-scale performance from small experiments, manage compute budgets, and compress models via distillation."
---

# Scaling & Efficiency

Commands for making informed decisions about compute allocation -- predicting whether scaling up is worth it, enforcing experiment budgets, and compressing large models for production deployment.

---

### `/turing:scale` -- Scaling law estimator

Run small experiments at different sizes, fit a power law, and predict full-scale performance before committing compute. Answers "is it worth training on the full dataset?" in 30 minutes instead of 3 days. Supports scaling along data, compute, and parameter axes. Use plan mode to generate scale point configs, then analyze mode to fit the curve after running them.

**Syntax:** `/turing:scale [--axis data|compute|params] [--points 4] [--analyze results.yaml] [--plot] [--json]`

**Examples:**
```
/turing:scale                                  # Plan: data axis, 4 points
/turing:scale --axis compute --points 3        # Plan: compute axis, 3 points
/turing:scale --analyze results.yaml --plot    # Analyze with ASCII plot
```

---

### `/turing:budget` -- Compute budget manager

Set a compute ceiling and let the system optimize within it. Prevents runaway experiment loops. The budget manager tracks allocation across explore and exploit phases, automatically shifting modes as the budget is consumed: exploration early, exploitation late, hard stop at 100%. The `/turing:train` loop checks the budget before each experiment.

**Syntax:** `/turing:budget <set|status|reset|check> [--experiments 50] [--hours 8] [--json]`

**Examples:**
```
/turing:budget set --experiments 50 --hours 8   # Set both constraints
/turing:budget set --experiments 30             # Experiment count only
/turing:budget status                           # Show usage and projections
/turing:budget reset                            # Remove budget limits
```

---

### `/turing:distill` -- Model distillation

Compress a large model into a smaller, faster one for production via knowledge distillation. A student model is trained to match the teacher's predictions. Measures the accuracy/size/latency tradeoff and issues a verdict from EXCELLENT to TOO MUCH LOSS. Supports soft-label, feature-matching, and dataset distillation methods across tree, neural, and scikit-learn model types.

**Syntax:** `/turing:distill <teacher-exp-id> [--compression 4] [--method soft_labels|feature_matching|dataset_distillation] [--target-latency 5] [--json]`

**Examples:**
```
/turing:distill exp-042                              # 4x compression, soft labels
/turing:distill exp-042 --compression 8              # Aggressive compression
/turing:distill exp-042 --method feature_matching    # Neural feature alignment
/turing:distill exp-042 --target-latency 5           # Meet 5ms latency target
```
