---
title: "Model Lifecycle"
description: "Incremental model updates with forgetting detection and a model registry for tracking promotion from candidate to production."
---

# Model Lifecycle

Commands for managing models beyond initial training -- updating them with new data without full retraining, and governing the promotion pipeline from candidate to production.

---

### `/turing:update` -- Incremental model update

Add new data to an existing model without starting from scratch. Model-specific strategies: continued boosting with additional rounds for XGBoost/LightGBM, fine-tuning with reduced learning rate and replay buffer for neural networks, and `partial_fit()` or `warm_start=True` for scikit-learn. Includes automatic catastrophic forgetting detection -- if performance on old data degrades beyond tolerance, the update is flagged.

**Syntax:** `/turing:update <exp-id> --new-data <path> [--replay-ratio 0.1] [--tolerance 0.005] [--json]`

**Examples:**
```
/turing:update exp-089 --new-data data/new_batch.csv
/turing:update exp-089 --new-data data/new.csv --replay-ratio 0.2
/turing:update exp-089 --new-data data/new.csv --tolerance 0.01
/turing:update exp-089 --new-data data/new.csv --json
```

---

### `/turing:registry` -- Model registry

Track which model is production, staging, candidate, or archived. Promotion between stages requires passing gates: candidate to staging requires a regression check and seed study; staging to production requires a methodology audit and calibration check. Use `--force` to skip gate checks when necessary. The registry maintains a full promotion/demotion history for each model.

**Syntax:** `/turing:registry [list|register|promote|demote|archive|history] [exp-id] [stage]`

**Examples:**
```
/turing:registry list
/turing:registry register exp-095 --version v4.1
/turing:registry promote exp-089 staging
/turing:registry promote exp-089 production --force
/turing:registry demote exp-078 staging --reason "latency regression"
/turing:registry archive exp-042 --reason "superseded by v4"
/turing:registry history
/turing:registry history exp-089
```
