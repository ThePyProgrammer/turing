---
title: "Pre-Training Intelligence"
description: "Sanity checks, automatic baselines, and data leakage detection. Everything to run before committing to a full training loop."
---

# Pre-Training Intelligence

Commands to run before investing compute in a full training loop. Catch broken pipelines in seconds, establish baseline performance floors, and detect the most common cause of "too good to be true" results.

---

### `/turing:sanity`: Pre-training sanity checks

Run a battery of fast checks before committing to a full training run. Catches wiring bugs in seconds: broken data loaders, misconfigured losses, dead gradients, and models that cannot memorize a single batch. The single-batch overfit test is the most powerful: if the model cannot fit one batch in 50 steps, something is fundamentally wrong.

**Syntax:** `/turing:sanity [--quick] [--verbose] [--json]`

**Examples:**
```
/turing:sanity                    # Full check (~30 seconds)
/turing:sanity --quick            # Skip overfit test (~5 seconds)
```

---

### `/turing:baseline`: Automatic baseline generation

Generate trivial baselines so you always know if your model is meaningfully better than simple approaches. Produces random, majority/mean, linear, and k-NN baselines in 60 seconds, evaluated with the same protocol as real experiments. Satisfies the "baseline comparison" check in `/turing:audit`.

**Syntax:** `/turing:baseline [--methods all|simple|linear] [--data data.npz] [--json]`

**Examples:**
```
/turing:baseline                           # All baselines
/turing:baseline --methods simple          # Just random + majority
/turing:baseline --data data/processed.npz # With actual data
```

---

### `/turing:leak`: Data leakage detection

Actively probe for data leakage, the number one cause of "too good to be true" results. Checks feature-target correlation, single-feature predictiveness (with `--deep`), and train/test overlap via hash-based deduplication. Issues CLEAN, SUSPICIOUS, or LEAKAGE DETECTED verdicts. Satisfies the "data leakage" check in `/turing:audit`.

**Syntax:** `/turing:leak [--deep] [--features "feat_1,feat_2"] [--json]`

**Examples:**
```
/turing:leak                    # Standard correlation + overlap checks
/turing:leak --deep             # Full single-feature analysis
```
