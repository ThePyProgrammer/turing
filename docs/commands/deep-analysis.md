---
title: "Deep Analysis"
description: "Deep diagnostic comparison of experiments, live training monitoring with early-warning alerts, and performance regression gating."
---

# Deep Analysis

Commands for understanding what changed between experiments, catching problems during training, and verifying that code or dependency changes have not silently degraded model performance.

---

### `/turing:diff` -- Deep experiment comparison

Goes beyond "which metric is higher" to show where, when, and why two experiments diverge. The diff covers config changes with magnitudes, metric deltas with statistical significance, per-class regressions hidden by aggregate improvement, training curve divergence points, and feature importance shifts. Optionally includes the git diff of `train.py` between the two experiments' commits.

**Syntax:** `/turing:diff <exp-a> <exp-b> [--code] [--json]`

**Examples:**
```
/turing:diff exp-042 exp-053                # Full diagnostic comparison
/turing:diff exp-042 exp-053 --code         # Include train.py code changes
/turing:diff exp-001 exp-010 --json         # Raw JSON output
```

---

### `/turing:watch` -- Live training monitor

Stream metrics during training with early-warning alerts for loss spikes, NaN values, overfitting onset, and metric plateaus. Catches problems mid-run instead of at the end. Supports both live monitoring (in a separate terminal) and post-hoc analysis of completed training logs. Alert thresholds are configurable via `config/watch_alerts.yaml`.

**Syntax:** `/turing:watch [--alerts] [--interval 10] [--analyze run.log] [--json]`

**Examples:**
```
/turing:watch --analyze run.log           # Analyze completed training
/turing:watch --analyze run.log --json    # JSON output for scripting
/turing:watch --alerts                    # Live: show only alerts
/turing:watch --interval 5               # Live: check every 5 seconds
```

---

### `/turing:regress` -- Performance regression gate

CI for your model. After any change to code, dependencies, or data, re-run the best experiment and verify metrics have not silently regressed. Reports PASS, WARNING, or FAIL with per-metric deltas and relative differences. On failure, suggests investigation commands including `/turing:diff` and environment comparison.

**Syntax:** `/turing:regress [--tolerance 0.01] [--against exp-id] [--quick] [--runs 5] [--json]`

**Examples:**
```
/turing:regress                              # Default: check best, 1% tolerance, 3 runs
/turing:regress --quick                      # Fast check: 1 run
/turing:regress --against exp-042            # Check specific experiment
/turing:regress --tolerance 0.005 --runs 5   # Strict: 0.5% tolerance, 5 runs
```
