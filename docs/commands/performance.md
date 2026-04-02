---
title: "Performance & Resources"
description: "Two commands for profiling training runs and managing model checkpoints: profile and checkpoint."
---

# Performance & Resources

These commands help you understand and optimize the computational cost of your experiments. `profile` identifies bottlenecks in training runs, and `checkpoint` manages model checkpoints using Pareto dominance to reclaim disk space intelligently.

---

### `/turing:profile`: Training Run Profiler

Profile a training run to identify performance bottlenecks. Measures timing breakdown, memory usage, throughput, and provides actionable recommendations for the detected bottleneck.

**Syntax:** `/turing:profile [exp-id] [--seed 42]`

- Defaults to the best experiment configuration if no ID is provided.
- `--seed 42` sets the random seed for the profiling run.

The profiler reports:
- **Timing:** total time, training time, and overhead breakdown.
- **Memory:** peak RSS, Python peak, and GPU peak (if applicable).
- **Throughput:** samples per second.
- **Bottleneck:** identified bottleneck type and severity.
- **Recommendations:** actionable fixes targeting the detected bottleneck.

**Examples:**

```
/turing:profile
# Profile the best experiment configuration

/turing:profile exp-042
# Profile a specific experiment's configuration
```

!!! tip
    Profile before running a large sweep. If a single training run takes 10 minutes but profiling shows 8 minutes is I/O overhead, fixing the data pipeline before the sweep saves hours. Results are saved to `experiments/profiles/`.

---

### `/turing:checkpoint`: Checkpoint Management

Smart checkpoint management using Pareto dominance. List all checkpoints, prune dominated ones to save disk, average top-K weights, resume from any point, and view disk usage statistics.

A checkpoint is **dominated** if another checkpoint is better on every metric. Pruning dominated checkpoints reclaims disk space without losing any Pareto-optimal model.

**Syntax:** `/turing:checkpoint <action> [exp-id] [--top N] [--dry-run]`

Actions:
- `list`: table of all checkpoints with metrics, size, and Pareto status.
- `prune`: remove dominated checkpoints, report space saved.
- `average`: list top-K checkpoints for weight averaging.
- `resume`: locate the checkpoint for a specific experiment.
- `stats`: disk usage summary by total, average, and model type.

**Examples:**

```
/turing:checkpoint list
# Show all checkpoints with metrics and Pareto status

/turing:checkpoint stats
# Disk usage summary

/turing:checkpoint prune --dry-run
# Preview which checkpoints would be pruned (without deleting)

/turing:checkpoint prune
# Remove dominated checkpoints and report space saved

/turing:checkpoint average --top 5
# Identify top 5 checkpoints for weight averaging

/turing:checkpoint resume exp-042
# Locate the checkpoint file for a specific experiment
```

!!! tip
    Run `/turing:checkpoint prune --dry-run` first to see what would be removed. Pareto-based pruning is conservative; it only removes checkpoints that are strictly worse than another on every metric. Previewing is always safer.
