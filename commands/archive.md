---
name: archive
description: Experiment lifecycle cleanup — compress old artifacts, prune checkpoints, create queryable summary index. Reclaim disk space.
argument-hint: "[--older-than 30d] [--keep-best 10] [--dry-run]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Keep your project directory manageable after 200+ experiments.

## Steps
1. **Sync environment:** `uv sync`
2. **Run:** `uv run python scripts/experiment_archive.py $ARGUMENTS`
3. **Protected experiments:** Pareto-optimal, current best, recent, top-N by metric
4. **Report:** archived count, preserved count, space reclaimed
5. **Saved output:** `experiments/archive/index.yaml`

## Examples
```
/turing:archive --dry-run                    # Preview what would be archived
/turing:archive --older-than 30 --keep-best 10  # Archive old, keep top 10
/turing:archive                              # Default: 30 days, keep 10
```
