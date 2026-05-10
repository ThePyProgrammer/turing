---
name: checkpoint
description: Smart checkpoint management — list, prune (Pareto-based), average top-K, resume from any point, disk usage stats.
argument-hint: "<list|prune|average|resume|stats> [exp-id] [--top 3] [--dry-run]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Manage model checkpoints intelligently using Pareto dominance.

## Steps

1. **Sync environment:**
   ```bash
   uv sync
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - First word is the action: `list`, `prune`, `average`, `resume`, `stats`
   - `resume` requires an experiment ID as second argument
   - `--top 3` sets the number of checkpoints for averaging
   - `--dry-run` previews pruning without deleting

3. **Run checkpoint manager:**
   ```bash
   uv run python scripts/checkpoint_manager.py $ARGUMENTS
   ```

4. **Report results by action:**
   - **list:** Table of all checkpoints with metrics, size, and Pareto status
   - **prune:** Removes dominated checkpoints, reports space saved
   - **average:** Lists top-K checkpoints for weight averaging
   - **resume:** Locates checkpoint for a specific experiment
   - **stats:** Disk usage summary by total, average, and model type

5. **Saved output:** report written to `experiments/checkpoints/checkpoint-report.yaml`

## Examples

```
/turing:checkpoint list              # Show all checkpoints
/turing:checkpoint stats             # Disk usage summary
/turing:checkpoint prune --dry-run   # Preview what would be pruned
/turing:checkpoint prune             # Remove dominated checkpoints
/turing:checkpoint average --top 5   # Top 5 for averaging
/turing:checkpoint resume exp-042    # Resume from checkpoint
```
