---
name: fork
description: Branch an experiment into parallel tracks — run both A and B, report the winner.
argument-hint: "<exp-id> --branches \"approach A\" \"approach B\" [--auto-promote]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Fork an experiment into parallel branches and compare results.

## Steps

1. **Sync environment:**
   ```bash
   uv sync
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - First argument is the parent experiment ID
   - `--branches "A" "B" "C"` — branch descriptions (2+ required)
   - `--auto-promote` — automatically keep the winning branch

3. **Run fork:**
   ```bash
   uv run python scripts/fork_experiment.py $ARGUMENTS
   ```

4. **Report results:**
   - Comparison tree showing each branch's metric
   - Winner identified and marked
   - Recommendation: promote winner, abandon rest

5. **Saved output:** report written to `experiments/forks/exp-NNN-fork.yaml`

## Examples

```
/turing:fork exp-042 --branches "LightGBM with dart" "XGBoost deeper trees"
/turing:fork exp-042 --branches "A" "B" "C" --auto-promote
```
