---
name: reproduce
description: Verify reproducibility of a specific experiment by re-running from logged config and checking metrics fall within tolerance.
argument-hint: "<exp-id> [--tolerance 0.02] [--strict] [--runs 3]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Verify that a logged experiment can be reproduced with consistent results.

## Steps

1. **Sync environment:**
   ```bash
   uv sync
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - First argument is the experiment ID (required), e.g. `exp-042`
   - `--tolerance 0.02` sets the relative tolerance (default 2%)
   - `--strict` requires exact float match (1e-6), overrides tolerance
   - `--runs 3` sets number of reproduction runs (default 3, 1 for strict)

3. **Run reproducibility verification:**
   ```bash
   uv run python scripts/reproduce_experiment.py $ARGUMENTS
   ```

4. **Report results:**
   - **reproducible:** metrics match exactly (deterministic algorithm)
   - **approximately_reproducible:** metrics within tolerance or original falls in 95% CI
   - **not_reproducible:** metrics outside tolerance and CI
   - **environment_changed:** metrics diverge AND library versions differ
   - Show environment diff if present (Python version, package versions)

5. **Saved output:** report written to `experiments/reproductions/exp-NNN-repro.yaml`

6. **If experiment ID not found:** list available experiment IDs from `experiments/log.jsonl`

7. **If no training pipeline exists:** suggest `/turing:init` first.

## Examples

```
/turing:reproduce exp-042                 # Default: 3 runs, 2% tolerance
/turing:reproduce exp-042 --strict        # Exact match required
/turing:reproduce exp-042 --tolerance 0.05 --runs 5   # Lenient, more runs
```
