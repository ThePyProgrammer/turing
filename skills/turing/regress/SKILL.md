---
name: regress
description: Performance regression gate — re-run best experiment after code/dependency changes and verify metrics haven't degraded.
argument-hint: "[--tolerance 0.01] [--against exp-id] [--quick]"
allowed-tools: Read, Bash(*), Grep, Glob
---

CI for your model. After any change to code, dependencies, or data, verify metrics haven't silently regressed.

## Steps

1. **Sync environment:**
   ```bash
   uv sync
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - `--tolerance 0.01` sets the relative tolerance (default 1%)
   - `--against exp-042` checks against a specific experiment (default: best)
   - `--quick` runs 1 seed instead of 3 for fast checks
   - `--runs 5` sets number of regression runs (default 3)
   - `--json` outputs raw JSON

3. **Run regression gate:**
   ```bash
   uv run python scripts/regression_gate.py $ARGUMENTS
   ```

4. **Report results:**
   - **PASS:** all metrics within tolerance — no regression
   - **WARNING:** some metrics degraded within 2x tolerance — investigate
   - **FAIL:** REGRESSION DETECTED — at least one metric degraded beyond tolerance
   - Shows per-metric comparison with deltas and relative differences
   - Shows environment diff if library versions changed (may explain regression)

5. **Saved output:** report written to `experiments/regressions/check-YYYY-MM-DD.yaml`

6. **If no experiments exist:** suggest running `/turing:train` first.

7. **On FAIL verdict:** suggest investigating with:
   - `/turing:diff <baseline> <latest>` to see what changed
   - `pip freeze` comparison to identify library version changes
   - `git diff` to review code changes

## Examples

```
/turing:regress                              # Default: check best, 1% tolerance, 3 runs
/turing:regress --quick                      # Fast check: 1 run
/turing:regress --against exp-042            # Check specific experiment
/turing:regress --tolerance 0.005 --runs 5   # Strict: 0.5% tolerance, 5 runs
```
