---
name: diff
description: Deep experiment comparison — config diffs, metric significance, per-class regressions, training curve divergence, feature importance shifts.
argument-hint: "<exp-a> <exp-b> [--code]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Deep diagnostic comparison of two experiments. Goes beyond "which metric is higher" to show where, when, and why two experiments diverge.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - First two arguments are experiment IDs (required), e.g. `exp-042 exp-053`
   - `--code` includes git diff of train.py between the two experiments' commits
   - `--json` outputs raw JSON instead of markdown

3. **Run deep comparison:**
   ```bash
   python scripts/experiment_diff.py $ARGUMENTS
   ```

4. **Report results — the diff includes:**
   - **Config diff:** which hyperparameters changed, with magnitude (e.g., `max_depth: 6 → 8 (+33%)`)
   - **Metric diff:** all metrics with deltas and statistical significance (if seed studies exist)
   - **Per-class diff:** which classes improved/regressed — flags regressions hidden by aggregate improvement
   - **Training curve divergence:** the epoch where the two experiments' loss/metric curves separate
   - **Feature importance shifts:** which features gained/lost importance
   - **Code diff (--code):** git diff of train.py between the two commits

5. **Saved output:** report written to `experiments/diffs/<exp-a>-vs-<exp-b>.yaml`

6. **If experiment ID not found:** list available experiment IDs from `experiments/log.jsonl`

7. **If no training pipeline exists:** suggest `/turing:init` first.

## Examples

```
/turing:diff exp-042 exp-053                # Full diagnostic comparison
/turing:diff exp-042 exp-053 --code         # Include train.py code changes
/turing:diff exp-001 exp-010 --json         # Raw JSON output
```
