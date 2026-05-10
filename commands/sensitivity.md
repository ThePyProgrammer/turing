---
name: sensitivity
description: Hyperparameter sensitivity analysis — rank parameters by impact, identify which matter and which are noise.
argument-hint: "[exp-id] [--params learning_rate,max_depth]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Which hyperparameters actually matter? Stop wasting time on the ones that don't.

## Steps

1. **Sync environment:**
   ```bash
   uv sync
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - Optional experiment ID
   - `--params "learning_rate,max_depth"` — specific parameters to analyze
   - `--json` — raw JSON output

3. **Run sensitivity analysis:**
   ```bash
   uv run python scripts/sensitivity_analysis.py $ARGUMENTS
   ```

4. **Report includes:**
   - Per-parameter sensitivity ranking: HIGH / MED / LOW / NONE
   - Metric range for each parameter sweep
   - Monotonicity detection (is there a sweet spot?)
   - Recommendations: focus tuning on X, stop tuning Y

5. **Saved output:** report in `experiments/sensitivity/<exp-id>-sensitivity.yaml`

## Examples

```
/turing:sensitivity exp-042                           # All tunable params
/turing:sensitivity --params "learning_rate,max_depth" # Specific params
```
