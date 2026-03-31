---
name: seed
description: Run multi-seed study on an experiment to compute mean/std/CI and flag seed-sensitive results. Prevents publishing lucky seeds.
disable-model-invocation: true
argument-hint: "[N] [--quick] [--exp-id <id>]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Run a multi-seed study to verify that experiment results are robust across random seeds.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - A bare number (e.g., `5`) sets the seed count
   - `--quick` runs 3 seeds instead of 5
   - `--exp-id exp-042` targets a specific experiment (defaults to best)
   - `--seed-list 42,123,456` uses specific seed values

3. **Run seed study:**
   ```bash
   python scripts/seed_runner.py $ARGUMENTS
   ```

4. **Report results:**
   - Show the per-seed results table
   - Show mean +/- std with 95% CI
   - **STABLE (CV < 5%):** result is robust, safe to report
   - **SEED-SENSITIVE (CV >= 5%):** result varies too much across seeds — do not report single-seed numbers
   - If seed-sensitive, recommend reporting as mean +/- std over N seeds

5. **Saved output:** results are written to `experiments/seed_studies/exp-NNN-seeds.yaml`

6. **If no training pipeline exists:** suggest `/turing:init` first.

## Examples

```
/turing:seed              # 5 seeds on best experiment
/turing:seed --quick      # 3 seeds for fast check
/turing:seed 10           # 10 seeds for thorough study
/turing:seed --exp-id exp-042   # Specific experiment
```
