---
name: validate
description: Run stability validation on the current experiment configuration. Executes N runs to measure metric variance and auto-configures multi-run evaluation if variance is too high.
argument-hint: "[--auto]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Validate the stability of the current ML pipeline by running it multiple times and measuring variance.

## Steps

1. **Sync environment:**
   ```bash
   uv sync
   ```

2. **Run stability check:**
   ```bash
   uv run python scripts/validate_stability.py
   ```

3. **If `$ARGUMENTS` contains `--auto`:**
   ```bash
   uv run python scripts/validate_stability.py --auto
   ```
   This auto-writes `evaluation.n_runs: 3` to `config.yaml` if CV > 5%.

4. **Report results:**
   - **Stable (CV < 5%):** metric is reliable, single-run evaluation is sufficient
   - **Unstable (CV >= 5%):** metric has high variance, multi-run with median is recommended
   - If `--auto` was used, report what was changed in config.yaml

5. **If no training pipeline exists:** suggest `/turing:init` first.
