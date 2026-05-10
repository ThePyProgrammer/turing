---
name: ablate
description: Run systematic ablation study — remove components one at a time, measure impact, produce publication-ready table with dead-weight flagging.
argument-hint: "[exp-id] [--components \"X,Y\"] [--seeds 3] [--latex]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Run a systematic ablation study to measure the contribution of each model component.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - First argument can be an experiment ID (e.g., `exp-042`); defaults to best
   - `--components "dropout,feature_X,regularization"` specifies components to ablate
   - `--seeds 3` runs each ablation 3 times for statistical robustness (uses seed runner)
   - `--latex` outputs a LaTeX-formatted table instead of markdown

3. **Run ablation study:**
   ```bash
   python scripts/ablation_study.py $ARGUMENTS
   ```

4. **Report results:**
   - Show the ablation table: Configuration | Metric | Δ from Full | % Change
   - Rank by impact (largest Δ first)
   - Flag **dead-weight** components (removing them improves the metric)
   - If `--latex`, output ready for copy-paste into a paper

5. **Saved output:** results written to `experiments/ablations/exp-NNN-ablation.yaml`

6. **If no ablatable components detected:** suggest using `--components` explicitly.

## Examples

```
/turing:ablate                                    # Auto-detect components
/turing:ablate exp-042                            # Specific experiment
/turing:ablate --components "dropout,subsample"   # Specific components
/turing:ablate --seeds 3                          # Multi-seed for robustness
/turing:ablate --latex                            # LaTeX table output
```
