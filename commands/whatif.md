---
name: whatif
description: What-if analysis — answer hypotheticals from existing experiment data without running new experiments.
disable-model-invocation: true
argument-hint: "\"<question>\" [--json]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Answer "what if?" questions using existing experiment data. Routes to the right estimator automatically.

## Steps
1. `source .venv/bin/activate`
2. `python scripts/whatif_engine.py $ARGUMENTS`
3. **Saved:** `experiments/whatif/`

## Supported question types
- **Data scaling:** "what if I had 2x more data" → scaling law extrapolation
- **Ablation:** "what if I removed class 3" → ablation study data
- **Pipeline stitch:** "what if I combined exp-031 with exp-042" → stitch estimation
- **Hyperparameters:** "what if learning_rate was 0.01" → sensitivity interpolation
- **Ensemble:** "what if I ensembled the top models" → correlation analysis
- **Pruning:** "what if I pruned to 50% sparsity" → pruning sweep interpolation
- **Budget:** "what if I spent my budget on X vs Y" → budget allocation

## Examples
```
/turing:whatif "what if I had 2x more data"
/turing:whatif "what if I removed class 3"
/turing:whatif "what if I combined exp-031 with exp-042"
/turing:whatif "what if learning_rate was 0.01" --json
```
