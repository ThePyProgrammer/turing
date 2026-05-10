---
name: baseline
description: Automatic baseline generation — random, majority/mean, linear, k-NN baselines in 60 seconds. Every experiment needs a "is this better than dumb?" reference.
disable-model-invocation: true
argument-hint: "[--methods all|simple|linear] [--data data.npz]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Generate trivial baselines so you always know if your model is meaningfully better than simple approaches.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - `--methods all|simple|linear` — baseline group (default: all)
   - `--data data.npz` — data file with X and y arrays
   - `--json` — raw JSON output

3. **Run baseline generation:**
   ```bash
   python scripts/generate_baselines.py $ARGUMENTS
   ```

4. **Baselines generated:**
   - **Classification:** Random, Majority class, Stratified random, Logistic Regression, k-NN
   - **Regression:** Random, Mean predictor, Median predictor, Ridge Regression, k-NN
   - Each evaluated with the same protocol as real experiments

5. **Report includes:** comparison table with metric values and notes (floor, ceiling, reference)

6. **Integration:** satisfies the "baseline comparison" check in `/turing:audit`

7. **Saved output:** report in `experiments/baselines/baselines-*.yaml`

## Examples

```
/turing:baseline                           # All baselines
/turing:baseline --methods simple          # Just random + majority
/turing:baseline --data data/processed.npz # With actual data
```
