---
name: calibrate
description: Probability calibration — measure ECE, plot reliability diagrams, apply Platt scaling or isotonic regression.
disable-model-invocation: true
argument-hint: "[exp-id] [--method platt|isotonic|temperature|auto]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Make model probabilities trustworthy. Does 80% confidence actually mean 80% correct?

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - Optional experiment ID
   - `--method platt|isotonic|temperature|auto` — calibration method (default: auto)
   - `--json` — raw JSON output

3. **Run calibration:**
   ```bash
   python scripts/calibration.py $ARGUMENTS
   ```

4. **Report includes:**
   - ECE/MCE before calibration
   - Reliability diagram (predicted vs actual per bin)
   - Calibration method comparison table
   - Verdict: ALREADY CALIBRATED / IMPROVED / NO IMPROVEMENT

5. **Methods:**
   - **Platt:** logistic regression on logits
   - **Isotonic:** non-parametric (more flexible, needs more data)
   - **Temperature:** single scalar T parameter
   - **Auto:** tries all, picks lowest ECE

6. **Saved output:** report in `experiments/calibration/<exp-id>-calibration.yaml`

## Examples

```
/turing:calibrate exp-042                  # Auto-select best method
/turing:calibrate exp-042 --method platt   # Platt scaling only
```
