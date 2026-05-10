---
name: sanity
description: Pre-training sanity checks — catch broken data loaders, misconfigured losses, and dead gradients in 30 seconds before wasting hours.
disable-model-invocation: true
argument-hint: "[--quick] [--verbose]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Run a battery of fast checks before committing to a full training run. Catches wiring bugs in seconds.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - `--quick` — skip single-batch overfit test (fastest, ~5 seconds)
   - `--verbose` — show detailed check output
   - `--json` — raw JSON output

3. **Run sanity checks:**
   ```bash
   python scripts/sanity_checks.py $ARGUMENTS
   ```

4. **Checks performed:**
   - **Data pipeline** (critical): first batch loads, shapes match, no NaN/Inf
   - **Initial loss** (high): loss at initialization matches theory (e.g., -log(1/C) for cross-entropy)
   - **Gradient flow** (high): all parameters have non-zero, non-exploding gradients
   - **Single-batch overfit** (critical): model can memorize 1 batch in 50 steps — if not, something is broken
   - **Output validation** (high): predictions are non-NaN, non-constant, reasonable range
   - **Config consistency** (medium): learning rate, batch size in reasonable ranges

5. **Verdicts:**
   - **PASS** — safe to proceed
   - **PASS (with warnings)** — review before training
   - **FAIL** — do not proceed, fix issues first

6. **Saved output:** report in `experiments/sanity/sanity-*.yaml`

## Examples

```
/turing:sanity                    # Full check (~30 seconds)
/turing:sanity --quick            # Skip overfit test (~5 seconds)
```
