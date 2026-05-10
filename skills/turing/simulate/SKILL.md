---
name: simulate
description: Experiment outcome prediction — predict which configs will beat the current best before running them.
argument-hint: "[--configs configs.yaml] [--top-k 5] [--threshold 0.001]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Predict outcomes before spending compute. Ranks proposed configs and recommends which to run vs skip.

## Steps
1. `uv sync`
2. `uv run python scripts/experiment_simulator.py $ARGUMENTS`
3. **Saved:** `experiments/simulations/`

## How it works
- Builds a surrogate model from experiment history (weighted k-NN)
- Predicts metric for each proposed config
- Applies novelty penalty for configs far from training distribution
- Ranks and filters: only recommend configs predicted to improve

## Examples
```
/turing:simulate --configs sweep_configs.yaml
/turing:simulate --configs candidates.yaml --top-k 3
/turing:simulate --configs proposals.yaml --threshold 0.005
/turing:simulate --configs sweep.yaml --json
```
