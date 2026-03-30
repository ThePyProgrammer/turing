# {{PROJECT_NAME}} ML Pipeline

{{TASK_DESCRIPTION}}

## Overview

This pipeline uses the [Helios](https://github.com/ThePyProgrammer/helios) autoresearch pattern — an AI agent iteratively trains, evaluates, and improves models by modifying `train.py` while the evaluation infrastructure (`prepare.py`, `evaluate.py`) remains immutable.

**Primary metric:** {{TARGET_METRIC}} ({{METRIC_DIRECTION}} is better)

## The Separation

| Layer | Files | Agent Access | Purpose |
|-------|-------|-------------|---------|
| Measurement | `prepare.py`, `evaluate.py` | READ-ONLY | Ensures all experiments are measured by the same yardstick |
| Hypothesis | `train.py`, `config.yaml` | READ-WRITE | All experimental changes go here |

This separation is the invariant that makes experiment comparisons valid.

## Quick Start

```bash
# 1. Set up the environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your training data to {{DATA_SOURCE}}

# 3. Create train/val/test splits
python prepare.py

# 4. Run training
python train.py > run.log 2>&1

# 5. Check results
grep -A 10 "^---" run.log

# 6. View experiment history
python scripts/show_metrics.py
```

## Using the Autoresearch Agent

The agent follows `program.md`. It:

1. Reads recent experiment results
2. Proposes a hypothesis
3. Modifies `train.py` or `config.yaml`
4. Runs training and evaluates
5. Keeps improvements, discards regressions
6. Repeats until convergence

To start: `/helios:train` in Claude Code.
For hands-off mode: `/loop 5m /helios:train`

## Directory Structure

```
{{ML_DIR}}/
  prepare.py              READ-ONLY: Data loading, splitting
  evaluate.py             READ-ONLY: Evaluation harness
  train.py                AGENT-EDITABLE: Training code
  config.yaml             Hyperparameters and settings
  sweep_config.yaml       Sweep parameter ranges
  program.md              Agent protocol instructions
  features/
    featurizers.py         Feature engineering pipeline
  scripts/
    log_experiment.py      Experiment JSONL logging
    show_metrics.py        Display experiment metrics
    compare_runs.py        Side-by-side comparison
    sweep.py               Hyperparameter sweep tool
    post-train-hook.sh     Auto-log after training
    stop-hook.sh           Convergence detection hook
  experiments/
    log.jsonl              Structured experiment log
    results.tsv            Quick-reference summary
  models/
    best/                  Current best model
    archive/               Previous best models
  data/
    splits/                Train/val/test splits
  tests/
    conftest.py            Shared test fixtures
```

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```
