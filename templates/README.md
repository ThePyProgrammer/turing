# {{PROJECT_NAME}} ML Pipeline

{{TASK_DESCRIPTION}}

## Overview

This pipeline uses the Helios autoresearch pattern (inspired by karpathy/autoresearch) -- an AI agent iteratively trains, evaluates, and improves models by modifying `train.py`.

**Primary metric:** {{TARGET_METRIC}} ({{METRIC_DIRECTION}} is better)

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

## Directory Structure

```
{{ML_DIR}}/
  README.md               # This file
  program.md              # Autoresearch agent instructions
  config.yaml             # Default experiment configuration
  sweep_config.yaml       # Hyperparameter sweep ranges
  requirements.txt        # Python dependencies
  pyproject.toml          # Project config (pytest settings)
  prepare.py              # READ-ONLY: Data loading, splitting
  evaluate.py             # READ-ONLY: Evaluation harness
  train.py                # AGENT-EDITABLE: Training code
  data/
    splits/
      train.jsonl          # Training split
      val.jsonl            # Validation split
      test.jsonl           # Test split
  experiments/
    log.jsonl              # Structured experiment log (append-only)
    results.tsv            # Quick-reference TSV summary
  models/
    best/
      model.joblib         # Best trained model
      metadata.json        # Best model metadata
    archive/               # Previous best models
  features/
    featurizers.py         # Feature engineering strategies
  scripts/
    log_experiment.py      # Experiment JSONL logging utility
    show_metrics.py        # Display experiment metrics table
    compare_runs.py        # Side-by-side experiment comparison
    sweep.py               # Hyperparameter sweep tool
    post-train-hook.sh     # Auto-log after training (Claude Code hook)
    stop-hook.sh           # Convergence detection hook
  tests/
    conftest.py            # Shared test fixtures
```

## Using the Autoresearch Agent

The autoresearch agent follows instructions in `program.md`. It:

1. Reads recent experiment results
2. Proposes a hypothesis (new model, hyperparams, features)
3. Modifies `train.py` with the experiment
4. Runs training and evaluates
5. Keeps improvements, discards regressions
6. Repeats until convergence

To start: run `/helios:train` in Claude Code.

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```
