---
name: sweep
description: Generate and run a hyperparameter sweep across configured parameter ranges.
disable-model-invocation: true
argument-hint: "[sweep_config.yaml]"
---

Run a systematic hyperparameter sweep using the sweep configuration.

## Steps

1. Check for sweep config:
   ```bash
   source .venv/bin/activate
   ```

2. If `$ARGUMENTS` is provided, use it as the sweep config path. Otherwise default to `sweep_config.yaml`.

3. Generate the sweep queue (if not already generated):
   ```bash
   python scripts/sweep.py [sweep_config.yaml]
   ```

4. Check queue status:
   ```bash
   python scripts/sweep.py --status
   ```

5. Process the queue sequentially:
   - Get next pending experiment: `python scripts/sweep.py --next`
   - Apply config overrides to `config.yaml`
   - Create an experiment branch: `git checkout -b exp/NNN-description`
   - Run training: `python train.py > run.log 2>&1`
   - Parse metrics: `grep -A 10 "^---" run.log | head -10`
   - Log the experiment
   - Mark complete: `python scripts/sweep.py --mark <name> complete`
   - If improved, merge to main. If not, return to main.
   - Repeat until queue is empty

6. Report final results with best configuration found.

## Rules

Follow the same safety rules as `/helios:train` -- see `rules/loop-protocol.md`.
