---
name: sweep
description: Generate and run a systematic hyperparameter sweep. Computes the cartesian product of configured parameter ranges and processes the queue sequentially with full experiment logging.
argument-hint: "[sweep_config.yaml]"
allowed-tools: Read, Write, Edit, Bash(uv run python train.py:*, uv run python scripts/*:*, git:*, uv sync:*, uv add:*), Grep, Glob
---

Run a systematic hyperparameter sweep using the sweep configuration.

## Steps

1. **Sync environment:**
   ```bash
   uv sync
   ```

2. **Resolve config:** Use `$ARGUMENTS` as sweep config path, or default to `sweep_config.yaml`.

3. **Generate queue** (if not already generated):
   ```bash
   uv run python scripts/sweep.py [sweep_config.yaml]
   ```

4. **Check queue status:**
   ```bash
   uv run python scripts/sweep.py --status
   ```

5. **Process queue sequentially:**
   - Get next: `uv run python scripts/sweep.py --next`
   - Apply config overrides to `config.yaml`
   - Create experiment branch: `git checkout -b exp/NNN-description`
   - Run training: `uv run python train.py > run.log 2>&1`
   - Parse metrics: `grep -A 10 "^---" run.log | head -10`
   - Log the experiment
   - Mark complete: `uv run python scripts/sweep.py --mark <name> complete`
   - If improved, merge to main. If not, return to main.
   - Repeat until queue is empty

6. **Report** final results with best configuration found.

## Rules

Follow the same safety constraints as `/turing:train` — see `rules/loop-protocol.md`.
