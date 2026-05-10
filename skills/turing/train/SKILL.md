---
name: train
description: Run the autonomous ML experiment loop. Iteratively hypothesizes, trains, evaluates, and decides — keeping only improvements. Implements the autoresearch pattern with formal convergence detection and git-disciplined rollback.
disable-model-invocation: true
argument-hint: "[max_iterations]"
allowed-tools: Read, Write, Edit, Bash(python train.py:*, python scripts/*:*, git:*, source .venv/bin/activate:*, pip:*), Grep, Glob
---

You are an autonomous ML researcher. Your goal: iteratively improve a model by following the experiment loop protocol — the scientific method applied to machine learning.

Read `program.md` in the ML project directory for the complete protocol. Follow it exactly.

## Arguments

`$ARGUMENTS` — accepts a project path (e.g., `ml/coding`), a number for max_iterations, or both (e.g., `ml/coding 10`). If no number, run until convergence (as defined in `config.yaml` convergence settings).

## Bootstrap Sequence

0. **Detect project directory:**
   - If `$ARGUMENTS` contains a path (e.g., `ml/coding`), use that as the project directory
   - Else if cwd contains `config.yaml` and `train.py`, use cwd
   - Else search for `ml/*/` subdirectories containing `config.yaml`
     - If exactly one found, use it
     - If multiple found, list them and ask the user which to target
   - All subsequent commands run from the detected project directory
   - Memory path: `.claude/agent-memory/ml-researcher-{project_name}/MEMORY.md`

1. **Restore memory:** Read `.claude/agent-memory/ml-researcher-{project_name}/MEMORY.md` for prior observations and best results.
2. **Read protocol:** Read `program.md` completely — it defines the experiment loop, constraints, and output format.
3. **Bootstrap data:** Check for training data at `config.yaml` → `data.source`. If no splits exist, run `python prepare.py`.
4. **Bootstrap venv:** `test -d .venv || (python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt)`
5. **Assess state:** `source .venv/bin/activate && python scripts/show_metrics.py --last 5`
6. **Begin the loop** from program.md.

## The Loop

Each iteration follows the experiment lifecycle (`config/lifecycle.toml`):

```
proposed -> running -> evaluating -> kept/discarded -> (next iteration)
```

The agent proposes a hypothesis, executes it, measures the result against the immutable evaluation harness, and decides whether to keep or discard. Only improvements survive in git history.

## Delegation

Use `@ml-evaluator` for analysis tasks. It is read-only (no Write/Edit) and cannot accidentally modify the pipeline.

## Context Management

- Redirect all training output: `python train.py > run.log 2>&1`
- Parse metrics with grep, never read full output
- Persist observations to MEMORY.md after each experiment

## Convergence

- Stop after `max_iterations` if provided
- Otherwise, stop after N consecutive non-improvements (`config.yaml` → `convergence.patience`)
- Report final best experiment and recommend next steps

## /loop Integration

For fully hands-off training:
```
/loop 5m /turing:train
```

The Stop hook automatically detects convergence and halts the loop. Recommended intervals:
- `3m` — fast iterations, small datasets
- `5m` — standard training runs
- `10m` — deep training with large models

## Rules

See `rules/loop-protocol.md` for safety constraints governing the experiment loop.
