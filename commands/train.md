---
name: train
description: Run autonomous ML training experiments. Iteratively trains, evaluates, and improves models with convergence detection and safety guardrails.
disable-model-invocation: true
argument-hint: "[max_iterations]"
---

You are an autonomous ML researcher. Your goal: iteratively train and improve a model by following the experiment loop protocol.

Read `program.md` in the ML project directory for the complete experiment loop protocol. Follow it exactly.

## Arguments

`$ARGUMENTS` -- if a number, use as max_iterations (stop after N experiments). If empty, run until convergence (as defined in config.yaml convergence settings).

## First Steps

0. Read `.claude/agent-memory/ml-researcher/MEMORY.md` for prior observations and best results.
1. Read `program.md` completely -- it defines the experiment loop, constraints, and output format.
2. Bootstrap data if missing:
   - Check for training data in the location specified by `config.yaml` `data.source`.
   - Check for splits in `config.yaml` `data.splits_dir` -- if no splits exist, run: `python prepare.py`
3. Bootstrap venv if missing: `test -d .venv || (python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt)`
4. Check current state: `source .venv/bin/activate && python scripts/show_metrics.py --last 5`
5. Begin the experiment loop from program.md.

## Delegation

Use the `@ml-evaluator` agent for analyzing results without risk of modifying code. The evaluator is read-only (no Write/Edit tools) and can safely run evaluation scripts.

## Context Management

- Redirect all training output to `run.log`: `python train.py > run.log 2>&1`
- Only read specific metric lines via grep -- never read full training output.
- Use agent memory to persist observations across sessions: current best metrics, failed approaches, promising directions.

## Convergence

Per the project convergence protocol:
- Stop after `max_iterations` if provided.
- Otherwise, stop after N consecutive non-improvements (as configured in `config.yaml` under `convergence.patience`).
- Report final best experiment and recommend next steps when stopping.

## /loop Integration

For fully hands-off training, use with `/loop`:
```
/loop 5m /helios:train
```

This runs the training skill every 5 minutes. The Stop hook automatically detects convergence and halts the loop.

Recommended intervals:
- `/loop 3m /helios:train` -- fast iterations, small datasets
- `/loop 5m /helios:train` -- standard training runs
- `/loop 10m /helios:train` -- deep training with large models

The agent reads MEMORY.md at each iteration start to maintain continuity across /loop cycles.

## Rules

See `rules/loop-protocol.md` for detailed safety rules governing the experiment loop.
