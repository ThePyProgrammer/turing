---
name: share
description: Experiment packaging — portable archive with config, metrics, seed study, annotations, reproduction instructions.
argument-hint: "<exp-ids...> [--include model,figures,code]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Package experiments for collaborator handoff or paper supplementary material.

## Steps
1. `uv sync`
2. `uv run python scripts/package_experiments.py $ARGUMENTS`
3. **Saved:** `exports/packages/<name>/`

## Examples
```
/turing:share exp-089
/turing:share exp-042 exp-089 --include model,figures
```
