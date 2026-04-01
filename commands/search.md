---
name: search
description: Natural language experiment search — query with text + structured filters over 200+ experiments.
disable-model-invocation: true
argument-hint: "<query> [--filter \"accuracy>0.85\"] [--limit 10]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Find specific experiments in a large history with natural language and structured filters.

## Steps
1. **Activate environment:** `source .venv/bin/activate`
2. **Run:** `python scripts/experiment_search.py $ARGUMENTS`
3. **Filters:** `accuracy>0.85`, `status:kept`, `family:baseline`, `date:last-week`
4. **Report:** ranked table of matching experiments

## Examples
```
/turing:search "LightGBM high accuracy" --filter "accuracy>0.85"
/turing:search "failed neural net" --filter "status:discarded"
/turing:search "last week" --limit 5
```
