---
name: postmortem
description: Failure postmortem — diagnose why experiments stopped improving and get actionable next steps.
argument-hint: "[--window 10] [--auto-trigger 5]"
allowed-tools: Read, Bash(*), Grep, Glob
---

When experiments stop improving, find out why. Diagnoses search space exhaustion, config errors, data issues, metric ceilings, and noise floors.

## Steps
1. `uv sync`
2. `uv run python scripts/failure_postmortem.py $ARGUMENTS`
3. **Saved:** `experiments/postmortems/`

## Diagnosis categories
- **Search space exhaustion:** micro-tuning params that don't matter
- **Systematic config error:** all experiments share a bad common config
- **Data issue:** all model types fail similarly
- **Metric ceiling:** near theoretical maximum
- **Noise floor:** improvements within seed variance

## Examples
```
/turing:postmortem
/turing:postmortem --window 15
/turing:postmortem --json
```
