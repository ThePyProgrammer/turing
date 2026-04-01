---
name: flashback
description: Session context restoration — "where was I?" summary after days away. Current best, pending hypotheses, last session, annotations.
disable-model-invocation: true
argument-hint: "[--days 7] [--last 10]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Come back to a project after a week and start working in 10 seconds instead of 30 minutes.

## Steps
1. **Activate environment:** `source .venv/bin/activate`
2. **Run:** `python scripts/session_flashback.py $ARGUMENTS`
3. **Report:** current best, last session experiments, pending hypotheses, annotations, budget, suggested next action
4. **Saved output:** `experiments/flashbacks/flashback-*.yaml`

## Examples
```
/turing:flashback                    # Default: last 7 days
/turing:flashback --days 14          # 2-week lookback
/turing:flashback --last 5           # Last 5 experiments
```
