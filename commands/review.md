---
name: review
description: Peer review simulation — generate likely reviewer objections with severity ratings and fix commands.
argument-hint: "[--venue neurips|icml|general] [--harsh]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Simulate a conference reviewer before you submit. Each weakness links to the command that fixes it.

## Steps
1. `source .venv/bin/activate`
2. `python scripts/simulate_review.py $ARGUMENTS`
3. **Saved:** `experiments/reviews/`

## Examples
```
/turing:review
/turing:review --venue neurips --harsh
```
