---
name: plan
description: Research planning assistant — design a strategic experiment campaign with budget-aware ROI allocation.
argument-hint: "[--budget 20] [--goal \"maximize F1 for production\"]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Design the next N experiments strategically, not randomly. Allocates budget by expected ROI.

## Steps
1. `source .venv/bin/activate`
2. `python scripts/research_planner.py $ARGUMENTS`
3. **Saved:** `experiments/plans/`

## How it works
- Analyzes experiment history to compute per-family ROI
- Adjusts strategy priorities based on project state and goal
- Allocates budget across: feature engineering, model search, ensemble, calibration, verification
- Generates phased plan with specific experiment descriptions

## Examples
```
/turing:plan --budget 20
/turing:plan --budget 10 --goal "maximize F1 for production deployment"
/turing:plan --budget 30 --json
```
