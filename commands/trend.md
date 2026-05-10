---
name: trend
description: Long-term trend analysis — improvement velocity, family ROI, diminishing returns detection, strategic research direction.
argument-hint: "[--window 30d] [--metric accuracy]"
allowed-tools: Read, Bash(*), Grep, Glob
---

See the arc of your research, not just the latest results. Strategic view over 100+ experiments.

## Steps
1. **Activate environment:** `source .venv/bin/activate`
2. **Run:** `python scripts/trend_analysis.py $ARGUMENTS`
3. **Report:** improvement velocity over time windows, family ROI ranking, diminishing returns prediction, phase transitions
4. **Saved output:** `experiments/trends/trend-*.yaml`

## Examples
```
/turing:trend                        # Full trend analysis
/turing:trend --window 14d           # Last 2 weeks
```
