---
name: replay
description: Experiment replay — re-run a historical experiment with current infrastructure to test if old approaches do better now.
argument-hint: "<exp-id> [--with-current-data] [--with-current-preprocessing]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Should you revisit old ideas? Infrastructure changes may make failed approaches work now.

## Steps
1. **Sync environment:** `uv sync`
2. **Run:** `uv run python scripts/experiment_replay.py $ARGUMENTS`
3. **Modes:** default (current code+data), --with-current-data, --with-current-preprocessing
4. **Report:** original vs replayed metrics, delta, verdict
5. **Saved output:** `experiments/replays/`

## Examples
```
/turing:replay exp-023                              # Replay with current infrastructure
/turing:replay exp-023 --with-current-data          # Current data, old code
/turing:replay --list                               # List replayable experiments
```
