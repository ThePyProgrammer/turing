---
name: budget
description: Compute budget manager — set experiment/time limits, track allocation across explore/exploit phases, auto-shift modes, hard stop.
argument-hint: "<set|status|reset> [--experiments 50] [--hours 8]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Set a compute ceiling and let the system optimize within it. Prevents runaway experiment loops.

## Steps

1. **Sync environment:**
   ```bash
   uv sync
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - First argument is action: `set`, `status`, `reset`, or `check`
   - `--experiments 50` — max experiment count
   - `--hours 8` — max wall-clock hours
   - `--json` — raw JSON output

3. **Run budget manager:**
   ```bash
   uv run python scripts/budget_manager.py $ARGUMENTS
   ```

4. **Actions:**
   - **set:** create a budget with experiment and/or time constraints
   - **status:** show usage, burn rate, projected exhaustion, allocation breakdown
   - **reset:** deactivate the current budget
   - **check:** returns whether another experiment is allowed (used by `/turing:train`)

5. **Budget allocation policy:**
   - **0-50% budget:** EXPLORE — try diverse hypotheses
   - **50-80% budget:** MIXED — explore promising, exploit best
   - **80-100% budget:** EXPLOIT ONLY — refine the winner
   - **100% budget:** HARD STOP — `/turing:train` refuses new experiments

6. **Budget state** stored in `experiment_state.yaml` under the `budget` key.

7. **If no budget exists:** `/turing:train` runs without limits.

## Examples

```
/turing:budget set --experiments 50 --hours 8   # Set both constraints
/turing:budget set --experiments 30             # Experiment count only
/turing:budget status                           # Show usage and projections
/turing:budget reset                            # Remove budget limits
```
