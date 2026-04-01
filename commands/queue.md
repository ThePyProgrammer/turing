---
name: queue
description: Queue experiments for batch execution with priority ordering and dependency chains. Load the queue, walk away, read the summary.
disable-model-invocation: true
argument-hint: "<add|list|run|pause|clear> [description] [--priority high] [--after q-001]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Manage the experiment queue for unattended batch execution.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - **add** `"description"` `--priority high` `--after q-001` — queue an experiment
   - **list** — show queue with status, priority, dependencies
   - **run** `--halt-on-error` — execute all queued experiments
   - **pause** — stop after current experiment finishes
   - **clear** — discard all queued items

3. **Run queue manager:**
   ```bash
   python scripts/experiment_queue.py $ARGUMENTS
   ```

4. **Report results by action:**
   - **add:** confirms ID and priority
   - **list:** table of queued/completed/failed items
   - **run:** batch summary with per-experiment status
   - **pause/clear:** confirmation message

5. **Queue persists in** `experiments/queue.yaml`

## Examples

```
/turing:queue add "try LightGBM" --priority high
/turing:queue add "deeper trees" --after q-001
/turing:queue list
/turing:queue run
/turing:queue run --halt-on-error
/turing:queue pause
/turing:queue clear
```
