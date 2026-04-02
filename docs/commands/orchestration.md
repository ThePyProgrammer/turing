---
title: "Experiment Orchestration"
description: "Queue experiments for batch execution, recover from failures automatically, and fork experiments into parallel branches."
---

# Experiment Orchestration

Commands for managing experiment execution at scale: queuing batch runs, recovering from crashes, and branching experiments into competing approaches.

---

### `/turing:queue`: Batch experiment queue

Manage the experiment queue for unattended batch execution. Load the queue, walk away, read the summary. Experiments can be prioritized and chained with dependency ordering so that one experiment waits for another to finish before starting.

**Syntax:** `/turing:queue <add|list|run|pause|clear> [description] [--priority high] [--after q-001]`

**Examples:**
```
/turing:queue add "try LightGBM" --priority high
/turing:queue add "deeper trees" --after q-001
/turing:queue list
/turing:queue run
/turing:queue run --halt-on-error
/turing:queue pause
/turing:queue clear
```

---

### `/turing:retry`: Smart failure recovery

Auto-diagnose crash type and retry with a targeted fix. OOM gets halved batch size. NaN gets gradient clipping. The system classifies the failure, applies the appropriate remedy, and retries up to a configurable limit. If all attempts are exhausted or the failure requires human intervention, the report says so.

**Syntax:** `/turing:retry <exp-id> [--max-attempts 3]`

**Examples:**
```
/turing:retry exp-042                    # Auto-diagnose and retry
/turing:retry exp-042 --max-attempts 5   # More retries
```

---

### `/turing:fork`: Branch into parallel tracks

Branch an experiment into parallel tracks: run both A and B, then report the winner. Each branch inherits the parent experiment's configuration and diverges on the specified approach. With `--auto-promote`, the winning branch is automatically kept without manual intervention.

**Syntax:** `/turing:fork <exp-id> --branches "approach A" "approach B" [--auto-promote]`

**Examples:**
```
/turing:fork exp-042 --branches "LightGBM with dart" "XGBoost deeper trees"
/turing:fork exp-042 --branches "A" "B" "C" --auto-promote
```
