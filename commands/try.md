---
name: try
description: Inject a hypothesis into the agent's experiment queue. This is how research taste reaches the agent — the human selects which coins to flip, the agent flips them.
disable-model-invocation: true
argument-hint: "<hypothesis description>"
allowed-tools: Read, Write, Edit, Bash(python scripts/*:*, source .venv/bin/activate:*), Grep, Glob
---

Inject a human hypothesis into the experiment queue for the next `/turing:train` iteration.

This is the taste-leverage mechanism: you provide judgment about what's worth trying, the agent provides disciplined execution.

## Steps

1. **Parse the hypothesis** from `$ARGUMENTS`. If empty, ask the user what they want the agent to try.

2. **Add to the queue:**
   ```bash
   source .venv/bin/activate && python scripts/manage_hypotheses.py add "$ARGUMENTS" --priority high --source human
   ```

3. **Confirm** with the hypothesis ID and instructions:
   - "Queued as hyp-NNN (high priority, human-injected)"
   - "The agent will prioritize this on the next `/turing:train` iteration"
   - Show current queue: `python scripts/manage_hypotheses.py list --status queued`

## Examples

```
/turing:try switch to LightGBM with dart boosting and lower learning rate
/turing:try add polynomial features for the numeric columns
/turing:try try a random forest ensemble — I think the data has too many interactions for linear boosting
/turing:try increase regularization, the train/val gap suggests overfitting
```

## How It Connects

The `/turing:train` loop checks `hypotheses.yaml` during the OBSERVE step. Human-injected hypotheses (high priority) are tried before the agent generates its own. After testing, the hypothesis is marked as `tested`, `promising`, or `dead-end` with a link to the resulting experiment.
