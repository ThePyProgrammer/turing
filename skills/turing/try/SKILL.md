---
name: try
description: Inject a hypothesis into the agent's experiment queue. This is how research taste reaches the agent — the human selects which coins to flip, the agent flips them.
argument-hint: "<hypothesis description>"
allowed-tools: Read, Write, Edit, Bash(python scripts/*:*, source .venv/bin/activate:*), Grep, Glob
---

Inject a human hypothesis into the experiment queue for the next `/turing:train` iteration.

This is the taste-leverage mechanism: you provide judgment about what's worth trying, the agent provides disciplined execution.

## Steps

1. **Parse the hypothesis** from `$ARGUMENTS`. If empty, ask the user what they want the agent to try.

2. **Check for archetype syntax.** If the argument starts with `archetype:`, expand it:
   ```bash
   source .venv/bin/activate && python scripts/manage_hypotheses.py add --archetype <name> --priority high --source human
   ```

   Otherwise, use the raw description:
   ```bash
   source .venv/bin/activate && python scripts/manage_hypotheses.py add "$ARGUMENTS" --priority high --source human
   ```

3. **Confirm** with the hypothesis ID and instructions:
   - "Queued as hyp-NNN (high priority, human-injected)"
   - "The agent will prioritize this on the next `/turing:train` iteration"
   - Show current queue: `python scripts/manage_hypotheses.py list --status queued`

## Examples

```
# Free-text hypotheses
/turing:try switch to LightGBM with dart boosting and lower learning rate
/turing:try add polynomial features for the numeric columns
/turing:try increase regularization, the train/val gap suggests overfitting

# Archetype-based structured strategies
/turing:try archetype:model_comparison
/turing:try archetype:feature_sweep
/turing:try archetype:ensemble_construction
/turing:try archetype:regularization_search
/turing:try archetype:ablation_study
```

## Available Archetypes

| Archetype | What it does | Expected experiments |
|-----------|-------------|---------------------|
| `model_comparison` | Compare XGBoost, LightGBM, RF, LR, MLP with statistical tests | ~5 |
| `hyperparameter_sweep` | Grid search with multi-seed validation | 15-36 |
| `feature_sweep` | Add/remove feature transforms one at a time | 6-10 |
| `regularization_search` | Binary search for optimal regularization | 4-6 |
| `ensemble_construction` | Voting, stacking, blending of top models | 4-6 |
| `learning_rate_schedule` | lr vs n_estimators tradeoff | 4-5 |
| `data_quality_audit` | Class balance, label noise, leakage checks | 3-5 |
| `ablation_study` | Remove features one at a time to measure importance | N+1 |

## How It Connects

The `/turing:train` loop checks `hypotheses.yaml` during the OBSERVE step. Human-injected hypotheses (high priority) are tried before the agent generates its own. After testing, the hypothesis is marked as `tested`, `promising`, or `dead-end` with a link to the resulting experiment.
