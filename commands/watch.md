---
name: watch
description: Live training monitor with early-warning alerts for loss spikes, NaN, overfitting, and metric plateaus.
disable-model-invocation: true
argument-hint: "[--alerts] [--interval 10] [--analyze run.log]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Stream metrics during training with early-warning alerts. Catches problems mid-run instead of at the end.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - `--analyze run.log` — post-hoc analysis of a completed log (non-blocking)
   - `--alerts` — show only alert lines, suppress normal output
   - `--interval 10` — check interval in seconds (default: 10)
   - `--alerts-config config/watch_alerts.yaml` — custom alert rules
   - `--json` — raw JSON output (for `--analyze` mode)

3. **For post-hoc analysis:**
   ```bash
   python scripts/training_monitor.py --analyze run.log
   ```

4. **For live monitoring (inform user):**
   Live monitoring requires a running training process. Suggest the user run in a separate terminal:
   ```bash
   python scripts/training_monitor.py --log run.log --interval 10
   ```

5. **Alert types:**
   - **Loss spike:** loss > 3x rolling mean (configurable multiplier)
   - **NaN detected:** any metric is NaN — CRITICAL, suggests pausing
   - **Overfitting onset:** train/val gap widening for 3+ consecutive epochs
   - **Plateau:** metric improvement < 0.001 for 5+ consecutive epochs

6. **Dashboard line format:**
   ```
   Epoch 23/100 | loss: 0.342 ↓ | acc: 0.865 ↑ | gap: 0.018 | ⚠ plateau
   ```

7. **Alert config:** rules are in `config/watch_alerts.yaml` — users can customize thresholds.

8. **Saved output:** analysis report written to `experiments/monitors/analysis-*.yaml`

9. **If no training log exists:** suggest running `/turing:train` first.

## Examples

```
/turing:watch --analyze run.log           # Analyze completed training
/turing:watch --analyze run.log --json    # JSON output for scripting
/turing:watch --alerts                    # Live: show only alerts
/turing:watch --interval 5               # Live: check every 5 seconds
```
