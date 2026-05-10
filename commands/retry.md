---
name: retry
description: Smart failure recovery — auto-diagnose crash type and retry with targeted fix. OOM → halve batch. NaN → add clipping.
argument-hint: "<exp-id> [--max-attempts 3]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Auto-diagnose and recover from experiment failures.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - First argument is the experiment ID (required)
   - `--max-attempts 3` limits retry count
   - `--classify "error text"` just classifies without retrying

3. **Run smart retry:**
   ```bash
   python scripts/smart_retry.py $ARGUMENTS
   ```

4. **Report results:**
   - **RECOVERED:** fix applied, retry succeeded
   - **FAILED:** all retry attempts exhausted
   - **MANUAL FIX NEEDED:** failure type requires human intervention
   - Shows failure classification, fix applied, and attempt history

5. **Saved output:** report written to `experiments/retries/exp-NNN-retry.yaml`

## Examples

```
/turing:retry exp-042                    # Auto-diagnose and retry
/turing:retry exp-042 --max-attempts 5   # More retries
```
