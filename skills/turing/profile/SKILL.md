---
name: profile
description: Profile a training run — timing breakdown, memory usage, throughput, bottleneck detection with actionable recommendations.
disable-model-invocation: true
argument-hint: "[exp-id] [--seed 42]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Profile a training run to identify performance bottlenecks.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - First argument can be an experiment ID (e.g., `exp-042`); defaults to best
   - `--seed 42` sets the random seed for the profiling run

3. **Run profiling:**
   ```bash
   python scripts/profile_training.py $ARGUMENTS
   ```

4. **Report results:**
   - **Timing:** total time, training time, overhead breakdown
   - **Memory:** peak RSS, Python peak, GPU peak (if applicable)
   - **Throughput:** samples/sec
   - **Bottleneck:** identified bottleneck type and severity
   - **Recommendations:** actionable fixes for the detected bottleneck

5. **Saved output:** results written to `experiments/profiles/exp-NNN-profile.yaml`

6. **If no training pipeline exists:** suggest `/turing:init` first.

## Examples

```
/turing:profile              # Profile best experiment config
/turing:profile exp-042      # Profile specific experiment
```
