---
name: curriculum
description: Training curriculum optimization — order data by difficulty, compare easy-to-hard vs hard-to-easy vs self-paced strategies.
argument-hint: "[exp-id] [--strategies easy-to-hard,random]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Does the order your model sees data matter? Find out systematically.

## Steps

1. **Sync environment:**
   ```bash
   uv sync
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - Optional experiment ID
   - `--strategies "easy_to_hard,hard_to_easy,self_paced,random"` — strategies to test
   - `--json` — raw JSON output

3. **Run curriculum analysis:**
   ```bash
   uv run python scripts/curriculum_optimizer.py $ARGUMENTS
   ```

4. **Strategies tested:**
   - **Random:** standard shuffling (control)
   - **Easy-to-hard:** classic curriculum learning
   - **Hard-to-easy:** anti-curriculum
   - **Self-paced:** start easy, gradually include harder samples

5. **Report includes:** strategy comparison table with metric, convergence epoch, and speedup vs random; impossible sample detection (likely mislabeled)

6. **Saved output:** report in `experiments/curriculum/<exp-id>-curriculum.yaml`

## Examples

```
/turing:curriculum exp-042                      # All strategies
/turing:curriculum --strategies easy_to_hard,random  # Specific strategies
```
