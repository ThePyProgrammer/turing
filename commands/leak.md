---
name: leak
description: Targeted leakage detection — probe for data leakage with single-feature tests, correlation checks, and train/test overlap detection.
disable-model-invocation: true
argument-hint: "[--deep] [--features feature_1,feature_2]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Actively probe for data leakage. The #1 cause of "too good to be true" results.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - `--deep` — run full single-feature analysis (slow but thorough)
   - `--features "feat_1,feat_2"` — check specific features
   - `--json` — raw JSON output

3. **Run leakage scan:**
   ```bash
   python scripts/leakage_detector.py $ARGUMENTS
   ```

4. **Checks performed:**
   - **Feature-target correlation:** flag features with >0.95 correlation to target
   - **Single-feature predictiveness (--deep):** train on each feature alone, flag any that achieve >80% of full model performance
   - **Train/test overlap:** hash-based deduplication across splits

5. **Verdicts:**
   - **CLEAN** — no leakage detected
   - **SUSPICIOUS** — warnings to review
   - **LEAKAGE DETECTED** — critical flags found

6. **Integration:** satisfies the "data leakage" check in `/turing:audit`

7. **Saved output:** report in `experiments/leakage/leak-*.yaml`

## Examples

```
/turing:leak                    # Standard correlation + overlap checks
/turing:leak --deep             # Full single-feature analysis
```
