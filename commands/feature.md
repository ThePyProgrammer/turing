---
name: feature
description: Automated feature selection — multi-method importance consensus, redundancy detection, and interaction feature generation.
disable-model-invocation: true
argument-hint: "[--method all|importance] [--top-k 20]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Systematically evaluate which features matter and which are noise.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - `--method all|importance|selection|generation` — analysis type (default: all)
   - `--top-k 20` — number of top features to consider
   - `--json` — raw JSON output

3. **Run feature analysis:**
   ```bash
   python scripts/feature_intelligence.py $ARGUMENTS
   ```

4. **Report includes:**
   - Consensus ranking: features ranked by number of methods placing them in top-K
   - Per-method ranks: mutual information, L1, tree-based
   - Redundant pairs: features with |r| > 0.95
   - Candidate interaction features from top consensus set
   - Drop recommendation for zero-consensus features

5. **Saved output:** report in `experiments/features/features-*.yaml`

## Examples

```
/turing:feature                      # Full analysis
/turing:feature --top-k 10           # Top-10 consensus
```
