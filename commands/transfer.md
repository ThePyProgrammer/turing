---
name: transfer
description: Cross-project knowledge transfer — find similar prior projects and surface what worked. Builds institutional ML memory.
argument-hint: "[--from project-path] [--auto]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Find similar prior projects and surface what worked. "Last time you had tabular classification with class imbalance, LightGBM beat everything by 3%."

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - `--from ~/projects/fraud-detection` — transfer from a specific project
   - `--auto` — auto-queue hypotheses from recommendations
   - `--index ~/.turing/project_index.yaml` — custom index path
   - `--json` — raw JSON output

3. **Run knowledge transfer:**
   ```bash
   python scripts/knowledge_transfer.py $ARGUMENTS
   ```

4. **Report includes:**
   - Similar prior projects ranked by similarity score
   - Per project: task type, winner model, key insights
   - Suggested hypotheses from winning strategies
   - Auto-queued hypotheses (with `--auto`)

5. **Similarity matching** uses:
   - Task type (classification/regression) — highest weight
   - Dataset size (log-scale comparison)
   - Feature types (tabular/image/text)
   - Class balance characteristics
   - Dimensionality

6. **Project index** at `~/.turing/project_index.yaml` — local only, never uploaded

7. **If no similar projects found:** suggest running on more projects first or specifying one with `--from`

8. **Saved output:** report in `experiments/transfers/transfer-*.yaml`

## Examples

```
/turing:transfer                                    # Search index for similar projects
/turing:transfer --from ~/projects/fraud-detection  # Transfer from specific project
/turing:transfer --auto                             # Auto-queue hypotheses
```
