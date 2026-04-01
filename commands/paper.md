---
name: paper
description: Draft mechanical paper sections (setup, results, ablation, hyperparameters) from experiment logs. LaTeX and markdown output.
disable-model-invocation: true
argument-hint: "[--sections setup,results,ablation] [--format latex|markdown]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Draft paper sections directly from experiment data.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - `--sections setup,results,ablation,hyperparameters` — which sections to draft (default: all)
   - `--format latex|markdown` — output format (default: latex)

3. **Run paper drafting:**
   ```bash
   python scripts/draft_paper_sections.py $ARGUMENTS
   ```

4. **Report results:**
   - **setup:** Experimental setup prose (dataset, metrics, split, seed methodology)
   - **results:** Comparison table with all model types, best bolded, seed study stats
   - **ablation:** Ablation table from `/turing:ablate` results
   - **hyperparameters:** Appendix-style parameter table per model

5. **Output:** Each section saved to `paper/sections/` as `.tex` or `.md`

6. **Numbers are pulled directly from experiment logs** — no manual transcription needed.

## Examples

```
/turing:paper                                        # All sections, LaTeX
/turing:paper --format markdown                      # All sections, markdown
/turing:paper --sections setup,results               # Just setup + results
/turing:paper --sections ablation --format latex      # Just ablation table
```
