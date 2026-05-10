---
name: lit
description: Literature search scoped to the current experiment domain — find papers, SOTA baselines, and related work without leaving the terminal.
argument-hint: "<query> | --baseline | --related <exp-id>"
allowed-tools: Read, Bash(*), Grep, Glob, WebSearch
---

Search the literature for papers, baselines, and related work.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - **Free query:** `"gradient boosting for tabular data"` — searches Semantic Scholar
   - **Baseline:** `--baseline` — finds SOTA results for the current task, compares against your best
   - **Related:** `--related exp-042` — finds papers using similar methods to a specific experiment
   - `--auto-queue` — auto-queues hypotheses from literature with `source: "literature"`
   - `--limit 10` — max number of results

3. **Run literature search:**
   ```bash
   python scripts/literature_search.py $ARGUMENTS
   ```

4. **Report results:**
   - **Papers:** title, authors, year, venue, citations, abstract snippet, URL
   - **Baseline mode:** SOTA comparison with gap analysis against current best
   - **Related mode:** methodological differences worth investigating
   - **Hypotheses:** if `--auto-queue`, shows queued experiments from findings

5. **Saved output:** results written to `experiments/literature/query-YYYY-MM-DD-HHMMSS.md`

6. **If API unavailable:** reports error and suggests manual search.

## Examples

```
/turing:lit "gradient boosting missing values"    # Free query
/turing:lit --baseline                             # SOTA comparison
/turing:lit --related exp-042                      # Related work
/turing:lit --auto-queue "ensemble methods"        # Queue hypotheses
```
