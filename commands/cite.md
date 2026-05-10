---
name: cite
description: Citation & attribution manager — track papers, datasets, methods. Audit for missing citations, generate BibTeX.
argument-hint: "<add|list|check|bib> [--key Chen2016 --title XGBoost --url ...]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Track which papers and methods influenced each experiment. Catch missing citations before submission.

## Steps
1. **Sync environment:** `uv sync`
2. **Run:** `uv run python scripts/citation_manager.py $ARGUMENTS`
3. **Operations:** add (associate citation with experiment), list (group by type), check (audit missing), bib (BibTeX)
4. **Stored in:** `experiments/citations.yaml`

## Examples
```
/turing:cite add exp-042 --key Chen2016 --title "XGBoost" --type method --url "https://arxiv.org/abs/1603.02754"
/turing:cite list
/turing:cite check                    # Audit for missing citations
/turing:cite bib                      # Generate BibTeX
```
