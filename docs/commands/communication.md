---
title: "Research Communication"
description: "Citation management, presentation figure generation, and model changelog generation for stakeholders."
---

# Research Communication

Commands for turning experiment results into communicable artifacts: tracking citations, generating publication-ready figures, and writing human-readable progress narratives.

---

### `/turing:cite`: Citation and attribution manager

Track which papers, datasets, and methods influenced each experiment. Associate citations with specific experiments, audit for missing attributions before submission, and generate BibTeX. Catches the "forgot to cite the method we used" problem before a reviewer does.

**Syntax:** `/turing:cite <add|list|check|bib> [--key Chen2016 --title "XGBoost" --type method --url ...]`

**Examples:**
```
/turing:cite add exp-042 --key Chen2016 --title "XGBoost" --type method --url "https://arxiv.org/abs/1603.02754"
/turing:cite list
/turing:cite check                    # Audit for missing citations
/turing:cite bib                      # Generate BibTeX
```

---

### `/turing:present`: Presentation figure generation

Generate presentation-ready figure specifications from experiment data in seconds. Produces training curves, comparison charts, ablation tables, Pareto plots, and sensitivity heatmaps. Supports light (papers), dark (demos), and poster (large fonts) styles.

**Syntax:** `/turing:present [--figures training,comparison,ablation,pareto,sensitivity] [--style light|dark|poster]`

**Examples:**
```
/turing:present                                  # All figures
/turing:present --figures training,comparison    # Specific figures
/turing:present --style dark                     # Dark theme
```

---

### `/turing:changelog`: Model changelog generation

Translate experiment logs into a narrative that PMs and stakeholders can read in 2 minutes. Auto-generates a human-readable progress summary from experiment history. Technical audience gets experiment IDs and configs; stakeholder audience gets plain English and percentages.

**Syntax:** `/turing:changelog [--since exp-id|date] [--audience technical|stakeholder]`

**Examples:**
```
/turing:changelog                                # Full changelog
/turing:changelog --audience stakeholder         # Non-technical summary
/turing:changelog --since exp-042                # Since specific experiment
```
