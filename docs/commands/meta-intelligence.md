---
title: "Meta-Intelligence"
description: "Cross-project knowledge transfer and pre-submission methodology auditing."
---

# Meta-Intelligence

Commands that operate above individual experiments -- transferring institutional knowledge across projects and auditing research methodology before submission.

---

### `/turing:transfer` -- Cross-project knowledge transfer

Find similar prior projects and surface what worked. Builds institutional ML memory: "Last time you had tabular classification with class imbalance, LightGBM beat everything by 3%." Similarity matching uses task type, dataset size, feature types, class balance, and dimensionality. With `--auto`, winning strategies are queued as hypotheses for the current project.

**Syntax:** `/turing:transfer [--from project-path] [--auto] [--index ~/.turing/project_index.yaml] [--json]`

**Examples:**
```
/turing:transfer                                    # Search index for similar projects
/turing:transfer --from ~/projects/fraud-detection  # Transfer from specific project
/turing:transfer --auto                             # Auto-queue hypotheses
```

---

### `/turing:audit` -- Pre-submission methodology audit

A reviewer checklist you run before submitting. Catches methodology mistakes that cause desk rejections: data leakage, missing baselines, cherry-picked seeds, incomplete ablations, undocumented hyperparameter budgets, and more. Each failure suggests the specific `/turing:` command to fix it. Supports venue-specific checklists for NeurIPS, ICML, and ICLR.

**Syntax:** `/turing:audit [--strict] [--checklist neurips|icml|iclr] [--json]`

**Examples:**
```
/turing:audit                          # Standard audit
/turing:audit --strict                 # Warnings become failures
/turing:audit --checklist neurips      # NeurIPS submission checklist
```
