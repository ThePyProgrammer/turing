---
name: audit
description: Pre-submission methodology audit — catch data leakage, missing baselines, cherry-picked seeds, and incomplete ablations before a reviewer does.
argument-hint: "[--strict] [--checklist neurips]"
allowed-tools: Read, Bash(*), Grep, Glob
---

A reviewer checklist you run before submitting. Catches methodology mistakes that cause desk rejections.

## Steps

1. **Sync environment:**
   ```bash
   uv sync
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - `--strict` — treat warnings as failures
   - `--checklist neurips|icml|iclr` — add venue-specific checks
   - `--json` — raw JSON output

3. **Run methodology audit:**
   ```bash
   uv run python scripts/methodology_audit.py $ARGUMENTS
   ```

4. **Checks performed:**
   - **Data leakage** (critical): verify prepare.py/evaluate.py separation
   - **CV strategy** (critical): verify appropriate cross-validation for data type
   - **Seed sensitivity** (high): seed studies exist for best experiments
   - **Ablation completeness** (high): ablation studies performed
   - **Baseline comparison** (high): simple baselines in experiment log
   - **Reproducibility** (high): best result successfully reproduced
   - **Hyperparameter budget** (medium): total tuning cost documented
   - **Regression stability** (medium): regression checks performed

5. **Verdicts:**
   - **PASS** — ready for submission
   - **PASS (with warnings)** — address before submission
   - **NEEDS WORK** — fix failures first
   - **FAIL** — critical issues found

6. **Actions:** each failure suggests the `/turing:` command to fix it

7. **Venue checklists:** `--checklist neurips` adds NeurIPS-specific checks (broader impact, reproducibility checklist, code availability)

8. **Saved output:** report in `experiments/audits/audit-YYYY-MM-DD.yaml`

## Examples

```
/turing:audit                          # Standard audit
/turing:audit --strict                 # Warnings become failures
/turing:audit --checklist neurips      # NeurIPS submission checklist
```
