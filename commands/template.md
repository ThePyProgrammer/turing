---
name: template
description: Experiment template library — save winning configs as reusable templates, apply to new projects.
argument-hint: "<save|list|apply|share> [--name name] [--from exp-id]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Turn your best experiment configs into reusable recipes that persist across projects.

## Steps
1. **Sync environment:** `uv sync`
2. **Run:** `uv run python scripts/experiment_templates.py $ARGUMENTS`
3. **Operations:** save (from experiment), list (all templates), apply (to current project), share (export)
4. **Stored at:** `~/.turing/templates/` (cross-project)

## Examples
```
/turing:template save --from exp-042 --name "tabular-xgboost-v2"
/turing:template list
/turing:template apply tabular-xgboost-v2
```
