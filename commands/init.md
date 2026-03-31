---
name: init
description: Initialize a new ML project with the Turing autoresearch harness. Scaffolds the full experiment infrastructure — immutable evaluation pipeline, agent-editable training code, structured logging, convergence detection hooks, and a Python virtual environment.
disable-model-invocation: true
argument-hint: "[project_name]"
---

Scaffold a new ML project with the Turing autoresearch harness. This creates the separation between the measurement apparatus (READ-ONLY) and the hypothesis space (AGENT-EDITABLE) that makes autonomous experimentation trustworthy.

## Interactive Setup

Ask the user for the following (or accept from `$ARGUMENTS` if provided as JSON):

1. **Project name** (`{{PROJECT_NAME}}`): Name of the ML project (e.g., "sentiment", "churn", "fraud-detection")
2. **Target metric** (`{{TARGET_METRIC}}`): Primary metric to optimize (e.g., "accuracy", "f1", "mae", "mse", "auc")
3. **Metric direction**: Is lower better (mae, mse, loss) or higher better (accuracy, f1, auc)?
4. **Task description** (`{{TASK_DESCRIPTION}}`): What the model does (e.g., "Predict customer churn from usage data")
5. **ML directory** (`{{ML_DIR}}`): Where ML files go relative to project root (e.g., "ml/sentiment")
6. **Data source** (`{{DATA_SOURCE}}`): Where training data comes from (e.g., "data/reviews.csv")

## Scaffolding

Once you have all 6 values, delegate to the unified scaffolding script:

```bash
python3 <templates_dir>/scripts/scaffold.py \
    --project-name "<project_name>" \
    --target-metric "<target_metric>" \
    --metric-direction "<metric_direction>" \
    --task-description "<task_description>" \
    --ml-dir "<ml_dir>" \
    --data-source "<data_source>" \
    --templates-dir "<templates_dir>"
```

The scaffold script handles everything in a single atomic operation:
- Copies all template files with placeholder substitution
- Creates data/, experiments/, models/ directories
- Sets up agent memory at `.claude/agent-memory/ml-researcher/MEMORY.md`
- Configures Claude Code hooks in `.claude/settings.local.json`
- Creates Python virtual environment and installs requirements
- Verifies all placeholders were replaced (fails loudly if any remain)

## Locating Templates

Find the templates directory using Glob:
```
~/.claude/plugins/*/templates/
```
Or check if installed via npm by looking for `node_modules/claude-turing/templates/`.

## After Scaffolding

Report what was created:
- The separation: READ-ONLY (`prepare.py`, `evaluate.py`) vs AGENT-EDITABLE (`train.py`)
- Next steps: add data to the configured data source path, run `python prepare.py`, then `/turing:train`
- The taste-leverage loop: `/turing:try` to inject hypotheses, `/turing:brief` for intelligence reports
