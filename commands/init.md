---
name: init
description: Initialize a new ML project in the current directory with Helios autoresearch scaffolding.
disable-model-invocation: true
argument-hint: "[project_name]"
---

Scaffold a new ML project in the current directory with the Helios autoresearch harness.

## Interactive Setup

Ask the user for the following (or accept from `$ARGUMENTS` if provided as JSON):

1. **Project name** (`{{PROJECT_NAME}}`): Name of the ML project (e.g., "sentiment", "churn", "fraud-detection")
2. **Target metric** (`{{TARGET_METRIC}}`): Primary metric to optimize (e.g., "accuracy", "f1", "mae", "mse", "auc")
3. **Metric direction**: Is lower better (mae, mse, loss) or higher better (accuracy, f1, auc)?
4. **Task description** (`{{TASK_DESCRIPTION}}`): What the model does (e.g., "Predict customer churn from usage data")
5. **ML directory** (`{{ML_DIR}}`): Where ML files go relative to project root (e.g., "ml/sentiment", "ml/churn")
6. **Data source** (`{{DATA_SOURCE}}`): Where training data comes from (e.g., "data/training.csv", "data/records.jsonl")

## Scaffolding Steps

1. **Find the Helios plugin templates directory.** The templates are bundled with the plugin installation. Look for them at the plugin's installed location under `templates/`.

2. **Copy templates** to the ML directory, replacing all `{{PLACEHOLDER}}` markers:
   - `prepare.py` -- data loading and splitting (READ-ONLY)
   - `evaluate.py` -- evaluation harness (READ-ONLY)
   - `train.py` -- training code (AGENT-EDITABLE)
   - `features/__init__.py` and `features/featurizers.py`
   - `scripts/` -- all utility scripts
   - `tests/__init__.py` and `tests/conftest.py`
   - `config.yaml`, `sweep_config.yaml`
   - `program.md`, `README.md`
   - `requirements.txt`, `pyproject.toml`

3. **Replace placeholders** in all copied files:
   - `{{PROJECT_NAME}}` with the project name
   - `{{TARGET_METRIC}}` with the primary metric
   - `{{TASK_DESCRIPTION}}` with the task description
   - `{{ML_DIR}}` with the ML directory path
   - `{{DATA_SOURCE}}` with the data source path
   - `{{METRIC_DIRECTION}}` with "lower" or "higher"

4. **Create agent memory directory:**
   ```
   .claude/agent-memory/ml-researcher/MEMORY.md
   ```
   Copy the MEMORY.md template with project-specific placeholders replaced.

5. **Configure hooks** in `.claude/settings.local.json`:
   - Add PostToolUse hook for auto-logging after training
   - Add Stop hook for convergence detection
   - Preserve any existing hooks in the file

6. **Create the Python virtual environment:**
   ```bash
   cd {{ML_DIR}} && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ```

7. **Make shell scripts executable:**
   ```bash
   chmod +x {{ML_DIR}}/scripts/post-train-hook.sh {{ML_DIR}}/scripts/stop-hook.sh
   ```

8. **Create initial directories:**
   ```bash
   mkdir -p {{ML_DIR}}/{data/splits,experiments,models/best,models/archive}
   ```

9. **Report** what was created and next steps:
   - Tell the user to add training data to `{{DATA_SOURCE}}`
   - Suggest running `python prepare.py` to create splits
   - Suggest running `/helios:train` to start the experiment loop

## Template Location

Templates are at the Helios plugin installation path under `templates/`. Use the Glob tool to find the plugin:
```
~/.claude/plugins/*/templates/
```
Or check if the plugin is installed globally via npm.
