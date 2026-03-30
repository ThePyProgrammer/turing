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

## Scaffolding Steps

1. **Locate Turing plugin templates.** Search for them at the plugin's installed location under `templates/`. Use Glob to find: `~/.claude/plugins/*/templates/` or check the npm global installation.

2. **Copy templates** to the ML directory, replacing all `{{PLACEHOLDER}}` markers:
   - `prepare.py` — data loading and splitting (READ-ONLY infrastructure)
   - `evaluate.py` — evaluation harness (READ-ONLY infrastructure)
   - `train.py` — training code (AGENT-EDITABLE hypothesis space)
   - `features/__init__.py` and `features/featurizers.py`
   - `scripts/` — all utility scripts
   - `tests/__init__.py` and `tests/conftest.py`
   - `config.yaml`, `sweep_config.yaml`
   - `program.md`, `README.md`
   - `requirements.txt`, `pyproject.toml`

3. **Replace placeholders** in all copied files:
   - `{{PROJECT_NAME}}` -> project name
   - `{{TARGET_METRIC}}` -> primary metric
   - `{{TASK_DESCRIPTION}}` -> task description
   - `{{ML_DIR}}` -> ML directory path
   - `{{DATA_SOURCE}}` -> data source path
   - `{{METRIC_DIRECTION}}` -> "lower" or "higher"

4. **Create agent memory:**
   ```
   .claude/agent-memory/ml-researcher/MEMORY.md
   ```
   Copy the MEMORY.md template with placeholders replaced.

5. **Configure hooks** in `.claude/settings.local.json`:
   - PostToolUse hook for auto-logging after training
   - Stop hook for convergence detection
   - Preserve any existing hooks

6. **Create Python virtual environment:**
   ```bash
   cd {{ML_DIR}} && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ```

7. **Make shell scripts executable:**
   ```bash
   chmod +x {{ML_DIR}}/scripts/post-train-hook.sh {{ML_DIR}}/scripts/stop-hook.sh
   ```

8. **Create directories:**
   ```bash
   mkdir -p {{ML_DIR}}/{data/splits,experiments,models/best,models/archive}
   ```

9. **Report** what was created:
   - The separation: READ-ONLY (`prepare.py`, `evaluate.py`) vs AGENT-EDITABLE (`train.py`)
   - Next steps: add data, run `python prepare.py`, then `/turing:train`
   - The safety model: why immutable evaluation matters

## Template Location

Templates are at the Turing plugin installation path under `templates/`. Use Glob:
```
~/.claude/plugins/*/templates/
```
