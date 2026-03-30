#!/usr/bin/env bash
# Helios CLI init script.
# Quick-start for scaffolding an ML project outside of Claude Code.
#
# Usage:
#   helios-init [project_name] [ml_dir]
#
# This script copies Helios templates to the specified directory
# and prints instructions for completing the setup.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
TEMPLATES_DIR="${PLUGIN_DIR}/templates"

PROJECT_NAME="${1:-my-ml-project}"
ML_DIR="${2:-.}"

echo "Helios ML Research Harness"
echo "========================="
echo ""
echo "Scaffolding ML project: ${PROJECT_NAME}"
echo "Target directory: ${ML_DIR}"
echo ""

# Check templates exist
if [[ ! -d "$TEMPLATES_DIR" ]]; then
    echo "Error: Templates not found at ${TEMPLATES_DIR}" >&2
    echo "Make sure the Helios plugin is installed correctly." >&2
    exit 1
fi

# Create target directory
mkdir -p "${ML_DIR}"

# Copy template files
echo "Copying templates..."
cp "${TEMPLATES_DIR}/prepare.py" "${ML_DIR}/"
cp "${TEMPLATES_DIR}/train.py" "${ML_DIR}/"
cp "${TEMPLATES_DIR}/evaluate.py" "${ML_DIR}/"
cp "${TEMPLATES_DIR}/config.yaml" "${ML_DIR}/"
cp "${TEMPLATES_DIR}/sweep_config.yaml" "${ML_DIR}/"
cp "${TEMPLATES_DIR}/program.md" "${ML_DIR}/"
cp "${TEMPLATES_DIR}/README.md" "${ML_DIR}/"
cp "${TEMPLATES_DIR}/requirements.txt" "${ML_DIR}/"
cp "${TEMPLATES_DIR}/pyproject.toml" "${ML_DIR}/"
cp "${TEMPLATES_DIR}/MEMORY.md" "${ML_DIR}/"

mkdir -p "${ML_DIR}/features"
cp "${TEMPLATES_DIR}/features/__init__.py" "${ML_DIR}/features/"
cp "${TEMPLATES_DIR}/features/featurizers.py" "${ML_DIR}/features/"

mkdir -p "${ML_DIR}/scripts"
cp "${TEMPLATES_DIR}/scripts/__init__.py" "${ML_DIR}/scripts/"
cp "${TEMPLATES_DIR}/scripts/log_experiment.py" "${ML_DIR}/scripts/"
cp "${TEMPLATES_DIR}/scripts/show_metrics.py" "${ML_DIR}/scripts/"
cp "${TEMPLATES_DIR}/scripts/compare_runs.py" "${ML_DIR}/scripts/"
cp "${TEMPLATES_DIR}/scripts/sweep.py" "${ML_DIR}/scripts/"
cp "${TEMPLATES_DIR}/scripts/post-train-hook.sh" "${ML_DIR}/scripts/"
cp "${TEMPLATES_DIR}/scripts/stop-hook.sh" "${ML_DIR}/scripts/"

mkdir -p "${ML_DIR}/tests"
cp "${TEMPLATES_DIR}/tests/__init__.py" "${ML_DIR}/tests/"
cp "${TEMPLATES_DIR}/tests/conftest.py" "${ML_DIR}/tests/"

# Create data directories
mkdir -p "${ML_DIR}/data/splits"
mkdir -p "${ML_DIR}/experiments"
mkdir -p "${ML_DIR}/models/best"
mkdir -p "${ML_DIR}/models/archive"

# Make shell scripts executable
chmod +x "${ML_DIR}/scripts/post-train-hook.sh"
chmod +x "${ML_DIR}/scripts/stop-hook.sh"

echo ""
echo "Scaffolding complete."
echo ""
echo "Next steps:"
echo "  1. Replace {{PLACEHOLDER}} markers in the copied files:"
echo "     - {{PROJECT_NAME}} -> your project name"
echo "     - {{TARGET_METRIC}} -> your primary metric (accuracy, f1, mae, etc.)"
echo "     - {{TASK_DESCRIPTION}} -> what your model does"
echo "     - {{ML_DIR}} -> relative path to this directory"
echo "     - {{DATA_SOURCE}} -> path to your training data"
echo "     - {{METRIC_DIRECTION}} -> 'lower' or 'higher'"
echo ""
echo "  2. Add your training data"
echo ""
echo "  3. Set up the virtual environment:"
echo "     cd ${ML_DIR} && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
echo ""
echo "  4. Create data splits:"
echo "     python prepare.py"
echo ""
echo "  5. Start training:"
echo "     python train.py > run.log 2>&1"
echo ""
echo "  Or use Claude Code: /helios:train"
echo ""
