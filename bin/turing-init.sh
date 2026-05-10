#!/usr/bin/env bash
# Turing CLI init script.
# Thin wrapper around scripts/scaffold.py — the unified scaffolding
# implementation that both CLI and Claude Code use.
#
# Usage:
#   turing-init [project_name] [ml_dir]
#   turing-init --interactive
#
# For Claude Code usage, use /turing:init instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
TEMPLATES_DIR="${PLUGIN_DIR}/templates"
SCAFFOLD_SCRIPT="${TEMPLATES_DIR}/scripts/scaffold.py"

# Check scaffold script exists
if [[ ! -f "$SCAFFOLD_SCRIPT" ]]; then
    echo "Error: scaffold.py not found at ${SCAFFOLD_SCRIPT}" >&2
    echo "Ensure the Turing plugin is installed correctly." >&2
    exit 1
fi

echo "╔══════════════════════════════════════════╗"
echo "║     Turing ML Research Harness           ║"
echo "║     Autonomous Experiment Infrastructure ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# If no args or --interactive, run interactive mode
if [[ $# -eq 0 ]] || [[ "${1:-}" == "--interactive" ]]; then
    python3 "$SCAFFOLD_SCRIPT" --interactive --templates-dir "$TEMPLATES_DIR" --no-venv
    echo ""
    echo "  To set up the uv environment:"
    echo "    cd <ml_dir> && uv sync"
    exit 0
fi

# Positional args mode: turing-init <project_name> [ml_dir]
PROJECT_NAME="${1:-my-ml-project}"
ML_DIR="${2:-ml/${PROJECT_NAME}}"

python3 "$SCAFFOLD_SCRIPT" \
    --project-name "$PROJECT_NAME" \
    --target-metric "accuracy" \
    --metric-direction "higher" \
    --task-description "ML task for ${PROJECT_NAME}" \
    --ml-dir "$ML_DIR" \
    --data-source "data/training.csv" \
    --templates-dir "$TEMPLATES_DIR" \
    --no-venv --no-hooks

echo ""
echo "Note: Run with --interactive for full setup (metric, data source, etc.)"
echo "Or use Claude Code: /turing:init"
