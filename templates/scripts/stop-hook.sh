#!/usr/bin/env bash
# Convergence detection hook for the autoresearch pipeline.
#
# Fired by Claude Code Stop hook after each training iteration.
# Thin wrapper around scripts/check_convergence.py — the actual
# algorithm lives in testable Python, not inline bash.
#
# This implements a discrete analogue of early stopping from gradient
# descent, adapted for the experiment loop context.
#
# Exit codes:
#   0 = not converged, agent should continue
#   2 = converged, agent should stop (signals /loop to halt)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENT_LOG="${ML_DIR}/experiments/log.jsonl"
CONFIG_FILE="${ML_DIR}/config.yaml"

# Check if log.jsonl exists
if [[ ! -f "$EXPERIMENT_LOG" ]]; then
    echo "stop-hook: No log.jsonl found, not enough data to judge." >&2
    exit 0
fi

cd "$ML_DIR"
source "${SCRIPT_DIR}/turing-run-python.sh"

run_python scripts/check_convergence.py \
    --config "$CONFIG_FILE" \
    --log "$EXPERIMENT_LOG"
