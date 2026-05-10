#!/usr/bin/env bash
# Post-training hook for the autoresearch pipeline.
#
# Fired by Claude Code PostToolUse hook after train.py executes.
# Delegates metric parsing to scripts/parse_metrics.py (the canonical
# parser) rather than reimplementing it in bash.
#
# This hook ensures that every training run is logged, even if the
# agent forgets to call log_experiment.py explicitly. Belt and suspenders.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENT_LOG="${ML_DIR}/experiments/log.jsonl"

# Check for run.log in ML_DIR first, then current directory
if [[ -f "${ML_DIR}/run.log" ]]; then
    LOG_FILE="${ML_DIR}/run.log"
elif [[ -f "run.log" ]]; then
    LOG_FILE="$(pwd)/run.log"
else
    echo "post-train-hook: No run.log found, skipping."
    exit 0
fi

cd "$ML_DIR"
source "${SCRIPT_DIR}/turing-run-python.sh"

# Parse metrics using the canonical parser
PARSED=$(run_python scripts/parse_metrics.py "$LOG_FILE" --raw 2>/dev/null) || {
    echo "post-train-hook: No metrics block found in run.log, skipping."
    exit 0
}

# Extract metrics and metadata via Python (avoids bash JSON construction)
METRICS_JSON=$(run_python -c "
import json, sys
data = json.loads(sys.argv[1])
metadata_keys = {'model_type', 'train_seconds'}
metrics = {k: v for k, v in data.items() if k not in metadata_keys}
print(json.dumps(metrics))
" "$PARSED")

CONFIG_JSON=$(run_python -c "
import json, sys
data = json.loads(sys.argv[1])
metadata_keys = {'model_type', 'train_seconds'}
config = {k: v for k, v in data.items() if k in metadata_keys}
print(json.dumps(config))
" "$PARSED")

# Get git commit
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Get next experiment ID
NEXT_ID=$(run_python -c "
import sys; sys.path.insert(0, 'scripts')
from log_experiment import get_next_experiment_id
print(get_next_experiment_id('$EXPERIMENT_LOG'))
")

# Log the experiment
run_python scripts/log_experiment.py \
    "$EXPERIMENT_LOG" \
    "$NEXT_ID" \
    "kept" \
    "$METRICS_JSON" \
    "$CONFIG_JSON" \
    "models/model.joblib" \
    "Auto-logged by post-train-hook"

echo "post-train-hook: Logged ${NEXT_ID}"
