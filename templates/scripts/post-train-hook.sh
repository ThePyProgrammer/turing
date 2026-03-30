#!/usr/bin/env bash
# Post-training hook for the autoresearch pipeline.
#
# Fired by Claude Code PostToolUse hook after train.py executes.
# Parses metrics from run.log (between --- delimiters) and appends
# a structured entry to experiments/log.jsonl.
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
    LOG_FILE="run.log"
else
    echo "post-train-hook: No run.log found, skipping."
    exit 0
fi

# Parse metrics between --- delimiters
METRICS_BLOCK=$(sed -n '/^---$/,/^---$/p' "$LOG_FILE" | head -20)

if [[ -z "$METRICS_BLOCK" ]]; then
    echo "post-train-hook: No metrics block found in run.log, skipping."
    exit 0
fi

# Extract individual metrics dynamically
# Parse all key:value pairs from the metrics block
declare -A METRICS
while IFS= read -r line; do
    # Skip delimiter lines
    [[ "$line" == "---" ]] && continue
    # Parse key:value
    key=$(echo "$line" | awk -F: '{print $1}' | xargs)
    value=$(echo "$line" | awk -F: '{print $2}' | xargs)
    if [[ -n "$key" && -n "$value" ]]; then
        METRICS["$key"]="$value"
    fi
done <<< "$METRICS_BLOCK"

MODEL_TYPE="${METRICS[model_type]:-unknown}"
TRAIN_SEC="${METRICS[train_seconds]:-0}"

# Remove non-metric keys for the metrics JSON
unset 'METRICS[model_type]'
unset 'METRICS[train_seconds]'

# Build metrics JSON
METRICS_JSON="{"
first=true
for key in "${!METRICS[@]}"; do
    if [[ "$first" == "true" ]]; then
        first=false
    else
        METRICS_JSON+=", "
    fi
    METRICS_JSON+="\"$key\": ${METRICS[$key]}"
done
METRICS_JSON+="}"

CONFIG_JSON="{\"model_type\": \"${MODEL_TYPE}\", \"train_seconds\": ${TRAIN_SEC}}"

# Get git commit
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Get next experiment ID
cd "$ML_DIR"
source .venv/bin/activate 2>/dev/null || true

NEXT_ID=$(python -c "
import sys; sys.path.insert(0, 'scripts')
from log_experiment import get_next_experiment_id
print(get_next_experiment_id('$EXPERIMENT_LOG'))
")

# Log the experiment
python scripts/log_experiment.py \
    "$EXPERIMENT_LOG" \
    "$NEXT_ID" \
    "kept" \
    "$METRICS_JSON" \
    "$CONFIG_JSON" \
    "models/model.joblib" \
    "Auto-logged by post-train-hook"

echo "post-train-hook: Logged ${NEXT_ID} (${MODEL_TYPE})"
