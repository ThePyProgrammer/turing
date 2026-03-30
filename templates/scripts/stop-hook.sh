#!/usr/bin/env bash
# Convergence detection hook for the autoresearch pipeline.
#
# Fired by Claude Code Stop hook after each training iteration.
# Reads experiments/log.jsonl and checks if the last N experiments
# (where N = convergence.patience from config.yaml) show insufficient
# improvement over the best prior result.
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

# Use inline Python for convergence detection
cd "$ML_DIR"
source .venv/bin/activate 2>/dev/null || true

python3 -c "
import json
import sys
import yaml

# Read config
config_path = '${CONFIG_FILE}'
try:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    convergence_cfg = config.get('convergence', {})
    patience = convergence_cfg.get('patience', 3)
    improvement_threshold = convergence_cfg.get('improvement_threshold', 0.005)
    eval_cfg = config.get('evaluation', {})
    primary_metric = eval_cfg.get('primary_metric', 'accuracy')
    lower_is_better = eval_cfg.get('lower_is_better', False)
except Exception:
    patience = 3
    improvement_threshold = 0.005
    primary_metric = 'accuracy'
    lower_is_better = False

# Read all experiments from log.jsonl
log_path = '${EXPERIMENT_LOG}'
experiments = []
with open(log_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            value = entry.get('metrics', {}).get(primary_metric)
            if value is not None and entry.get('status') == 'keep':
                experiments.append({'id': entry.get('experiment_id', '?'), 'value': float(value)})
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

total = len(experiments)

# Not enough data to judge convergence
if total < patience:
    print(f'stop-hook: Only {total} experiments, need {patience} to check convergence.', file=sys.stderr)
    sys.exit(0)

# Find the best value across ALL experiments
if lower_is_better:
    best_value = min(e['value'] for e in experiments)
else:
    best_value = max(e['value'] for e in experiments)

# Get the last N experiments
last_n = experiments[-patience:]
last_values = [e['value'] for e in last_n]

print(f'stop-hook: Checked {total} experiments, best {primary_metric}={best_value:.4f}, last {patience} values={[round(v, 4) for v in last_values]}', file=sys.stderr)

# Check convergence: all of the last N experiments must show < threshold improvement over best
non_improvements = 0
for i in range(total - patience, total):
    # Best value before this experiment
    prior_values = [e['value'] for e in experiments[:i]]
    if not prior_values:
        continue

    if lower_is_better:
        prior_best = min(prior_values)
        current_value = experiments[i]['value']
        # For lower-is-better, improvement = (prior_best - current) / prior_best
        if prior_best > 0:
            relative_improvement = (prior_best - current_value) / prior_best
        else:
            relative_improvement = 1.0
    else:
        prior_best = max(prior_values)
        current_value = experiments[i]['value']
        # For higher-is-better, improvement = (current - prior_best) / prior_best
        if prior_best > 0:
            relative_improvement = (current_value - prior_best) / prior_best
        else:
            relative_improvement = 1.0

    if relative_improvement < improvement_threshold:
        non_improvements += 1

if non_improvements >= patience:
    print(f'CONVERGED: {patience} consecutive non-improvements (threshold: {improvement_threshold*100:.1f}% relative {primary_metric} gain)', file=sys.stderr)
    sys.exit(2)
else:
    print(f'stop-hook: Not converged ({non_improvements}/{patience} non-improvements).', file=sys.stderr)
    sys.exit(0)
"
