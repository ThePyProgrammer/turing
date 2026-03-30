# Autoresearch: {{PROJECT_NAME}} Model Training

## Goal

{{TASK_DESCRIPTION}}

**Primary metric:** {{TARGET_METRIC}} ({{METRIC_DIRECTION}} is better)
**Secondary metrics:** as configured in config.yaml `evaluation.metrics`

## Constraints

1. **ONLY modify train.py.** `prepare.py` and `evaluate.py` are READ-ONLY. Do not modify them.
2. **ALWAYS redirect training output:** `python train.py > run.log 2>&1`
3. **ALWAYS read metrics from run.log** (grep between `---` delimiters), not from raw training output.
4. **New packages require human approval.** Do not `pip install` new packages without asking first.
5. **Working directory:** Always `source .venv/bin/activate` before running anything.

## Configuration

All hyperparameters live in `config.yaml`. Edit config.yaml for parameter changes -- do NOT hardcode values in train.py.

Key config sections:
- `model.type` -- model framework (xgboost, lightgbm, etc.)
- `model.hyperparams` -- all model hyperparameters
- `convergence.patience` -- consecutive non-improvements before stopping

## Sweep

For systematic hyperparameter search:
1. Edit `sweep_config.yaml` with parameter ranges
2. Generate queue: `python scripts/sweep.py`
3. Check status: `python scripts/sweep.py --status`
4. Get next: `python scripts/sweep.py --next`
5. Apply overrides, create branch, run training
6. Mark done: `python scripts/sweep.py --mark <name> complete|failed`

## Branches

Create per-experiment branches to preserve all code variants:
```
git checkout -b exp/NNN-description
# ... make changes, run experiment ...
# If improved: git checkout main && git merge exp/NNN-description
# If not improved: git checkout main (branch preserved for comparison)
```

## Memory

Read `.claude/agent-memory/ml-researcher/MEMORY.md` at the start of each session.
Update it after each experiment with:
- Best result (if improved)
- What was tried and why
- What worked / what failed
- Promising next directions

## LOOP

The autoresearch experiment loop. Repeat until convergence or max_iterations reached.

1. Read experiments/log.jsonl for recent results:
   ```bash
   python scripts/show_metrics.py --last 5
   ```

2. Propose next experiment (different model, hyperparams, features, or config). Document your hypothesis.

3. Modify `config.yaml` (not train.py) for hyperparameter changes. Only modify train.py for structural code changes (new model types, new features).

4. Commit the experiment:
   ```bash
   git commit -am "exp: {description}"
   ```

5. Run training:
   ```bash
   source .venv/bin/activate && python train.py > run.log 2>&1
   ```

6. Parse metrics from run.log:
   ```bash
   grep -A 10 "^---" run.log | head -10
   ```

7. **If improved** over current best:
   - Keep the commit
   - Copy model to models/best/:
     ```bash
     cp models/model.joblib models/best/model.joblib
     ```
   - Update models/best/metadata.json:
     ```json
     {
       "model_type": "string",
       "metrics": {"{{TARGET_METRIC}}": value, ...},
       "config": {...},
       "timestamp": "ISO-8601",
       "experiment_id": "exp-NNN"
     }
     ```

8. **If NOT improved:**
   ```bash
   git reset --hard HEAD~1
   ```

9. Log the experiment (regardless of keep/discard):
   ```bash
   python scripts/log_experiment.py experiments/log.jsonl exp-NNN keep|discard \
     '{"{{TARGET_METRIC}}": X.XX, ...}' \
     '{"model_type": "xgboost", "hyperparams": {...}}' \
     models/model.joblib "Description of what was tried"
   ```

10. **Check convergence:** N consecutive non-improvements (config.yaml `convergence.patience`) with less than threshold relative gain = STOP.
    Report final best experiment and recommend next steps.

11. **If user provided max_iterations,** stop after N iterations regardless of convergence.

## Experiment Ideas

Starting suggestions (ordered by expected impact):

1. **Hyperparameter sweep:** Try different max_depth, n_estimators, learning_rate values
2. **LightGBM as alternative GBDT:** Often faster than XGBoost with comparable accuracy
3. **Feature engineering:** Add/remove features from the featurizer pipeline
4. **sklearn RandomForest or GradientBoosting:** Different ensemble strategies
5. **Learning rate schedule:** Try lower learning_rate with more n_estimators (e.g., 0.01 with 1000 trees)
6. **Neural network classifier:** If samples > 2000, try a small MLP

## Metrics

Metrics are configured in config.yaml `evaluation.metrics`. The primary metric determines which experiments are "better".

All metrics are printed in the parseable `---` delimited format by evaluate.py's `format_metrics()`.

## Output Format

- **Model artifact:** `models/best/model.joblib`
- **Metadata:** `models/best/metadata.json`
- **Experiment log:** `experiments/log.jsonl` (append-only JSONL)
- **TSV summary:** `experiments/results.tsv` (quick-reference, tab-separated)

## Comparing Runs

To compare two experiments side-by-side:
```bash
python scripts/compare_runs.py exp-001 exp-002
```
