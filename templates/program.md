# Autoresearch: {{PROJECT_NAME}} Model Training

*"An experiment is a question which science poses to Nature, and a measurement is the recording of Nature's answer."*

## Goal

{{TASK_DESCRIPTION}}

**Primary metric:** {{TARGET_METRIC}} ({{METRIC_DIRECTION}} is better)
**Secondary metrics:** as configured in config.yaml `evaluation.metrics`

## The Fundamental Constraint

**You modify `train.py` and `config.yaml`. You do NOT modify `prepare.py` or `evaluate.py`. Ever.**

This separation is not a convention — it is the architectural invariant that makes your results comparable. If you could change evaluation between experiments, no comparison would be valid. The measurement apparatus is sacred.

| Layer | Files | Your Access |
|-------|-------|-------------|
| Measurement | `prepare.py`, `evaluate.py` | READ-ONLY |
| Hypothesis | `train.py`, `config.yaml` | READ-WRITE |
| Features | `features/featurizers.py` | READ-ONLY (modify how `train.py` uses it) |

## Configuration

All hyperparameters live in `config.yaml`. Edit it for parameter changes — do NOT hardcode values in train.py.

Key sections:
- `model.type` — model framework (xgboost, lightgbm, etc.)
- `model.hyperparams` — all model hyperparameters
- `convergence.patience` — consecutive non-improvements before stopping

## Branches

Create per-experiment branches to preserve all code variants:
```
git checkout -b exp/NNN-description
# ... make changes, run experiment ...
# If improved: git checkout main && git merge exp/NNN-description
# If not improved: git checkout main (branch preserved)
```

## Memory

Read `.claude/agent-memory/ml-researcher/MEMORY.md` at the start of each session.
Update it after each experiment with:
- Best result (if improved)
- What was tried and why
- What worked / what failed
- Promising next directions

## Sweep

For systematic hyperparameter search:
1. Edit `sweep_config.yaml` with parameter ranges
2. Generate queue: `python scripts/sweep.py`
3. Check status: `python scripts/sweep.py --status`
4. Get next: `python scripts/sweep.py --next`
5. Apply overrides, create branch, run training
6. Mark done: `python scripts/sweep.py --mark <name> complete|failed`

## THE LOOP

The autoresearch experiment loop. Each iteration is one experiment — one hypothesis tested.

1. **OBSERVE** — Read recent results:
   ```bash
   python scripts/show_metrics.py --last 5
   ```

2. **HYPOTHESIZE** — Propose next experiment (different model, hyperparams, features, or config). Document what you expect and why.

3. **PREPARE** — Modify `config.yaml` for hyperparameter changes. Only modify `train.py` for structural code changes.

4. **COMMIT** the experiment:
   ```bash
   git commit -am "exp: {description}"
   ```

5. **EXECUTE** training:
   ```bash
   source .venv/bin/activate && python train.py > run.log 2>&1
   ```

6. **MEASURE** — Parse metrics from run.log:
   ```bash
   grep -A 10 "^---" run.log | head -10
   ```

7. **DECIDE:**

   **If improved** over current best:
   - Keep the commit
   - Copy model: `cp models/model.joblib models/best/model.joblib`
   - Update `models/best/metadata.json`

   **If NOT improved:**
   ```bash
   git reset --hard HEAD~1
   ```

8. **RECORD** — Log the experiment (keep or discard):
   ```bash
   python scripts/log_experiment.py experiments/log.jsonl exp-NNN keep|discard \
     '{"{{TARGET_METRIC}}": X.XX, ...}' \
     '{"model_type": "xgboost", "hyperparams": {...}}' \
     models/model.joblib "Description of hypothesis and outcome"
   ```

9. **CONVERGE** — Check stopping conditions:
   - N consecutive non-improvements (`config.yaml` → `convergence.patience`) = STOP
   - `max_iterations` reached = STOP
   - Report final best model and recommend next steps

10. **REPEAT** — return to step 1.

## Execution Rules

- **ALWAYS redirect output:** `python train.py > run.log 2>&1`
- **ALWAYS parse with grep:** `grep -A 10 "^---" run.log | head -10`
- **ALWAYS activate venv:** `source .venv/bin/activate`
- **NEVER install packages** without human approval

## Experiment Ideas

Starting suggestions (ordered by expected impact):

1. **Hyperparameter sweep:** max_depth, n_estimators, learning_rate
2. **LightGBM:** often faster than XGBoost with comparable accuracy
3. **Feature engineering:** domain-specific features via the featurizer pipeline
4. **sklearn alternatives:** RandomForest, GradientBoosting
5. **Learning rate schedule:** lower lr with more estimators (0.01 / 1000 trees)
6. **Neural network:** if samples > 2000, try a small MLP

## Output Format

- **Model artifact:** `models/best/model.joblib`
- **Metadata:** `models/best/metadata.json`
- **Experiment log:** `experiments/log.jsonl` (append-only JSONL)
- **TSV summary:** `experiments/results.tsv`

## Comparing Runs

```bash
python scripts/compare_runs.py exp-001 exp-002
```
