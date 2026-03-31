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
| Hidden | `evaluate.py` | NONE — do not read, reference, or access |
| Measurement | `prepare.py` | READ-ONLY |
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

1. **OBSERVE** — Read recent results, check hypothesis queue, and review failed diffs:
   ```bash
   python scripts/show_metrics.py --last 5
   python scripts/manage_hypotheses.py next 2>/dev/null || echo "No queued hypotheses"
   ```

   For the most recent discarded experiments, read the actual git diff to understand what was tried and failed — do NOT rely on your own memory of what you changed:
   ```bash
   # Show diffs from recent discarded experiment branches
   for branch in $(git branch --list 'exp/*' | tail -3); do
     echo "=== $branch ==="
     git diff main...$branch -- train.py config.yaml 2>/dev/null | head -40
   done
   ```

2. **HYPOTHESIZE** — Check the queue first. If a queued hypothesis exists (especially human-injected, high priority), use it. Otherwise, generate your own and **register it in the queue before executing**:

   **If using a queued hypothesis:**
   ```bash
   python scripts/manage_hypotheses.py mark hyp-NNN in-progress
   ```

   **If generating your own hypothesis**, register it with structured detail:
   ```bash
   python scripts/manage_hypotheses.py add "your hypothesis description" \
     --priority medium --source agent \
     --model-type xgboost \
     --hyperparams '{"max_depth": 8, "n_estimators": 200}' \
     --family optimizer-sweep \
     --tags "depth,estimators" \
     --parent exp-NNN \
     --expected "deeper trees should capture feature interactions"
   python scripts/manage_hypotheses.py mark hyp-NNN in-progress
   ```

   This creates both an index entry in `hypotheses.yaml` and a detailed file at `hypotheses/hyp-NNN.yaml` with full architecture, hyperparameters, expected outcome, and lineage.

   Every experiment must have a corresponding hypothesis in the queue. This ensures the hypothesis database is a complete record of every idea — human and agent alike.

   To read a hypothesis's full detail:
   ```bash
   python scripts/manage_hypotheses.py show hyp-NNN
   ```

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

8. **RECORD** — Log the experiment (kept or discarded):
   ```bash
   python scripts/log_experiment.py experiments/log.jsonl exp-NNN kept|discarded \
     '{"{{TARGET_METRIC}}": X.XX, ...}' \
     '{"model_type": "xgboost", "hyperparams": {...}}' \
     models/model.joblib "Description of hypothesis and outcome"
   ```

   Update the hypothesis status with result metrics:
   ```bash
   python scripts/manage_hypotheses.py mark hyp-NNN tested \
     --result exp-NNN \
     --metrics '{"{{TARGET_METRIC}}": X.XX, ...}' \
     --notes "Brief explanation of what happened and why"
   # or: mark hyp-NNN promising (if it improved significantly)
   # or: mark hyp-NNN dead-end (if it clearly failed)
   ```

   Then synthesize a decision packet and auto-queue follow-ups:
   ```bash
   python scripts/synthesize_decision.py --experiment exp-NNN --auto-queue
   ```
   This produces a verdict (promote/branch_followup/abandon/fix_and_retry) and automatically queues follow-up hypotheses for `branch_followup` and `fix_and_retry` outcomes.

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

## Strategy Escalation Protocol

When consecutive experiments fail to improve, escalate your approach rather than repeating similar attempts:

| Consecutive Failures | Strategy | Description |
|---------------------|----------|-------------|
| 0-1 | **EXPLOIT** | Push further in the current direction — small tweaks, parameter refinement |
| 2-3 | **RE-READ** | Stop. Re-read ALL code from scratch. Your mental model is likely stale. |
| 4-5 | **COMBINE** | Combine two previously successful ideas that haven't been tried together |
| 6+ | **RADICAL** | Abandon the current approach entirely. Try a fundamentally different model, architecture, or feature strategy. |

Track your consecutive failure count. When you hit a new tier, announce it: "Escalating to COMBINE strategy after 4 consecutive failures."

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
