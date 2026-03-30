# Autoresearch Loop Protocol Rules

These rules govern the autonomous ML experiment loop. They are non-negotiable safety constraints that preserve the integrity of the experimental process.

## The Fundamental Separation

The autoresearch harness enforces a strict separation between the **hypothesis space** (what the agent can change) and the **measurement apparatus** (how results are evaluated). This separation is the architectural invariant that makes autonomous experimentation trustworthy.

| Layer | Files | Agent Access | Rationale |
|-------|-------|-------------|-----------|
| Measurement | `prepare.py`, `evaluate.py` | READ-ONLY | Modifying evaluation invalidates all comparisons |
| Hypothesis | `train.py` | READ-WRITE | All experimental changes go here |
| Configuration | `config.yaml` | READ-WRITE | Hyperparameter changes without code changes |
| Features | `features/featurizers.py` | READ-ONLY | Modify how `train.py` *uses* featurizers instead |

## Execution Rules

- **ALWAYS redirect training output:** `python train.py > run.log 2>&1`
- **ALWAYS parse metrics with grep** between `---` delimiters: `grep -A 10 "^---" run.log | head -10`
- **ALWAYS activate the venv first:** `source .venv/bin/activate`
- **NEVER install new packages** without human approval

## Git Discipline

### Per-Experiment Branches (preferred)

- **Create branch before each experiment:** `git checkout -b exp/{NNN}-{short-description}`
- **Commit changes on the branch:** `git commit -am "exp: {description}"`
- **Run the experiment on the branch**
- **If improved:** `git checkout main && git merge exp/{NNN}-{short-description}`. Copy model to `models/best/`.
- **If NOT improved:** `git checkout main`. Branch preserved for comparison.
- **Keep all experiment branches** — they preserve code variants for later analysis.

### Fallback: Commit/Revert (mid-sweep)

- **ALWAYS commit before running:** `git commit -am "exp: {description}"`
- **If improved:** keep commit, copy model to `models/best/`
- **If NOT improved:** `git reset --hard HEAD~1`

## Sweep Workflow

1. Generate queue: `python scripts/sweep.py`
2. Check status: `python scripts/sweep.py --status`
3. Get next: `python scripts/sweep.py --next`
4. Apply overrides, create branch, run training
5. Mark: `python scripts/sweep.py --mark <name> complete|failed`
6. Repeat until queue is empty

## Logging Rules

- **Log every experiment** to `experiments/log.jsonl` via `python scripts/log_experiment.py` — kept and discarded alike.
- **Include all metrics, config, and description** of the hypothesis and its outcome.

## Convergence Rules

- **N consecutive non-improvements** (from `config.yaml` `convergence.patience`) with less than threshold relative gain = STOP.
- **max_iterations** (if provided) overrides convergence.
- **Always report** final best model, metrics, and recommended next steps when stopping.

## Safety

- Do not modify files outside the ML project directory.
- Do not delete experiment logs or model archives.
- If something breaks unexpectedly, stop and report — do not auto-fix evaluation infrastructure.
