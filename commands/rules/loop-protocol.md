# Autoresearch Loop Protocol Rules

These rules govern the autonomous ML experiment loop. They are non-negotiable safety constraints.

## File Modification Rules

- **NEVER modify prepare.py or evaluate.py.** These are READ-ONLY evaluation infrastructure. Modifying them invalidates all experiment comparisons.
- **ONLY modify train.py** for experiment changes. All training logic lives there.
- **Do not modify features/featurizers.py directly.** Change how train.py uses the featurizer instead.

## Execution Rules

- **ALWAYS redirect training output:** `python train.py > run.log 2>&1`
- **ALWAYS parse metrics with grep**, never read full training output. Metrics are between `---` delimiters: `grep -A 10 "^---" run.log | head -10`
- **ALWAYS activate the venv first:** `source .venv/bin/activate`
- **NEVER install new packages** without human approval. The venv has all required dependencies.

## Git Discipline

### Per-Experiment Branches (preferred)

- **Create a branch before each experiment:** `git checkout -b exp/{NNN}-{short-description}` (e.g., `exp/003-lightgbm-depth8`)
- **Commit changes on the branch:** `git commit -am "exp: {description}"`
- **Run the experiment on the branch.**
- **If improved:** Merge to main: `git checkout main && git merge exp/{NNN}-{short-description}`. Copy model to `models/best/`.
- **If NOT improved:** Return to main without merging: `git checkout main`. The branch is preserved for comparison.
- **Keep all experiment branches.** They preserve code variants for later comparison.

### Fallback: Commit/Revert (legacy)

If branching is impractical (e.g., mid-sweep):
- **ALWAYS commit before running:** `git commit -am "exp: {description}"` -- this creates a restore point.
- **If metrics improved:** Keep the commit. Copy model to `models/best/`. Update `models/best/metadata.json`.
- **If metrics NOT improved:** Revert immediately: `git reset --hard HEAD~1`
- **Keep the git log clean.** Only successful experiments remain in history.

## Sweep Workflow

When running a hyperparameter sweep:
1. Generate the queue: `python scripts/sweep.py`
2. Check queue status: `python scripts/sweep.py --status`
3. Get next experiment: `python scripts/sweep.py --next`
4. Apply overrides to config.yaml, create branch, run training
5. Mark experiment: `python scripts/sweep.py --mark <name> complete|failed`
6. Repeat until queue is empty

## Logging Rules

- **Log every experiment** to `experiments/log.jsonl` via `python scripts/log_experiment.py` -- regardless of whether the experiment was kept or discarded.
- **Include all metrics, config, and a description** of what was tried and why.

## Convergence Rules

- **N consecutive non-improvements** (configured in config.yaml `convergence.patience`) with less than the configured threshold relative gain = STOP.
- **max_iterations** (if provided by user) overrides convergence -- stop after N iterations.
- **Always report** the final best model, its metrics, and recommended next steps when stopping.

## Safety

- Do not modify any files outside the ML project directory.
- Do not delete experiment logs or model archives.
- If something breaks unexpectedly, stop and report rather than trying to auto-fix evaluation infrastructure.
