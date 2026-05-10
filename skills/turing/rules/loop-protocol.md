# Autoresearch Loop Protocol Rules

These rules govern the autonomous ML experiment loop. They are non-negotiable safety constraints that preserve the integrity of the experimental process.

## The Fundamental Separation

The autoresearch harness enforces a strict separation between the **hypothesis space** (what the agent can change) and the **measurement apparatus** (how results are evaluated). This separation is the architectural invariant that makes autonomous experimentation trustworthy.

| Layer | Files | Agent Access | Rationale |
|-------|-------|-------------|-----------|
| Hidden | `evaluate.py` | NONE — do not read, write, or reference | Reading evaluation code enables seed exploitation and metric gaming |
| Measurement | `prepare.py` | READ-ONLY | Data loading is visible but immutable |
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

## Tool Restrictions

The researcher agent's Bash access is restricted to a whitelist of necessary commands:

| Allowed Pattern | Purpose |
|-----------------|---------|
| `python train.py:*` | Execute training |
| `python scripts/*:*` | Run utility scripts (logging, metrics, sweep) |
| `git:*` | Branch, commit, merge, reset operations |
| `source .venv/bin/activate:*` | Virtual environment activation |
| `pip:*` | Package installation (requires human approval) |

**Blocked by omission:** `cat`, `head`, `tail`, `less` (prevents reading hidden files via shell), `curl`, `wget` (prevents data exfiltration), arbitrary command execution.

The agent's Read tool is separately governed by the file access tiers above — hidden files are denied at the tool level.

## Reproducibility Rules

Every experiment must be fully reproducible. The training template handles this automatically, but the agent must not subvert it:

- **NEVER use unseeded randomness.** All random state flows from `config.yaml → data.random_state`. The `pin_all_seeds()` function in `train.py` sets stdlib `random`, `numpy`, `PYTHONHASHSEED`, and `torch`/`cuda` seeds from this single source.
- **NEVER modify seeds mid-experiment.** If you need a different seed, use `--seed` flag for multi-run comparison (Phase 2.1). Do not hardcode seeds in `train.py`.
- **Environment is captured automatically.** `train_metadata.json` records python version, package versions, platform, GPU info, and a config hash. Do not modify this recording — it's used by behavioral probes.
- **Config snapshot:** The config at training time is stored inside the model artifact (`model.joblib` contains the full config dict). For any saved model, the exact configuration can be recovered.
- **If adding new dependencies** (requires human approval), note that the environment capture in `train_metadata.json` will automatically record the new package version.

## Safety

- Do not modify files outside the ML project directory.
- Do not delete experiment logs or model archives.
- If something breaks unexpectedly, stop and report — do not auto-fix evaluation infrastructure.
