# ADR-0005: Git-Disciplined Experiment Lifecycle with Branch-per-Experiment

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-31 |
| **Author** | Prannaya Gupta |
| **Supersedes** | (none) |
| **Category** | Process |

## Context

The autoresearch agent runs experiments by modifying `train.py`, executing training, and evaluating results. Two problems arise:

1. **Rollback**: if an experiment doesn't improve metrics, the code changes must be undone cleanly. A failed experiment that leaves residue in the codebase will corrupt subsequent experiments.

2. **Provenance**: every code variant should be recoverable for later comparison. If the agent tries 20 hyperparameter configurations, the code for each should be accessible even if only the best was kept on main.

Git provides both mechanisms — but only if used with discipline. An agent that commits before each experiment and reverts on failure gets clean rollback. An agent that creates branches per experiment gets permanent provenance.

## Options Considered

### Option 1: Branch-per-Experiment (preferred)

Create `exp/NNN-description` branch for each experiment. Run the experiment on the branch. If improved, merge to main. If not, return to main without merging (branch preserved).

Trade-offs: creates many branches. Requires merge discipline. But preserves every code variant permanently.

### Option 2: Commit/Revert on Main (fallback)

Commit before each experiment on main. If improved, keep the commit. If not, `git reset --hard HEAD~1`.

Trade-offs: simpler. But failed experiments leave no trace — the code variant is lost. Fine for sweeps where the configuration (not the code) varies.

### Option 3: External Experiment Tracking (MLflow/W&B)

Use an external experiment tracking system to record code snapshots, hyperparameters, and metrics.

Trade-offs: richer tracking features (parameter logging, artifact storage, comparison UI). But adds an external dependency, requires a tracking server, and doesn't integrate with git-based rollback.

### Option 4: No Version Control Integration

Modify code in place, track results in the experiment log only.

Trade-offs: simplest. But no rollback mechanism — a failed experiment that breaks the pipeline requires manual recovery.

## Decision

**We will use branch-per-experiment as the primary git workflow and commit/revert as a fallback for sweeps** because branches preserve every code variant while keeping main clean, and the commit/revert fallback handles the high-frequency sweep case where branch overhead is impractical.

## Rationale

Git is the lab notebook. Each branch is a page recording one experiment. The main branch is the "current best" state. This maps cleanly to the experiment lifecycle in `config/lifecycle.toml`: proposed (committed on branch) → running → evaluating → kept (merged) or discarded (branch preserved but not merged).

The fallback (commit/revert) is explicitly documented as a second-class option for sweeps, where creating 36 branches for a 3x4x3 grid search would be excessive. In the fallback mode, the experiment log (`experiments/log.jsonl`) is the only record of discarded experiments.

## Consequences

### Positive

- Every successful experiment is a clean merge commit on main
- Every attempted experiment (successful or not) has a recoverable branch
- Rollback is clean and automatic — `git checkout main` leaves no residue
- Git log on main tells the story of progressive improvement

### Negative

- Accumulates branches over time — may need periodic cleanup
- Branch management adds complexity to the agent's git operations
- The commit/revert fallback loses code provenance for discarded experiments

### Neutral

- Experiment branches follow a naming convention (`exp/NNN-description`) for discoverability

## References

- [Version Control as Lab Notebook](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004668) — Ram, 2013
- `commands/rules/loop-protocol.md` — git discipline rules
- `config/lifecycle.toml` — experiment state machine (proposed → kept/discarded maps to branch → merge/preserve)
- `templates/program.md` — the COMMIT and DECIDE steps of the loop
