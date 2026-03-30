# ADR-0007: JSONL Append-Only Experiment Logging with TSV Summary

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-31 |
| **Author** | Prannaya Gupta |
| **Supersedes** | (none) |
| **Category** | Data Model |

## Context

Every experiment — whether kept or discarded — must be recorded. The experiment log is the ground truth for what was tried, what worked, and what didn't. It serves three consumers:

1. **The agent**: reads recent results to inform next hypothesis (OBSERVE step)
2. **The human**: reviews experiment history to understand the research trajectory
3. **Automated tools**: convergence detection reads the log to count non-improvements

The format must support append-only writes (never edit past entries), fast tail access (read last N), and machine parsing.

## Options Considered

### Option 1: JSONL (JSON Lines) + TSV Summary

Append-only JSONL file where each line is a self-contained JSON object with experiment_id, timestamp, git_commit, status, config, metrics, model_path, and description. Companion TSV file for human-readable quick reference.

Trade-offs: JSONL is widely supported, streamable, and grep-friendly. TSV provides a spreadsheet-compatible view. Two files to maintain.

### Option 2: SQLite Database

Store experiments in a SQLite database with structured queries.

Trade-offs: richer query capabilities. But adds a binary file to git (merge conflicts), requires SQLite library, and is harder for agents to inspect with grep/cat.

### Option 3: CSV

Flat CSV with one row per experiment.

Trade-offs: universally readable. But CSV quoting is fragile for nested data (hyperparameter dicts), no standard for nested objects, and append-only CSV with header management is error-prone.

### Option 4: MLflow / Weights & Biases

External experiment tracking service.

Trade-offs: rich comparison UI, collaboration features, artifact management. But adds external dependency, requires network access, and creates a second source of truth outside the repo.

## Decision

**We will use JSONL as the primary experiment log and TSV as a derived summary** because JSONL supports append-only writes, self-contained records, and machine parsing, while TSV provides human-readable quick reference. Both are text files that diff and grep naturally.

## Rationale

JSONL is the natural format for append-only structured logging. Each line is independent — parsing doesn't require reading the entire file, appending doesn't require reading the existing file, and corruption of one line doesn't affect others. This is important because the post-train hook may append entries concurrently with the agent.

The TSV summary is a convenience, not a source of truth. It is derived from the JSONL log by `log_experiment.py` and can be regenerated if corrupted.

The "every experiment is logged" invariant — including discarded experiments — ensures that the agent's search history is complete. Without it, convergence detection would only see successful experiments and could not count consecutive failures.

## Consequences

### Positive

- Append-only semantics prevent accidental modification of past entries
- JSONL is grep-friendly: `grep "exp-005" experiments/log.jsonl | python -m json.tool`
- Self-contained records: each entry has full metadata, no foreign keys
- TSV provides spreadsheet-compatible view for quick scanning
- Text format diffs naturally in git

### Negative

- JSONL is verbose for simple scans (hence the TSV companion)
- No query language — complex analysis requires loading into Python/pandas
- Log file grows unboundedly over many experiments (manageable at Turing scale)

### Neutral

- `log_experiment.py` maintains both files atomically — the agent calls one function

## References

- [JSON Lines](https://jsonlines.org/) — specification
- `templates/scripts/log_experiment.py` — logging implementation
- `templates/scripts/show_metrics.py` — TSV/JSONL reader
- ADR-0006 — convergence detection reads the JSONL log
