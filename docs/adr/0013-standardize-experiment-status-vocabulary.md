# ADR-0013: Standardize Experiment Status Vocabulary to Match Lifecycle State Machine

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-03-31 |
| **Author** | Prannaya Gupta |
| **Supersedes** | (none) |
| **Category** | Architecture Pattern |

## Context

ADR-0004 established that domain knowledge is encoded in TOML config files and that "agents validate transitions against data, not English." The experiment lifecycle state machine in `config/lifecycle.toml` defines terminal states as `kept` and `discarded` (past tense). However, all Python runtime code uses `keep` and `discard` (present tense/imperative):

- `log_experiment.py:91` — `status: str = "keep"` (default parameter)
- `log_experiment.py:156` — `if entry.get("status") != "keep":`
- `show_metrics.py:53` — `if exp.get("status") != "keep":`
- `post-train-hook.sh:88` — hardcodes `"keep"` status
- `program.md` — documents `keep|discard` as CLI arguments

If anyone writes code that validates against `lifecycle.toml` states — as ADR-0004 promises agents should — every experiment would fail validation because the log says `"keep"` but the state machine says `"kept"`.

## Options Considered

### Option 1: Standardize on `kept`/`discarded` (match lifecycle.toml)

Update all Python scripts, bash hooks, and documentation to use `kept` and `discarded`. The JSONL log records completed state transitions, so past tense is semantically correct.

Trade-offs: breaking change for any existing experiment logs. But no production deployments exist yet.

### Option 2: Standardize on `keep`/`discard` (match runtime code)

Update `lifecycle.toml` to use `keep` and `discard`. Imperative form matches the agent's action ("keep this experiment").

Trade-offs: changes the formal model to match the implementation. Arguably less correct (the log records what happened, not what to do).

### Option 3: Accept Both Forms

Add aliases in the state machine. Both `keep`/`kept` are valid.

Trade-offs: avoids breaking anything. But doubles the vocabulary, making validation more complex.

## Decision

**We will standardize on `kept`/`discarded`** (matching lifecycle.toml) because the experiment log records completed state transitions — past tense is semantically correct — and the formal model should be the source of truth per ADR-0004.

## Rationale

The JSONL log records facts about what happened: "this experiment was kept" or "this experiment was discarded." Past tense reflects this. The lifecycle state machine is the authoritative vocabulary; runtime code should conform to it, not the other way around.

No production logs exist yet, so this is a free change.

## Consequences

### Positive

- Single vocabulary across formal model and runtime code
- Code that validates against lifecycle.toml will work correctly
- ADR-0004's promise ("agents validate against data") becomes actually true

### Negative

- Must update 6+ files (log_experiment.py, show_metrics.py, post-train-hook.sh, program.md, loop-protocol.md, conftest.py)

### Neutral

- The change is mechanical find-and-replace

## References

- Architecture Evaluation Report (2026-03-31) — consistency dimension flagged this as the critical deviation
- ADR-0004 — TOML config DSL as authoritative vocabulary
- `config/lifecycle.toml` — defines `kept` and `discarded`
- `templates/scripts/log_experiment.py` — uses `keep` and `discard`
