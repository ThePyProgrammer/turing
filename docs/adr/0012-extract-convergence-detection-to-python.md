# ADR-0012: Extract Convergence Detection from Shell to Testable Python

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-31 |
| **Author** | Prannaya Gupta |
| **Supersedes** | (none) |
| **Category** | Architecture Pattern |

## Context

The convergence detection logic in `templates/scripts/stop-hook.sh` (lines 33-124) is a 60-line Python program embedded inside a bash heredoc. Bash variables (`${CONFIG_FILE}`, `${EXPERIMENT_LOG}`) are interpolated into the Python source at runtime.

This creates three problems:

1. **Untestable.** The Python code cannot be imported, linted, or unit tested. It exists only as a string inside a bash script. The convergence algorithm — which determines when autonomous training stops — has never been verified with known data.

2. **Fragile.** If `ML_DIR` contains a single quote (e.g., `/home/user/it's-ml/`), the bash-to-Python string interpolation produces a Python syntax error. The `except Exception` catch-all on line 49 silently falls through to hardcoded defaults, masking the error.

3. **Invisible to tools.** Python formatters, type checkers, and linters cannot see the code. It is invisible to `pytest --collect-only`. IDE navigation cannot find it.

This is the second-highest-risk component in the system (after `evaluate.py`). It decides when to stop spending compute on autonomous experimentation.

## Options Considered

### Option 1: Extract to `scripts/check_convergence.py`

Move the convergence logic into a standalone Python module with a CLI interface. The bash script becomes a thin 10-line wrapper that calls the Python module and translates the return value to an exit code.

Trade-offs: the Python module is importable, testable, lintable. Adds one more Python file to templates.

### Option 2: Rewrite Stop Hook Entirely in Python

Replace `stop-hook.sh` with `stop-hook.py`. Claude Code hooks can execute any script.

Trade-offs: eliminates bash entirely for this hook. But changes the hook interface from bash to Python, which may affect portability.

### Option 3: Keep Inline but Add Integration Tests

Keep the bash script as-is but add integration tests that invoke it with mock data.

Trade-offs: tests the actual deployed code. But the tests must set up a full file structure (config.yaml, log.jsonl, venv) to test a single function. Slow, fragile, and still invisible to linters.

## Decision

**We will extract the convergence logic into `scripts/check_convergence.py`** with a CLI interface (`python scripts/check_convergence.py --config config.yaml --log experiments/log.jsonl`), reducing the bash script to a thin wrapper. The Python module exposes `check_convergence()` as an importable function for unit testing.

## Rationale

The convergence algorithm has exactly the properties that make unit testing valuable: it is a pure function of data (experiment log + config), it has clearly defined edge cases (fewer experiments than patience, `prior_best == 0`, `lower_is_better` flag), and a bug produces catastrophic silent failure (premature stop or infinite loop). Extracting it to a testable module is the highest-leverage testing improvement available.

## Consequences

### Positive

- Convergence logic is importable and unit-testable
- Python linters, formatters, and type checkers can see the code
- The bash variable interpolation vulnerability is eliminated
- The `except Exception` catch-all can be replaced with specific error handling

### Negative

- Adds one more Python file to `templates/scripts/`
- The stop hook now has a dependency on `check_convergence.py` (previously self-contained)

### Neutral

- Exit code semantics remain the same (0 = continue, 2 = converged)

## References

- Architecture Evaluation Report (2026-03-31) — bug surface dimension flagged inline Python
- ADR-0006 — patience-based convergence detection (the algorithm being extracted)
- `templates/scripts/stop-hook.sh` lines 33-124 — current inline implementation
