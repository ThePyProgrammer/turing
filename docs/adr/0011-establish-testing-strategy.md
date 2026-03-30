# ADR-0011: Establish Testing Strategy for Plugin Infrastructure

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-03-31 |
| **Author** | Prannaya Gupta |
| **Supersedes** | (none) |
| **Category** | Process |

## Context

Turing has 10 accepted ADRs documenting architectural invariants and zero automated tests enforcing any of them. The codebase ships an autonomous ML experimentation framework — a system that modifies code, executes training, manages experiment state, detects convergence, and makes keep/discard decisions — with no proof that any of it works correctly.

The gap between documentation quality and test quality is the widest in the project. The architecture is thoughtful; the testing is nonexistent. The result is a system that looks safe on paper and has zero automated verification that it actually is.

Specific untested critical paths:
- `evaluate.py` — the measurement apparatus (ADR-0002's core invariant) with no edge case coverage
- `stop-hook.sh` — convergence detection (ADR-0006) as 60 lines of untestable inline Python in bash
- `log_experiment.py` — append-only logging (ADR-0007) with no verification of ID generation or append semantics
- `sweep.py` — queue management with no verification of cartesian product generation or state transitions
- `featurizers.py` — feature pipeline with no verification of fit/transform contract
- `src/install.js` — plugin deployment with no verification of file placement or CLAUDE.md management

## Options Considered

### Option 1: Tiered Test Suite at Repo Root

Create a `tests/` directory at the repo root (separate from `templates/tests/` which is scaffolding for user projects). Use pytest for Python templates, shell-based integration tests for bash scripts, and optionally vitest for Node.js installer. Organize tests in tiers by risk:

- Tier 1: Measurement apparatus and convergence detection (blocks v1.0.0)
- Tier 2: Experiment logging and sweep queue (blocks v1.0.0)
- Tier 3: Data pipeline and featurizers
- Tier 4: Anti-pattern tests (invariant enforcement)
- Tier 5: Installer and CLI

Trade-offs: comprehensive coverage. Requires maintaining tests in two languages (Python + optionally Node.js). The `templates/` Python files need to be importable from the repo root.

### Option 2: Template-Only Testing

Only test the Python templates via the existing `templates/tests/` directory. Skip Node.js and bash testing.

Trade-offs: simpler. But leaves the installer, CLI, and convergence hook untested — the convergence hook is the second-highest-risk component.

### Option 3: Integration Tests Only

Skip unit tests. Write end-to-end tests that scaffold a project, run a training loop with sample data, and verify the output.

Trade-offs: tests the full pipeline. But slow, hard to debug failures, and doesn't catch edge cases in individual functions.

### Option 4: Status Quo / Do Nothing

Ship without tests. Trust the ADRs and code review.

Trade-offs: fastest to ship. But the ADRs are aspirational documents until tests enforce them. The autonomous training loop could silently corrupt experiment data or fail to detect convergence.

## Decision

**We will implement a tiered test suite at the repo root** with pytest for Python templates, bash integration tests for hooks, and GitHub Actions CI. Tier 1-2 tests (measurement apparatus, convergence, logging, sweep queue) are required before v1.0.0. Tier 3-5 tests are required before v1.1.0.

## Rationale

The testing strategy must match the risk profile. The highest-risk components are the measurement apparatus (`evaluate.py`) and convergence detection (`stop-hook.sh`) — both are on the critical path of autonomous operation and both have edge cases that produce silent failures. Unit testing these functions is cheap and high-value.

Anti-pattern tests (tier 4) are uniquely valuable for this project: they verify that the ADR invariants hold in code, not just in documentation. For example, a test that parses `agents/ml-evaluator.md` and confirms Write/Edit are absent from the tools list enforces ADR-0003 automatically.

## Consequences

### Positive

- Every ADR invariant has at least one test that fails if violated
- Convergence detection is verified with known data before deployment
- Measurement apparatus edge cases (NaN, empty, single-class) are caught
- CI prevents regressions on push

### Negative

- Adds ~500-800 lines of test code to maintain
- Requires pytest, possibly vitest as dev dependencies
- CI adds latency to the development loop

### Neutral

- `templates/tests/conftest.py` remains as scaffolding for user projects, separate from plugin tests

## References

- Architecture Evaluation Report (2026-03-31) — testing dimension scored CRITICAL
- ADR-0002 — hypothesis-measurement separation (untested invariant)
- ADR-0003 — agent capability boundaries (untested invariant)
- ADR-0006 — convergence detection (untested implementation)
- ADR-0007 — append-only logging (untested semantics)
