# ADR-0006: Patience-Based Convergence Detection with Stop Hook

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-31 |
| **Author** | Prannaya Gupta |
| **Supersedes** | (none) |
| **Category** | Architecture Pattern |

## Context

An autonomous ML research agent cannot get bored. Without an explicit stopping criterion, it will continue experimenting indefinitely — consuming compute, generating diminishing returns, and potentially degrading the model through overexploration.

Convergence detection answers: "should I keep searching or has the current approach been exhausted?" This is the discrete-time analogue of early stopping in gradient descent, applied not to training iterations within a single model but to the experiment-level loop across different configurations.

The challenge is integrating convergence detection with Claude Code's `/loop` mechanism, which provides the recurring execution but needs an external signal to halt.

## Options Considered

### Option 1: Patience-Based with Stop Hook

Track consecutive non-improvements. After N experiments (patience) each failing to improve by at least threshold%, signal convergence via exit code 2 (Claude Code's stop signal). Configured via `config.yaml`.

Trade-offs: simple, interpretable, configurable. Can be fooled by plateaus that precede breakthroughs.

### Option 2: Statistical Convergence Test

Use a statistical test (e.g., Mann-Whitney U) to determine if recent experiments are drawn from a different distribution than the best. Stop when the null hypothesis (no improvement) cannot be rejected.

Trade-offs: more robust to noise. But requires more experiments to achieve statistical power, and the formalism is hard to explain to non-statisticians.

### Option 3: Bayesian Optimization with Expected Improvement

Model the metric surface with a Gaussian process and stop when expected improvement falls below a threshold.

Trade-offs: theoretically optimal. But requires significant infrastructure (GP fitting, acquisition function), and the overhead is disproportionate for the typical use case of iterating through a handful of model configurations.

### Option 4: Fixed Iteration Count

Run exactly N experiments, no convergence detection.

Trade-offs: predictable compute budget. But wastes resources if convergence happens early and stops too soon if it hasn't converged.

## Decision

**We will use patience-based convergence detection with a Claude Code Stop hook** because it is simple enough to implement in a bash script, configurable per project, interpretable by non-specialists, and integrates cleanly with `/loop` via the exit code protocol.

## Rationale

The patience heuristic is borrowed from early stopping in neural network training (Prechelt, 1998). It is conservative by design — it detects when the current approach is exhausted, not when the global optimum has been found. This is the right default for autonomous operation: stop when marginal returns are negligible, report what was found, and let the human decide whether to restart with a different strategy.

The improvement threshold (default 0.5% relative) prevents the agent from counting trivially different results as improvements. Without it, numerical noise would keep the patience counter at zero indefinitely.

## Consequences

### Positive

- Fully autonomous operation with `/loop 5m /turing:train` — convergence halts the loop
- Configurable per project: `convergence.patience` and `convergence.improvement_threshold`
- Exit code 2 integrates with Claude Code's loop halt mechanism
- Simple enough to implement in a 50-line bash script

### Negative

- Can be fooled by plateaus — a legitimate improvement may exist just beyond the patience horizon
- Does not account for the dimensionality of unexplored parameter space
- Binary signal (converged/not) — no nuance about *how close* to convergence

### Neutral

- The user can override with `max_iterations` argument, disabling convergence in favor of a fixed budget

## References

- [Early Stopping](https://en.wikipedia.org/wiki/Early_stopping) — Prechelt, "Early Stopping — But When?", 1998
- [Multi-Armed Bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit) — the exploration/exploitation tradeoff
- `templates/scripts/stop-hook.sh` — convergence detection implementation
- `config/defaults.yaml` — patience=3, threshold=0.005 defaults
