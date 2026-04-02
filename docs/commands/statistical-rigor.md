---
title: "Statistical Rigor"
description: "Three commands for validating experiment results: stability checks, multi-seed studies, and reproducibility verification."
---

# Statistical Rigor

These commands ensure your results are trustworthy before you report them. Validate checks metric stability, seed prevents publishing lucky seeds, and reproduce verifies that logged experiments can be re-run with consistent outcomes.

---

### `/turing:validate` -- Stability Validation

Run stability validation on the current experiment configuration. Executes multiple runs to measure metric variance and auto-configures multi-run evaluation if variance is too high.

A coefficient of variation (CV) below 5% means single-run evaluation is sufficient. At 5% or above, the command recommends multi-run evaluation with median aggregation. The `--auto` flag writes `evaluation.n_runs: 3` to `config.yaml` automatically when variance is high.

**Syntax:** `/turing:validate [--auto]`

- `--auto` auto-updates `config.yaml` with multi-run settings if CV >= 5%.

**Examples:**

```
/turing:validate
# Run stability check, report CV and verdict

/turing:validate --auto
# Same check, but auto-configure multi-run if unstable
```

!!! tip
    Run `/turing:validate` early in a project, before investing in many experiments. If the pipeline is unstable, every single-run comparison is unreliable. Catching this early saves hours of wasted training.

---

### `/turing:seed` -- Multi-Seed Study

Run a multi-seed study on an experiment to compute mean, standard deviation, and 95% confidence interval. Flags seed-sensitive results to prevent publishing lucky seeds.

Results are classified as **STABLE** (CV < 5%, safe to report) or **SEED-SENSITIVE** (CV >= 5%, do not report single-seed numbers). Seed-sensitive results should be reported as mean +/- std over N seeds.

**Syntax:** `/turing:seed [N] [--quick] [--exp-id <id>] [--seed-list 42,123,456]`

- A bare number sets the seed count (default: 5).
- `--quick` runs 3 seeds for a fast check.
- `--exp-id exp-042` targets a specific experiment (defaults to best).
- `--seed-list 42,123,456` uses specific seed values.

**Examples:**

```
/turing:seed
# 5 seeds on the best experiment

/turing:seed --quick
# 3 seeds for a fast stability check

/turing:seed 10
# 10 seeds for a thorough study

/turing:seed --exp-id exp-042
# Multi-seed study on a specific experiment
```

!!! tip
    Always run `/turing:seed` before reporting final numbers. A result that looks like accuracy=0.92 on one seed might be 0.89 +/- 0.03 across seeds -- a very different story. Results are saved to `experiments/seed_studies/` for reference.

---

### `/turing:reproduce` -- Reproducibility Verification

Verify reproducibility of a specific experiment by re-running from logged config and checking metrics fall within tolerance. Catches environment drift, non-determinism, and configuration errors.

Results are classified into four verdicts: **reproducible** (exact match), **approximately_reproducible** (within tolerance or 95% CI), **not_reproducible** (outside tolerance and CI), or **environment_changed** (metrics diverge AND library versions differ).

**Syntax:** `/turing:reproduce <exp-id> [--tolerance 0.02] [--strict] [--runs 3]`

- Experiment ID is required.
- `--tolerance 0.02` sets relative tolerance (default: 2%).
- `--strict` requires exact float match (1e-6), overrides tolerance.
- `--runs 3` sets number of reproduction runs (default: 3, 1 for strict).

**Examples:**

```
/turing:reproduce exp-042
# Default: 3 runs, 2% tolerance

/turing:reproduce exp-042 --strict
# Exact match required (for deterministic algorithms)

/turing:reproduce exp-042 --tolerance 0.05 --runs 5
# Lenient tolerance with more runs for noisy models
```

!!! tip
    Run `/turing:reproduce` before exporting a model to production. If the environment has drifted (Python version, package updates), the reproduction check catches it before deployment surprises you. Reports are saved to `experiments/reproductions/`.
