# Changelog

All notable changes to Turing are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/).


## [4.2.0] — 2026-04-01 — What-If Analysis

### Added
- `/turing:whatif` — answer hypothetical questions from existing experiment data: routes to scaling, ablation, sensitivity, ensemble, pruning, stitch, and budget estimators with confidence levels
- `/turing:counterfactual` — input-level counterfactual explanations: greedy perturbation + prototype-based search for minimum-change prediction flips, batch mode for misclassified samples
- `/turing:simulate` — experiment outcome prediction: weighted k-NN surrogate model with novelty penalty, pre-filters configs to save compute budget
- What-if and simulation sections integrated into `/turing:brief` research briefing

### Phase
- **27.1** Counterfactual Experiment Simulation
- **27.2** Input-Level Counterfactual Explanations
- **27.3** Experiment Outcome Prediction

**1740 tests | 69 commands | 88 scripts | 18 commits**


## [4.1.0] — 2026-04-01 — Collaboration

### Added
- `/turing:onboard` — project onboarding walkthrough for new collaborators with audience adaptation (researcher/engineer/stakeholder)
- `/turing:share` — experiment packaging into portable archives with manifest, README, optional model/code/figures
- `/turing:review` — peer review simulation with 10 checks, venue calibration (NeurIPS/ICML), fix commands, 1-10 scoring

### Phase
- **26.1** Project Onboarding
- **26.2** Experiment Packaging
- **26.3** Peer Review Simulation

**1576 tests | 66 commands | 85 scripts | 15 commits**


## [4.0.0] — 2026-04-01 — Research Communication

*The v4.0 milestone. Every result becomes a shareable artifact — citations tracked, presentations generated, progress communicated. All 25 phases complete.*

### Added
- `/turing:cite` — citation & attribution manager: add/list/check/bib, track papers/datasets/methods per experiment, audit missing citations, generate BibTeX
- `/turing:present` — presentation figure generation: training curves, comparison charts, ablation tables, Pareto plots, sensitivity heatmaps with light/dark/poster styles
- `/turing:changelog` — model changelog generation: version-grouped improvements, technical and stakeholder audiences, narrative progress summaries

### Phase
- **25.1** Citation & Attribution Manager
- **25.2** Presentation Figure Generation
- **25.3** Model Changelog Generation
- All 25 phases (72 implementation items) complete

**1566 tests | 63 commands | 82 scripts | 14 commits**

## [3.5.0] — 2026-04-01 — Experiment Archaeology

### Added
- `/turing:trend` — long-term trend analysis: improvement velocity, family ROI, diminishing returns detection, phase transition detection
- `/turing:flashback` — session context restoration: current best, recent experiments, pending hypotheses, annotations, budget, suggested next action
- `/turing:archive` — experiment lifecycle cleanup: identify archivable experiments, compress artifacts, create queryable summary index, dry-run support
- `/turing:annotate` — retrospective annotations: add human notes + tags to experiments, search by content/tag, list by experiment
- `/turing:search` — natural language experiment search: keyword + structured filters (metric>0.85, status:kept, family:baseline), ranked results
- `/turing:template` — experiment template library: save/list/apply/delete reusable configs at ~/.turing/templates/ (cross-project)
- `/turing:replay` — experiment replay: re-run old approach with current infrastructure, compare original vs replayed metrics, verdict

### Phase
- **24.1** Long-Term Trend Analysis
- **24.2** Session Context Restoration
- **24.3** Experiment Lifecycle Cleanup
- **24.4** Retrospective Annotations
- **24.5** Natural Language Search
- **24.6** Template Library
- **24.7** Experiment Replay

**1547 tests | 60 commands | 79 scripts | 26 commits**

## [3.4.0] — 2026-04-01 — Model Surgery

### Added
- `/turing:prune` — weight pruning: magnitude/structured/lottery methods, sparsity sweep with knee point detection, speedup and size estimates
- `/turing:quantize` — post-training quantization: FP16/INT8 dynamic/static, accuracy-latency comparison, QAT recommendation when loss > 1%
- `/turing:merge` — model merging: uniform soup, greedy soup, TIES, DARE — free accuracy with zero latency overhead
- `/turing:surgery` — architecture modification: add/remove layer, widen/narrow, swap activation, inject skip connections, add normalization, auto warm-start

### Phase
- **23.1** Weight Pruning
- **23.2** Post-Training Quantization
- **23.3** Model Merging
- **23.4** Architecture Modification

**1494 tests | 53 commands | 72 scripts | 18 commits**

## [3.3.0] — 2026-04-01 — Feature & Training Intelligence

### Added
- `/turing:feature` — automated feature selection: mutual information, L1, tree-based importance with consensus ranking, redundancy detection, interaction feature generation
- `/turing:curriculum` — training curriculum optimization: difficulty scoring (loss, margin, disagreement), strategy comparison (easy-to-hard, hard-to-easy, self-paced, random), impossible sample detection

### Phase
- **22.1** Automated Feature Selection
- **22.2** Training Curriculum Optimization

**1440 tests | 49 commands | 68 scripts | 11 commits**

## [3.2.0] — 2026-04-01 — Model Debugging

### Added
- `/turing:xray` — internal model diagnostics: gradient flow, dead neurons, activation stats, weight distributions (neural), depth utilization and feature dominance (tree), coefficient magnitudes (sklearn)
- `/turing:sensitivity` — hyperparameter sensitivity analysis: sweep generation, normalized sensitivity scoring (HIGH/MED/LOW/NONE), monotonicity detection, tuning recommendations
- `/turing:calibrate` — probability calibration: ECE/MCE computation, reliability diagrams, Platt scaling, isotonic regression, temperature scaling, auto-method selection

### Fixed
- Skip tunable param check when sweep data is pre-computed in sensitivity analysis

### Phase
- **21.1** Internal Model Diagnostics
- **21.2** Hyperparameter Sensitivity Analysis
- **21.3** Probability Calibration

**1398 tests | 47 commands | 66 scripts | 15 commits**

## [3.1.0] — 2026-04-01 — Pre-Training Intelligence

### Added
- `/turing:sanity` — pre-training sanity checks: initial loss vs theory, single-batch overfit test, gradient flow analysis, output validation, data pipeline check, config consistency
- `/turing:baseline` — automatic baseline generation: random, majority class/mean, stratified/median, logistic regression/ridge, k-NN with comparison table
- `/turing:leak` — targeted leakage detection: feature-target correlation, single-feature predictiveness, train/test hash overlap detection

### Phase
- **20.1** Pre-Training Sanity Checks
- **20.2** Automatic Baseline Generation
- **20.3** Targeted Leakage Detection

**1339 tests | 44 commands | 63 scripts | 15 commits**

## [3.0.0] — 2026-04-01 — Meta-Intelligence

*The final milestone. Turing becomes project-aware — learning across projects, not just within one.*

### Added
- `/turing:transfer` — cross-project knowledge transfer: scan prior projects for similar task characteristics, compute similarity scores, surface winning strategies and key insights, generate transfer hypotheses, maintain local project index at `~/.turing/project_index.yaml`
- `/turing:audit` — pre-submission methodology audit: 8 checks (data leakage, CV strategy, seed sensitivity, ablation completeness, baseline comparison, reproducibility, hyperparameter budget, regression stability) with venue-specific checklists (NeurIPS, ICML, ICLR) and auto-fix suggestions
- Methodology Audit section in `/turing:brief`

### Fixed
- Handle empty signatures in similarity matching — require non-None task type and feature types

### Phase
- **19.1** Cross-Project Knowledge Transfer
- **19.2** Pre-Submission Methodology Audit
- All 19 phases (50 implementation items) complete

**1272 tests | 41 commands | 60 scripts | 15 commits**

## [2.5.0] — 2026-04-01 — Scaling & Efficiency

### Added
- `/turing:scale` — scaling law estimator: run experiments at different data/compute/model sizes, fit power law, predict full-scale performance, ASCII plots, diminishing-returns verdicts
- `/turing:budget` — compute budget manager: set experiment/time limits, auto-shift explore→mixed→exploit phases, burn rate projections, hard stop at 100%
- `/turing:distill` — model compression via distillation: auto-select student architecture, soft labels/feature matching/dataset distillation, accuracy/size/latency tradeoff verdicts
- Budget and Scaling Predictions sections in `/turing:brief`

### Phase
- **18.1** Scaling Law Estimator
- **18.2** Compute Budget Manager
- **18.3** Model Compression

**1195 tests | 39 commands | 58 scripts | 18 commits**

## [2.4.0] — 2026-04-01 — Model Composition

### Added
- `/turing:ensemble` — automated ensemble construction: voting, weighted voting, stacking, and blending from top-K models with diversity filtering and correlation analysis
- `/turing:stitch` — pipeline composition: decompose ML pipelines into swappable stages (preprocess, features, model, postprocess) with stage hashing and intermediate caching
- `/turing:warm` — warm-start from prior model: auto-detects model type (tree/neural/sklearn), plans continue-boosting, weight loading with layer freezing, or warm_start param
- Ensembles section in `/turing:brief`

### Phase
- **17.1** Automated Ensemble Construction
- **17.2** Pipeline Composition
- **17.3** Warm-Start from Prior Model

**1072 tests | 36 commands | 55 scripts | 18 commits**

## [2.3.0] — 2026-04-01 — Deep Analysis

### Added
- `/turing:diff` — deep experiment comparison: config diffs with magnitude, metric deltas with statistical significance, per-class regression detection, training curve divergence points, feature importance shifts, optional code diff
- `/turing:watch` — live training monitor with configurable alert rules (loss spikes, NaN detection, overfitting onset, metric plateaus), compact dashboard, post-hoc analysis mode
- `/turing:regress` — performance regression gate: re-run best experiment after changes, verify metrics haven't degraded, capture environment diffs, pass/warning/fail verdicts
- `config/watch_alerts.yaml` — configurable alert rules for training monitor
- Stability section in `/turing:brief` with regression check history
- `/implement` skill for systematic roadmap phase implementation

### Fixed
- Filter NaN values from rolling statistics in training monitor

### Phase
- **16.1** Deep Experiment Comparison
- **16.2** Live Training Monitor
- **16.3** Performance Regression Gate

**931 tests | 33 commands | 52 scripts | 23 commits**

## [2.2.1] — 2026-04-01

### Fixed
- Add missing runtime dependencies (`pandas>=2.0`, `scipy>=1.12`, `joblib>=1.3`) to `pyproject.toml`
- Add `uv.lock` for reproducible installs
- Fix 14 test failures caused by missing packages in clean virtual environments

**778 tests passing (0 failures) | 30 commands | 49 scripts | 2 commits**

## [2.2.0] — 2026-04-01 — Orchestration

### Added
- `/turing:queue` — batch experiment scheduler with priority ordering and dependency chains
- `/turing:retry` — smart failure recovery: auto-diagnose crash type (OOM, NaN, timeout, import, convergence, data) and retry with targeted fix
- `/turing:fork` — experiment branching: run parallel tracks from common parent, report winner
- `config/failure_modes.yaml` — configurable failure taxonomy for retry
- Queue Report section in `/turing:brief`

### Phase
- **15.1** Experiment Scheduler
- **15.2** Smart Failure Recovery
- **15.3** Experiment Branching

**727 tests | 30 commands | 49 scripts | 17 commits**

## [2.1.0] — 2026-04-01 — Research Workflow

### Added
- `/turing:lit` — literature search via Semantic Scholar API (free query, baseline SOTA comparison, related papers)
- `/turing:paper` — draft paper sections from experiment logs (setup, results, ablation, hyperparameters) in LaTeX and markdown
- Auto-hypothesis generation from literature findings with `source: "literature"`

### Phase
- **14.1** Literature Integration
- **14.2** Paper Section Drafting
- All 14 original roadmap phases complete

**664 tests | 27 commands | 46 scripts | 12 commits**

## [2.0.0] — 2026-04-01 — Deployment Bridge

### Added
- `/turing:export` — export model to production formats (joblib, xgboost_json, lightgbm_text, onnx, torchscript, tflite)
- `equivalence_checker.py` — inference equivalence verification (exact, approximate, divergent)
- `latency_benchmark.py` — p50/p95/p99 inference latency with warm-up and speedup comparison
- `export_card.py` — deployment model card with metrics, seed study, equivalence, latency, dependencies
- Format registry with auto-detection based on model type

### Phase
- **13.1** Model Export — Turing crosses from experiment engine to production pipeline

**611 tests | 25 commands | 44 scripts | 14 commits**

## [1.5.0] — 2026-03-31 — Performance & Resources

### Added
- `/turing:profile` — computational profiling: timing breakdown, memory (RSS + Python + GPU), throughput, bottleneck detection with 5 pattern types and actionable recommendations
- `/turing:checkpoint` — smart checkpoint management: list, prune (Pareto-based), average top-K, resume, disk usage stats
- Performance Profile section in `/turing:brief`

### Phase
- **12.1** Computational Profiling
- **12.2** Smart Checkpoint Manager

**542 tests | 24 commands | 39 scripts | 13 commits**

## [1.4.0] — 2026-03-31 — Experiment Intelligence

### Added
- `/turing:diagnose` — error analysis: confusion matrix, per-class P/R/F1, most-confused pairs, regression residual analysis, feature-range bias, failure mode clustering, auto-hypothesis generation
- `/turing:ablate` — systematic ablation studies: auto-detect components, per-component runs, delta table, dead-weight flagging, LaTeX output
- `/turing:frontier` — N-dimensional Pareto frontier: multi-objective dominance, closest-neighbor for dominated points, ASCII scatter plots
- Error Analysis section in `/turing:brief`

### Phase
- **11.1** Error Analysis
- **11.2** Ablation Studies
- **11.3** Pareto Frontier Visualization

**487 tests | 22 commands | 37 scripts | 18 commits**

## [1.3.0] — 2026-03-31 — Statistical Rigor

### Added
- `/turing:seed` — multi-seed study: mean/std/95% CI, coefficient of variation, seed-sensitivity flagging
- `/turing:reproduce` — reproducibility verification: re-run from logged config, tolerance checking, environment drift detection (pip freeze diff)
- Seed study sections in `/turing:brief` and `/turing:card`
- Seed study step in convergence protocol (`program.md`)
- `load_seed_study()` and `load_reproduction()` in `turing_io.py`

### Phase
- **10.1** Multi-Seed Runner
- **10.2** Reproducibility Verification

**407 tests | 19 commands | 34 scripts | 28 commits**

## [1.2.0] — 2026-03-31 — Tree-Search Hypothesis Exploration

### Added
- `/turing:explore` — AB-MCTS tree search over hypothesis space via TreeQuest integration
- `treequest_suggest.py` — seed generation, deterministic child expansion (18 strategies), critique-based scoring
- `treequest` as hypothesis source with priority ordering: human > literature > treequest > taxonomy > agent
- `[treequest]` markers in `/turing:brief`
- Greedy best-first fallback when TreeQuest not installed

### Phase
- **9.1** TreeQuest Integration

**379 tests | 17 commands | 31 scripts | 11 commits**

## [1.1.0] — 2026-03-31 — Cost-Performance Frontier and Model Cards

### Added
- `cost_frontier.py` — Pareto-optimal cost-performance analysis with efficiency scoring
- `/turing:card` — standardized model card generation (performance, limitations, intended use, artifact contract)
- `leaderboard.py` — ranked experiment table with delta-vs-leader
- `diff_configs.py`, `export_results.py`, `plot_trajectory.py`
- Cost-Performance Analysis section in `/turing:brief`

### Phase
- **8.1** Cost-Performance Frontier
- **8.2** Model Cards

**345 tests | 16 commands | 30 scripts | 19 commits**

## [1.0.1] — 2026-03-31 — Multi-Project, Model Contracts, Model Registry

### Added
- Multi-project support: scaffold multiple ML projects in one repo with independent state
- `model_contract.md` — formal schema for trained model bundles
- `model_registry.yaml` — catalog of model architectures with default hyperparameters
- Multi-project detection in router and brief commands

**311 tests | 15 commands | 25 scripts | 11 commits**

## [1.0.0] — 2026-03-31 — The Research Assistant That Can't Fool Itself

### Added
- Core autoresearch loop: `/turing:init`, `/turing:train`, `/turing:status`, `/turing:compare`
- Hypothesis database: `/turing:try` with priority queue, status transitions, detail files
- Research briefing: `/turing:brief` with campaign summary, trajectory, failure patterns
- Immutable evaluation infrastructure: hidden `evaluate.py`, behavioral probes
- Anti-cheating guardrails: 6 defense layers
- Novelty guard with configurable alias tables
- Statistical comparison (Mann-Whitney U test, multi-run evaluation)
- Experiment families with exhaustion detection
- Decision packets (promote, branch_followup, abandon, fix_and_retry)
- Hyperparameter sweep: `/turing:sweep`
- Literature-grounded model suggestions: `/turing:suggest`
- Experiment design scaffolding: `/turing:design`
- Research mode switching: `/turing:mode`
- Pre-flight resource check: `/turing:preflight`
- Logbook, poster, report generation
- 2 specialized agents: `@ml-researcher` (read/write), `@ml-evaluator` (read-only)
- 16 Architecture Decision Records

**257 tests | 14 commands | 23 scripts | 193 commits**

[4.1.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v4.1.0
[4.0.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v4.0.0
[4.2.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v4.2.0
[4.1.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v4.1.0
[4.0.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v4.0.0
[3.5.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v3.5.0
[3.4.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v3.4.0
[3.3.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v3.3.0
[3.2.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v3.2.0
[3.1.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v3.1.0
[3.0.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v3.0.0
[2.5.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v2.5.0
[2.4.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v2.4.0
[2.3.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v2.3.0
[2.2.1]: https://github.com/ThePyProgrammer/turing/releases/tag/v2.2.1
[2.2.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v2.2.0
[2.1.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v2.1.0
[2.0.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v2.0.0
[1.5.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v1.5.0
[1.4.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v1.4.0
[1.3.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v1.3.0
[1.2.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v1.2.0
[1.1.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v1.1.0
[1.0.1]: https://github.com/ThePyProgrammer/turing/releases/tag/v1.0.1
[1.0.0]: https://github.com/ThePyProgrammer/turing/releases/tag/v1.0.0
