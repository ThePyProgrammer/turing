---
title: Changelog
description: All notable changes to Turing, from v1.0.0 through v4.7.0. 29 phases, 84 features, 27 releases.
icon: lucide/history
---

# Changelog

All notable changes to Turing are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## v4.7.0: Modern Skills Layout Mirror { #v470 }

*2026-05-10: Turing now ships a modern Claude Code `skills/turing/` mirror without changing the `/turing` runtime contract.*

- Added a generated `skills/turing/` mirror for modern `SKILL.md` package layout conventions
- Added `src/sync-skills-layout.js` with write/check modes to keep the mirror synchronized from `commands/`
- Added mirror drift tests for missing, stale, and divergent skill files
- Tightened npm package contents to exclude generated Python cache/build artifacts while including `skills/turing/**`

**2032 tests | 74 commands | 93 scripts | 5 commits**

---

## v4.6.0: Command Registry Contract { #v460 }

*2026-05-10: Command semantics are now explicit instead of implied by scattered manifests.*

- Added `config/commands.yaml` as the canonical command and config-file registry
- Moved installer and verifier command manifests to registry-derived loading
- Clarified `/turing` router behavior for `slash_only` and `disable-model-invocation` commands
- Added registry/frontmatter/router/install/verify drift tests, including mutation semantics checks

**28 registry/manifest tests | 74 commands | 11 config files | 10 commits**

---

## v4.5.0: Release Reliability { #v450 }

*2026-05-10: The docs got a public home and the installer/doctor path got boring in the best way.*

- Published the documentation site: homepage, getting started, architecture, command reference, changelog, and philosophy sections
- Hardened `turing-init` template discovery and scaffold verification across installed layouts
- Fixed Claude Stop hook generation and added `/turing:doctor --fix` migration for legacy hook settings
- Polished README/docs prose, diagrams, homepage layout, and visual styling

**2010 tests | 74 commands | 93 scripts | 68 commits**

---

## v4.4.0: Operational Intelligence { #v440 }

*2026-04-02: The final phase. All 29 phases (84 features) complete.*

- `/turing:postmortem`: automated failure diagnosis with 5 root cause categories
- `/turing:doctor`: harness self-diagnosis with 7 health checks and `--fix` mode
- `/turing:plan`: strategic research campaign planning with ROI-based budget allocation

**1986 tests | 74 commands | 93 scripts | 18 commits**

---

## v4.3.0: Model Lifecycle { #v430 }

*2026-04-01*

- `/turing:update`: incremental model update with forgetting detection
- `/turing:registry`: 4-stage model registry with promotion gates
- Enhanced `/turing:card` with fairness analysis and registry status

**1876 tests | 71 commands | 90 scripts | 16 commits**

---

## v4.2.0: What-If Analysis { #v420 }

*2026-04-01*

- `/turing:whatif`: answer hypotheticals from existing data (7 estimators)
- `/turing:counterfactual`: minimum-change prediction explanations
- `/turing:simulate`: predict experiment outcomes with surrogate model

**1740 tests | 69 commands | 88 scripts | 18 commits**

---

## v4.1.0: Collaboration { #v410 }

*2026-04-01*

- `/turing:onboard`: project walkthrough for new collaborators
- `/turing:share`: experiment packaging into portable archives
- `/turing:review`: peer review simulation with venue calibration

**1576 tests | 66 commands | 85 scripts | 15 commits**

---

## v4.0.0: Research Communication { #v400 }

*2026-04-01: The v4.0 milestone. Every result becomes a shareable artifact.*

- `/turing:cite`: citation & attribution manager with BibTeX generation
- `/turing:present`: presentation figure generation (5 chart types, 3 styles)
- `/turing:changelog`: model changelog for technical and stakeholder audiences

**1566 tests | 63 commands | 82 scripts | 14 commits**

---

## v3.5.0: Experiment Archaeology { #v350 }

*2026-04-01*

- `/turing:trend`: long-term improvement velocity and diminishing returns
- `/turing:flashback`: session context restoration
- `/turing:archive`: experiment lifecycle cleanup
- `/turing:annotate`: retrospective annotations
- `/turing:search`: natural language experiment search
- `/turing:template`: reusable config library
- `/turing:replay`: re-run old experiments with current infrastructure

**1547 tests | 60 commands | 79 scripts | 26 commits**

---

## v3.4.0: Model Surgery { #v340 }

*2026-04-01*

- `/turing:prune`: weight pruning (magnitude/structured/lottery)
- `/turing:quantize`: post-training quantization (FP16/INT8)
- `/turing:merge`: model merging (soup/TIES/DARE)
- `/turing:surgery`: architecture modification

**1494 tests | 53 commands | 72 scripts | 18 commits**

---

## v3.3.0: Feature & Training Intelligence { #v330 }

*2026-04-01*

- `/turing:feature`: automated feature selection with consensus ranking
- `/turing:curriculum`: training curriculum optimization

**1440 tests | 49 commands | 68 scripts | 11 commits**

---

## v3.2.0: Model Debugging { #v320 }

*2026-04-01*

- `/turing:xray`: internal model diagnostics
- `/turing:sensitivity`: hyperparameter sensitivity analysis
- `/turing:calibrate`: probability calibration (ECE/MCE, Platt, isotonic)

**1398 tests | 47 commands | 66 scripts | 15 commits**

---

## v3.1.0: Pre-Training Intelligence { #v310 }

*2026-04-01*

- `/turing:sanity`: pre-training sanity checks (6 fast checks)
- `/turing:baseline`: automatic baseline generation
- `/turing:leak`: targeted data leakage detection

**1339 tests | 44 commands | 63 scripts | 15 commits**

---

## v3.0.0: Meta-Intelligence { #v300 }

*2026-04-01: The v3.0 milestone. Turing becomes project-aware.*

- `/turing:transfer`: cross-project knowledge transfer
- `/turing:audit`: pre-submission methodology audit (8 checks)

**1272 tests | 41 commands | 60 scripts | 15 commits**

---

## v2.5.0: Scaling & Efficiency { #v250 }

*2026-04-01*

- `/turing:scale`: scaling law estimator with power-law fit
- `/turing:budget`: compute budget manager with phase shifting
- `/turing:distill`: model compression via distillation

**1195 tests | 39 commands | 58 scripts | 18 commits**

---

## v2.4.0: Model Composition { #v240 }

*2026-04-01*

- `/turing:ensemble`: automated ensemble (voting, stacking, blending)
- `/turing:stitch`: pipeline composition with stage caching
- `/turing:warm`: warm-start from prior model

**1072 tests | 36 commands | 55 scripts | 18 commits**

---

## v2.3.0: Deep Analysis { #v230 }

*2026-04-01*

- `/turing:diff`: deep experiment comparison
- `/turing:watch`: live training monitor with alerts
- `/turing:regress`: performance regression gate

**931 tests | 33 commands | 52 scripts | 23 commits**

---

## v2.2.0: Orchestration { #v220 }

*2026-04-01*

- `/turing:queue`: batch experiment scheduler
- `/turing:retry`: smart failure recovery
- `/turing:fork`: experiment branching

**727 tests | 30 commands | 49 scripts | 17 commits**

---

## v2.1.0: Research Workflow { #v210 }

*2026-04-01*

- `/turing:lit`: literature search via Semantic Scholar
- `/turing:paper`: draft paper sections from experiment logs

**664 tests | 27 commands | 46 scripts | 12 commits**

---

## v2.0.0: Deployment Bridge { #v200 }

*2026-04-01: The v2.0 milestone. Experiments cross into production.*

- `/turing:export`: model export to 6 production formats with equivalence checks

**611 tests | 25 commands | 44 scripts | 14 commits**

---

## v1.5.0: Performance & Resources { #v150 }

*2026-03-31*

- `/turing:profile`: computational profiling with bottleneck detection
- `/turing:checkpoint`: smart checkpoint management (Pareto pruning)

**542 tests | 24 commands | 39 scripts | 13 commits**

---

## v1.4.0: Experiment Intelligence { #v140 }

*2026-03-31*

- `/turing:diagnose`: error analysis with failure mode clustering
- `/turing:ablate`: systematic ablation studies
- `/turing:frontier`: Pareto frontier visualization

**487 tests | 22 commands | 37 scripts | 18 commits**

---

## v1.3.0: Statistical Rigor { #v130 }

*2026-03-31*

- `/turing:seed`: multi-seed study (mean/std/CI)
- `/turing:reproduce`: reproducibility verification

**407 tests | 19 commands | 34 scripts | 28 commits**

---

## v1.2.0: Tree-Search Exploration { #v120 }

*2026-03-31*

- `/turing:explore`: AB-MCTS tree search over hypothesis space via TreeQuest

**379 tests | 17 commands | 31 scripts | 11 commits**

---

## v1.1.0: Cost-Performance & Model Cards { #v110 }

*2026-03-31*

- Cost-performance Pareto frontier
- `/turing:card`: standardized model card generation
- Leaderboard, config diffs, trajectory plotting

**345 tests | 16 commands | 30 scripts | 19 commits**

---

## v1.0.1: Multi-Project Support { #v101 }

*2026-03-31*

- Multi-project scaffold support
- Model contract and registry templates

**311 tests | 15 commands | 25 scripts | 11 commits**

---

## v1.0.0: Initial Release { #v100 }

*2026-03-31: The research assistant that can't fool itself.*

- Core autoresearch loop (init, train, status, compare)
- Hypothesis database with priority queue and detail files
- Research briefing with campaign summary and recommendations
- Immutable evaluation with 6-layer anti-cheating stack
- Novelty guard, experiment families, decision packets
- Literature-grounded suggestions, experiment design, mode switching
- 2 specialized agents (@ml-researcher, @ml-evaluator)
- 16 Architecture Decision Records

**257 tests | 14 commands | 23 scripts | 193 commits**
