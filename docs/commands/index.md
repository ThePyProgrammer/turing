---
title: "Command Reference"
description: "Complete reference for all 74 Turing commands organized into 16 categories, from project scaffolding to model export."
---

# Command Reference

Turing provides **74 commands** organized into **16 categories**, covering the full lifecycle of autonomous ML research: scaffolding, training, analysis, statistical validation, reporting, and production export.

## Essential Commands

If you read nothing else, learn these three:

| Command | What it does |
|---------|-------------|
| [`/turing:init`](core.md#turinginit--scaffold-a-new-ml-project) | Scaffold a new ML project with the autoresearch harness |
| [`/turing:train`](core.md#turingtrain--run-the-autonomous-experiment-loop) | Run the autonomous experiment loop until convergence |
| [`/turing:brief`](taste-leverage.md#turingbrief--research-intelligence-report) | Generate a 2-minute intelligence briefing on experiment progress |

## Categories

| Category | Commands | Page |
|----------|----------|------|
| **Core Loop** | init, train, sweep, status, compare | [Core Loop](core.md) |
| **Taste-Leverage Interface** | try, brief, suggest, explore, design, mode | [Taste-Leverage](taste-leverage.md) |
| **Statistical Rigor** | validate, seed, reproduce | [Statistical Rigor](statistical-rigor.md) |
| **Experiment Intelligence** | diagnose, ablate, frontier, card, logbook, report, poster | [Experiment Intelligence](experiment-intelligence.md) |
| **Performance & Resources** | profile, checkpoint | [Performance](performance.md) |
| **Research Workflow** | lit, paper, export | [Research Workflow](research-workflow.md) |
| **Experiment Orchestration** | queue, retry, fork | [Orchestration](orchestration.md) |
| **Deep Analysis** | diff, watch, regress | [Deep Analysis](deep-analysis.md) |
| **Model Composition** | ensemble, stitch, warm | [Composition](composition.md) |
| **Scaling & Efficiency** | scale, budget, distill | [Scaling & Efficiency](scaling-efficiency.md) |
| **Meta-Intelligence** | transfer, audit | [Meta-Intelligence](meta-intelligence.md) |
| **Pre-Training Intelligence** | sanity, baseline, leak | [Pre-Training](pre-training.md) |
| **Model Debugging** | xray, sensitivity, calibrate, feature, curriculum | [Model Debugging](model-debugging.md) |
| **Model Surgery** | prune, quantize, merge, surgery | [Model Surgery](model-surgery.md) |
| **Experiment Archaeology** | trend, flashback, archive, annotate, search, template, replay | [Archaeology](archaeology.md) |
| **Research Communication** | cite, present, changelog | [Communication](communication.md) |
| **Collaboration** | onboard, share, review | [Collaboration](collaboration.md) |
| **What-If Analysis** | whatif, counterfactual, simulate | [What-If Analysis](whatif-analysis.md) |
| **Model Lifecycle** | update, registry | [Model Lifecycle](model-lifecycle.md) |
| **Operational Intelligence** | postmortem, doctor, plan, preflight | [Operational](operational.md) |

## Command Anatomy

Every Turing command follows a consistent pattern:

```
/turing:<command> [positional args] [--flags]
```

- **Positional arguments** are typically project paths (`ml/sentiment`) or experiment IDs (`exp-042`).
- **Flags** modify behavior (`--auto`, `--deep`, `--format latex`).
- All commands auto-detect the project directory from `config.yaml` if not specified.
- Read-only commands (status, compare, brief) never modify code or data.
