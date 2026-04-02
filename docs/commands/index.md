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
| **Core Loop** | init, train, sweep, status, compare | [core.md](core.md) |
| **Taste-Leverage Interface** | try, brief, suggest, explore, design, mode | [taste-leverage.md](taste-leverage.md) |
| **Statistical Rigor** | validate, seed, reproduce | [statistical-rigor.md](statistical-rigor.md) |
| **Experiment Intelligence** | diagnose, ablate, frontier, card, logbook, report, poster | [experiment-intelligence.md](experiment-intelligence.md) |
| **Performance & Resources** | profile, checkpoint | [performance.md](performance.md) |
| **Research Workflow** | lit, paper, export | [research-workflow.md](research-workflow.md) |
| **Pre-flight & Diagnostics** | preflight, doctor, sanity | -- |
| **Data Pipeline** | feature, leak, calibrate, sensitivity | -- |
| **Model Operations** | ensemble, distill, prune, quantize, transfer, warm, stitch, surgery | -- |
| **Experiment Management** | fork, merge, diff, flashback, replay, retry, queue, archive, registry | -- |
| **Monitoring & Trends** | watch, trend, regress, xray | -- |
| **Collaboration** | share, review, annotate, onboard, present, changelog | -- |
| **Planning & Strategy** | plan, baseline, budget, scale, curriculum, counterfactual, whatif, simulate | -- |
| **Search & Discovery** | search, template | -- |
| **Maintenance** | update, postmortem, cite | -- |
| **Router** | turing | -- |

## Command Anatomy

Every Turing command follows a consistent pattern:

```
/turing:<command> [positional args] [--flags]
```

- **Positional arguments** are typically project paths (`ml/sentiment`) or experiment IDs (`exp-042`).
- **Flags** modify behavior (`--auto`, `--deep`, `--format latex`).
- All commands auto-detect the project directory from `config.yaml` if not specified.
- Read-only commands (status, compare, brief) never modify code or data.
