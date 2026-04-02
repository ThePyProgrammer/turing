---
title: Multiple Projects
description: Run independent ML experiments across multiple projects in the same repository.
---

# Multiple Projects

Each Turing project is fully self-contained. You can run multiple experiments in parallel without any cross-contamination.

## What each project owns

Every `/turing:init` creates an isolated directory with its own:

- `config.yaml`: independent hyperparameters, metric, data source
- `data/`: separate splits and preprocessing
- `experiments/`: its own `log.jsonl` and `results.tsv`
- `models/`: independent model artifacts and archive
- `hypotheses.yaml`: separate hypothesis queue
- `MEMORY.md`: per-project agent memory (what worked, what failed, domain context)

No state is shared between projects.

## Setting up multiple projects

Scaffold each project in its own subdirectory:

```bash
# From your repo root
/turing:init    # prompts: name=sentiment, metric=f1_weighted, data=data/reviews.csv
/turing:init    # prompts: name=pricing, metric=rmse, data=data/transactions.csv
```

This produces:

```
repo/
├── ml/
│   ├── sentiment/
│   │   ├── config.yaml
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── hypotheses.yaml
│   │   └── ...
│   └── pricing/
│       ├── config.yaml
│       ├── train.py
│       ├── evaluate.py
│       ├── hypotheses.yaml
│       └── ...
```

## Targeting a specific project

Pass the project path to any command:

```
/turing:train ml/sentiment
/turing:train ml/pricing
/turing:status ml/sentiment
/turing:brief ml/pricing
```

## Auto-detection from working directory

If you are already inside a project directory (or a subdirectory of one), Turing detects it automatically. The heuristic is simple: walk up from `cwd` until a `config.yaml` with Turing's schema is found.

```bash
cd ml/sentiment
/turing:train          # targets sentiment -- no path argument needed
/turing:status         # also targets sentiment
```

## Comparing across projects

Each project has independent metrics, so cross-project comparison is not built in. If you need to compare results across projects, use `/turing:brief` in each and compare the reports manually; the metrics are different scales anyway.

## Per-project memory

Each project maintains its own `MEMORY.md` that the agent reads at the start of every session. This file accumulates:

- What model architectures worked and what did not
- Feature engineering insights specific to that dataset
- Domain constraints the agent should respect
- Dead ends to avoid repeating

This means the agent working on `sentiment` never sees `pricing`'s memory, and vice versa. Domain knowledge stays scoped.
