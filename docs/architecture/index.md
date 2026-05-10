---
title: "Architecture"
description: "High-level architecture of the Turing ML research harness: command surface, agent model, configuration system, and the three-tier access model that makes autonomous experimentation trustworthy."
---

# Architecture

Turing is a Claude Code plugin that scaffolds ML experiment infrastructure into user projects, then provides AI agents that autonomously iterate through a formal experiment loop. The system enforces a strict separation between the **hypothesis space** (what the agent can change) and the **measurement apparatus** (how results are evaluated).

## At a Glance

| Dimension | Count |
|-----------|-------|
| Commands | 74 |
| Agents | 2 |
| Config files | 10 |
| Template scripts | 93 |
| ADRs | 16 |
| Tests | 2010 |

## Directory Structure

```
turing/
├── commands/                  # Skill layer -- 74 markdown skill files
│   ├── train.md               # The experiment loop
│   ├── try.md                 # Inject hypotheses
│   ├── sweep.md               # Hyperparameter sweeps
│   ├── validate.md            # Metric stability
│   ├── seed.md                # Multi-seed studies
│   ├── reproduce.md           # Reproducibility verification
│   ├── status.md              # Experiment dashboard
│   ├── compare.md             # Side-by-side comparison
│   ├── brief.md               # Research intelligence report
│   ├── rules/
│   │   └── loop-protocol.md   # Safety constraints
│   └── ...                    # 64 more commands
│
├── agents/                    # Agent layer -- 2 agents
│   ├── ml-researcher.md       # Read/Write/Edit/Bash -- 200 turns
│   └── ml-evaluator.md        # Read/Bash only -- 50 turns
│
├── config/                    # Domain knowledge layer -- 10 files
│   ├── defaults.yaml          # Fallback hyperparameters
│   ├── lifecycle.toml         # Experiment state machine
│   ├── taxonomy.toml          # Experiment type classification
│   ├── experiment_archetypes.yaml  # 8 structured strategies
│   ├── novelty_aliases.yaml   # Duplicate detection normalization
│   ├── task_taxonomy.yaml     # ML task classification
│   ├── failure_modes.yaml     # Failure categorization
│   ├── relationships.toml     # ADR dependency graph
│   ├── state.toml             # Session state
│   └── watch_alerts.yaml      # Watch alert configuration
│
├── templates/                 # Scaffolded into user projects
│   ├── train.py               # Agent-editable training pipeline
│   ├── prepare.py             # READ-ONLY data preparation
│   ├── evaluate.py            # HIDDEN evaluation + probes
│   ├── config.yaml            # Hyperparameters
│   ├── program.md             # The experiment loop protocol
│   ├── features/              # Feature engineering pipeline
│   ├── scripts/               # 22 Python scripts + 2 hooks
│   └── tests/                 # Test fixtures
│
├── tests/                     # Plugin test suite (2010 tests)
├── src/                       # npm installer (5 JS files)
├── bin/                       # CLI entrypoints
├── docs/                      # Documentation + 16 ADRs
└── hypotheses/                # Hypothesis detail files
```

## The Three-Tier Access Model

The core architectural invariant is a strict separation of file access into three tiers. This is not a convention; it is enforced at the tool level.

```mermaid
flowchart TD
    subgraph HYPO["HYPOTHESIS SPACE — READ-WRITE"]
        direction LR
        H1[train.py]
        H2[config.yaml]
    end

    subgraph MEAS["MEASUREMENT APPARATUS — READ-ONLY"]
        direction LR
        M1[prepare.py]
        M2[features/featurizers.py]
    end

    subgraph HIDE["HIDDEN TIER — NONE"]
        direction LR
        E1[evaluate.py]
    end

    HYPO -- "produces run.log" --> MEAS
    MEAS -- "calls at runtime" --> HIDE

    style HYPO fill:#1a472a,stroke:#2d6a4f,color:#fff
    style MEAS fill:#1b3a4b,stroke:#2a6f97,color:#fff
    style HIDE fill:#3d1308,stroke:#9d0208,color:#fff
```

!!! warning "Why evaluate.py is hidden"
    If the agent could read the scoring function, it could exploit fixed seeds, memorize test data distributions, or reverse-engineer the metric calculation. The separation is not a prompt instruction; it is an architectural constraint enforced by tool-level access control.

## Layer Boundaries

```mermaid
flowchart TD
    USER["USER / CLAUDE CODE<br/>Invokes /turing:* commands"]
    USER -- "skill invocation" --> SKILL

    SKILL["SKILL LAYER (commands/)<br/>Thin dispatchers, no business logic, no state"]
    SKILL -- "spawn agent" --> AGENT
    SKILL -- "run script" --> PROJECT

    subgraph AGENT["AGENT LAYER (agents/)"]
        RES["@ml-researcher<br/>read/write"]
        EVAL["@ml-evaluator<br/>read only"]
    end

    subgraph PROJECT["SCAFFOLDED PROJECT (templates/ → user repo)"]
        HYPO["train.py, config.yaml<br/>(hypothesis space)"]
        SCRIPTS["scripts/*.py<br/>(analysis tools)"]
        APPARATUS["prepare.py, evaluate.py<br/>(measurement apparatus)"]
    end

    RES --> HYPO
    EVAL --> SCRIPTS
    PROJECT -- "reads" --> CONFIG["DOMAIN KNOWLEDGE (config/)<br/>lifecycle.toml, taxonomy.toml, defaults.yaml"]
```

## Configuration Philosophy

Domain knowledge is encoded in config files, not agent prompts:

- **TOML** for system-wide domain knowledge (lifecycle state machine, taxonomy, relationships)
- **YAML** for project-specific parameters (hyperparameters, data paths, experiment archetypes)

This means the agent's behavior changes by editing config files, not by rewriting prompts. The experiment lifecycle, convergence rules, and experiment classification all live in parseable, testable data structures.

## Further Reading

- [The Experiment Loop](experiment-loop.md): the 9-step protocol
- [Anti-Cheating Guardrails](anti-cheating.md): six defense layers
- [Agent Architecture](agents.md): capability boundaries
- [Convergence Detection](convergence.md): when to stop
- [The Hypothesis Database](hypothesis-database.md): structured experiment queue
