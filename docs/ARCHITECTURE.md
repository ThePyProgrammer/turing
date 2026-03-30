# ARCHITECTURE.md

*A bird's-eye view of the Helios codebase, following [matklad's philosophy](https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html): if you maintain this document, you save every future contributor 30 minutes of spelunking.*

This document describes the high-level architecture. For **why** things are the way they are, see the [ADRs](adr/README.md). This document describes **what** and **where**.

## One-Paragraph Summary

Helios is a Claude Code plugin that scaffolds ML experiment infrastructure into user projects, then provides AI agents that autonomously iterate through an experiment loop. The system enforces a strict separation between the **hypothesis space** (agent-modifiable training code) and the **measurement apparatus** (immutable evaluation code) — this is the load-bearing invariant documented in [ADR-0002](adr/0002-separate-hypothesis-space-from-measurement-apparatus.md). Two agents with distinct capability boundaries ([ADR-0003](adr/0003-two-agent-architecture-with-least-privilege-boundaries.md)) execute the loop and analyze results. Domain knowledge is encoded in TOML config files ([ADR-0004](adr/0004-toml-config-dsl-for-domain-knowledge.md)), not agent prompts.

## Codemap

```
helios/
│
├── commands/                  ← SKILL LAYER (6 files)
│   │                            Thin dispatchers. Each command is a markdown
│   │                            skill file consumed by Claude Code. No logic
│   │                            lives here — commands orchestrate agents and
│   │                            shell commands.
│   │
│   ├── helios.md              Router. Intent detection → dispatch to sub-command.
│   ├── init.md                Scaffolding orchestration. Reads templates,
│   │                          replaces placeholders, creates venv.
│   ├── train.md               Experiment loop entry point. Delegates to
│   │                          @ml-researcher agent.
│   ├── status.md              Read-only status display. Runs show_metrics.py.
│   ├── compare.md             Side-by-side experiment comparison.
│   ├── sweep.md               Hyperparameter sweep orchestration.
│   └── rules/
│       └── loop-protocol.md   Safety constraints for the experiment loop.
│                              Contains the access control matrix (ADR-0002).
│
├── agents/                    ← AGENT LAYER (2 files)
│   │                            Agent definitions with tool access, turn limits,
│   │                            and behavioral instructions.
│   │
│   ├── ml-researcher.md       WRITER agent. Read/Write/Edit/Bash/Grep/Glob.
│   │                          200 max turns. Has persistent memory.
│   │                          Modifies train.py and config.yaml.
│   │                          THE agent that runs the experiment loop.
│   │
│   └── ml-evaluator.md        READER agent. Read/Bash/Grep/Glob only.
│                              50 max turns. No Write, no Edit.
│                              Structural guarantee: cannot modify code.
│                              Analyzes results for status/compare commands.
│
├── config/                    ← DOMAIN KNOWLEDGE LAYER (3 files)
│   │                            Structured data encoding system-wide vocabulary.
│   │                            TOML for domain knowledge, YAML for defaults.
│   │                            See ADR-0004.
│   │
│   ├── defaults.yaml          Fallback values when project config.yaml is
│   │                          missing keys. Conservative starting points.
│   │                          Also defines template placeholders.
│   │
│   ├── lifecycle.toml         Experiment state machine:
│   │                          proposed → running → evaluating → kept/discarded
│   │                          Transition requirements documented as data.
│   │
│   └── taxonomy.toml          Classification system:
│                              - experiment_types (hyperparameter, architecture, ...)
│                              - failure_modes (overfitting, underfitting, ...)
│                              - model_families (gradient_boosting, linear, ...)
│                              - severity_levels (critical, major, minor, info)
│
├── templates/                 ← SCAFFOLDING LAYER (15+ files)
│   │                            Complete, runnable files with {{PLACEHOLDER}}
│   │                            markers. Copied to user projects by /helios:init.
│   │                            See ADR-0008.
│   │
│   │  ┌─ MEASUREMENT APPARATUS (READ-ONLY after scaffolding) ──┐
│   ├── prepare.py             │ Data loading, stratified splitting.          │
│   ├── evaluate.py            │ Metrics computation, parseable output.       │
│   │                          └─────────────────────────────────────────────┘
│   │
│   │  ┌─ HYPOTHESIS SPACE (AGENT-EDITABLE) ────────────────────┐
│   ├── train.py               │ Default XGBoost pipeline. THE file the       │
│   │                          │ agent modifies. See ADR-0009.                │
│   │                          └─────────────────────────────────────────────┘
│   │
│   │  ┌─ INFRASTRUCTURE ──────────────────────────────────────┐
│   ├── config.yaml            │ Project-specific hyperparameters.            │
│   ├── sweep_config.yaml      │ Cartesian product sweep parameters.          │
│   ├── program.md             │ Agent protocol: the experiment loop.          │
│   ├── README.md              │ Per-project README template.                  │
│   ├── MEMORY.md              │ Agent memory bootstrap template.              │
│   ├── requirements.txt       │ Python deps (xgboost, lightgbm, sklearn).    │
│   ├── pyproject.toml         │ pytest config.                                │
│   │                          └─────────────────────────────────────────────┘
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── featurizers.py     Pluggable feature pipeline: BaseFeaturizer →
│   │                          NumericFeaturizer, CategoricalFeaturizer →
│   │                          CompositeFeaturizer. fit/transform interface.
│   │
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── log_experiment.py  Append-only JSONL + TSV logging. See ADR-0007.
│   │   ├── show_metrics.py    Tabular display of experiment history.
│   │   ├── compare_runs.py    Side-by-side experiment comparison.
│   │   ├── sweep.py           Cartesian product queue generation + management.
│   │   ├── post-train-hook.sh Claude Code PostToolUse hook — auto-logs metrics.
│   │   └── stop-hook.sh       Claude Code Stop hook — convergence detection.
│   │                          Returns exit code 2 to halt /loop. See ADR-0006.
│   │
│   └── tests/
│       ├── __init__.py
│       └── conftest.py        Deterministic test fixtures with sample data.
│
├── src/                       ← INSTALLER LAYER (3 files)
│   │                            npm deployment machinery. See ADR-0010.
│   │
│   ├── install.js             Deploys commands/agents/config to ~/.claude/.
│   │                          Manages CLAUDE.md section with idempotent markers.
│   ├── verify.js              Checks all expected files are in place.
│   └── postinstall.js         npm postinstall — shows setup instructions.
│
├── bin/                       ← CLI LAYER (2 files)
│   ├── cli.sh                 Unified CLI: install | verify | init | help.
│   │                          Entry point for `npx claude-helios`.
│   └── helios-init.sh         Direct scaffolding for non-Claude-Code usage.
│
├── docs/                      ← DOCUMENTATION LAYER
│   ├── ARCHITECTURE.md        This file.
│   └── adr/                   Architecture Decision Records (10 ADRs).
│       ├── README.md          Lifecycle, index, principles.
│       ├── template.md        ADR template.
│       └── 0001-*.md ...      Individual decisions.
│
├── .claude-plugin/
│   └── plugin.json            Claude Code plugin registration metadata.
│
├── package.json               npm package definition.
├── README.md                  Philosophical README for the plugin itself.
├── LICENSE                    MIT.
└── .gitignore
```

## Invariants

These are the rules that must not be broken. They are documented formally in the ADRs; this section provides a quick reference.

1. **Hypothesis-Measurement Separation** ([ADR-0002](adr/0002-separate-hypothesis-space-from-measurement-apparatus.md))
   - `prepare.py` and `evaluate.py` are READ-ONLY to the agent
   - `train.py` and `config.yaml` are the ONLY agent-modifiable files
   - Violation → experiment comparisons become invalid

2. **Agent Capability Boundary** ([ADR-0003](adr/0003-two-agent-architecture-with-least-privilege-boundaries.md))
   - `@ml-researcher`: Read/Write/Edit/Bash/Grep/Glob (200 turns)
   - `@ml-evaluator`: Read/Bash/Grep/Glob only (50 turns)
   - The evaluator MUST NOT have Write or Edit tools

3. **Append-Only Experiment Log** ([ADR-0007](adr/0007-jsonl-append-only-experiment-logging.md))
   - `experiments/log.jsonl` is append-only — never edit past entries
   - Every experiment (kept AND discarded) is logged

4. **Config Format Separation** ([ADR-0004](adr/0004-toml-config-dsl-for-domain-knowledge.md))
   - TOML = system-wide domain knowledge (lifecycle, taxonomy)
   - YAML = project-specific parameters (hyperparameters, data paths)

## Layer Boundaries

```
┌─────────────────────────────────────────────────────────┐
│  USER / CLAUDE CODE                                     │
│  Invokes /helios:* commands                             │
└────────────────────────┬────────────────────────────────┘
                         │ skill invocation
                         ▼
┌─────────────────────────────────────────────────────────┐
│  SKILL LAYER  (commands/)                               │
│  Thin dispatchers. Route intent → agent or script.      │
│  No business logic. No state.                           │
└────────────┬──────────────────────────┬─────────────────┘
             │ spawn agent              │ run script
             ▼                          ▼
┌────────────────────────┐  ┌─────────────────────────────┐
│  AGENT LAYER           │  │  SCAFFOLDED PROJECT          │
│  (agents/)             │  │  (templates/ → user repo)    │
│                        │  │                              │
│  ml-researcher ─────────────→ train.py, config.yaml     │
│  (read/write)          │  │  (hypothesis space)          │
│                        │  │                              │
│  ml-evaluator ──────────────→ scripts/*.py               │
│  (read only)           │  │  (analysis tools)            │
└────────────────────────┘  │                              │
                            │  prepare.py, evaluate.py     │
                            │  (measurement apparatus)     │
                            └──────────────────────────────┘
                                         │
                                         │ reads
                                         ▼
                            ┌──────────────────────────────┐
                            │  DOMAIN KNOWLEDGE (config/)   │
                            │  lifecycle.toml, taxonomy.toml│
                            │  defaults.yaml                │
                            └──────────────────────────────┘
```

### Data Flow: One Experiment Iteration

```
1. Agent reads MEMORY.md + experiments/log.jsonl       (OBSERVE)
2. Agent modifies train.py or config.yaml              (HYPOTHESIZE)
3. Agent commits: git commit -am "exp: description"    (COMMIT)
4. Agent runs: python train.py > run.log 2>&1          (EXECUTE)
5. evaluate.py computes metrics → run.log              (MEASURE)
6. Agent reads: grep -A 10 "^---" run.log              (MEASURE)
7. Agent decides: keep (merge) or discard (revert)     (DECIDE)
8. scripts/log_experiment.py → log.jsonl               (RECORD)
9. Agent updates MEMORY.md                             (RECORD)
10. stop-hook.sh checks convergence                    (CONVERGE?)
```

## Cross-Cutting Concerns

### Placeholder Substitution

The `{{PLACEHOLDER}}` system ([ADR-0008](adr/0008-template-based-project-scaffolding.md)) affects every file in `templates/`. Six placeholders are defined in `config/defaults.yaml` and resolved by `commands/init.md` (Claude Code) or `bin/helios-init.sh` (CLI). Unreplaced placeholders are detectable by grepping for `{{`.

### Hook Integration

Two Claude Code hooks bridge the plugin into the runtime:
- **PostToolUse** → `scripts/post-train-hook.sh`: auto-logs metrics after training
- **Stop** → `scripts/stop-hook.sh`: convergence detection, exit code 2 halts `/loop`

Both hooks are configured in `.claude/settings.local.json` during `/helios:init`.

### Agent Memory

The researcher agent maintains persistent memory at `.claude/agent-memory/ml-researcher/MEMORY.md`. This is read at the START of each session and updated after EACH experiment. It bridges context across `/loop` iterations and separate Claude Code sessions.

## What This Document Does NOT Cover

- Individual ADR rationale → see `docs/adr/`
- Agent prompt engineering → see `agents/*.md`
- Template file contents → see `templates/` directly
- The philosophical foundations → see `README.md`
