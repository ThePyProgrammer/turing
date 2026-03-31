# ARCHITECTURE.md

*A bird's-eye view of the Turing codebase, following [matklad's philosophy](https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html): if you maintain this document, you save every future contributor 30 minutes of spelunking.*

This document describes the high-level architecture. For **why** things are the way they are, see the [ADRs](adr/README.md). This document describes **what** and **where**.

## One-Paragraph Summary

Turing is a Claude Code plugin that scaffolds ML experiment infrastructure into user projects, then provides AI agents that autonomously iterate through an experiment loop. The system enforces a strict separation between the **hypothesis space** (agent-modifiable training code) and the **measurement apparatus** (immutable evaluation code) — this is the load-bearing invariant documented in [ADR-0002](adr/0002-separate-hypothesis-space-from-measurement-apparatus.md). Two agents with distinct capability boundaries ([ADR-0003](adr/0003-two-agent-architecture-with-least-privilege-boundaries.md)) execute the loop and analyze results. Domain knowledge is encoded in TOML config files ([ADR-0004](adr/0004-toml-config-dsl-for-domain-knowledge.md)), not agent prompts.

## Codemap

```
turing/
│
├── commands/                  ← SKILL LAYER (15 files)
│   │                            Thin dispatchers. Each command is a markdown
│   │                            skill file consumed by Claude Code. No logic
│   │                            lives here — commands orchestrate agents and
│   │                            shell commands.
│   │
│   │  ┌─ CORE COMMANDS ──────────────────────────────────────┐
│   ├── turing.md              │ Router. Intent detection → dispatch.         │
│   ├── init.md                │ Project scaffolding (delegates to scaffold.py)│
│   ├── train.md               │ Experiment loop. Delegates to @ml-researcher.│
│   ├── status.md              │ Read-only status. Runs show_metrics.py.      │
│   ├── compare.md             │ Side-by-side experiment comparison.          │
│   ├── sweep.md               │ Hyperparameter sweep orchestration.          │
│   │                          └─────────────────────────────────────────────┘
│   │  ┌─ TASTE-LEVERAGE COMMANDS ────────────────────────────┐
│   ├── try.md                 │ Inject hypothesis (free-text or archetype).  │
│   ├── brief.md               │ Research intelligence report.                │
│   ├── suggest.md             │ Literature-grounded model suggestions.       │
│   ├── design.md              │ Structured experiment design from hypothesis.│
│   ├── mode.md                │ Research strategy (explore/exploit/replicate).│
│   │                          └─────────────────────────────────────────────┘
│   │  ┌─ REPORTING COMMANDS ─────────────────────────────────┐
│   ├── logbook.md             │ Generate HTML experiment logbook.            │
│   ├── poster.md              │ Generate research poster.                    │
│   ├── report.md              │ Generate research report.                    │
│   ├── validate.md            │ Metric stability validation.                │
│   │                          └─────────────────────────────────────────────┘
│   └── rules/
│       └── loop-protocol.md   Safety constraints for the experiment loop.
│                              Contains the access control matrix (ADR-0002).
│
├── agents/                    ← AGENT LAYER (2 files)
│   │
│   ├── ml-researcher.md       WRITER agent. Read/Write/Edit/Bash/Grep/Glob.
│   │                          200 max turns. Has persistent memory.
│   │                          Modifies train.py and config.yaml.
│   │
│   └── ml-evaluator.md        READER agent. Read/Bash/Grep/Glob only.
│                              50 max turns. No Write, no Edit.
│                              Structural guarantee: cannot modify code.
│
├── config/                    ← DOMAIN KNOWLEDGE LAYER (8 files)
│   │                            TOML for domain knowledge, YAML for defaults
│   │                            and complex nested structures. See ADR-0004.
│   │
│   ├── defaults.yaml          Fallback hyperparameters and placeholders.
│   ├── lifecycle.toml         Experiment state machine (ADR-0006).
│   ├── taxonomy.toml          Experiment types, failure modes, model families.
│   ├── experiment_archetypes.yaml  8 structured experiment strategies.
│   ├── novelty_aliases.yaml   Token normalization for novelty guard.
│   ├── task_taxonomy.yaml     ML task classification system.
│   ├── relationships.toml     ADR dependency graph (blueprint).
│   └── state.toml             Blueprint session state.
│
├── templates/                 ← SCAFFOLDING LAYER (30+ files)
│   │                            Complete, runnable files with {{PLACEHOLDER}}
│   │                            markers. Copied to user projects by /turing:init.
│   │
│   │  ┌─ MEASUREMENT APPARATUS (HIDDEN/READ-ONLY) ────────────┐
│   ├── prepare.py             │ Data loading, stratified splitting.           │
│   ├── evaluate.py            │ Metrics computation + behavioral probes.      │
│   │                          └──────────────────────────────────────────────┘
│   │  ┌─ HYPOTHESIS SPACE (AGENT-EDITABLE) ───────────────────┐
│   ├── train.py               │ Default XGBoost pipeline + seed/env pinning.  │
│   │                          └──────────────────────────────────────────────┘
│   │  ┌─ INFRASTRUCTURE ─────────────────────────────────────────┐
│   ├── config.yaml            │ Hyperparameters, eval settings, constraints.   │
│   ├── sweep_config.yaml      │ Cartesian product sweep parameters.            │
│   ├── program.md             │ Agent protocol: the experiment loop.            │
│   ├── README.md, MEMORY.md   │ Per-project docs and agent memory template.    │
│   ├── requirements.txt       │ Python deps.                                   │
│   ├── pyproject.toml         │ pytest config.                                 │
│   │                          └───────────────────────────────────────────────┘
│   ├── features/
│   │   └── featurizers.py     Pluggable feature pipeline (fit/transform).
│   │
│   ├── scripts/               22 Python scripts + 2 bash hooks
│   │   │
│   │   │  ┌─ CORE LOOP ──────────────────────────────────────┐
│   │   ├── turing_io.py       │ Shared data loaders (JSONL, YAML, config).  │
│   │   ├── log_experiment.py  │ Append-only JSONL + TSV logging (ADR-0007). │
│   │   ├── parse_metrics.py   │ Canonical metric parser (ADR-0015).         │
│   │   ├── check_convergence.py│ Convergence detection (ADR-0006/0012).     │
│   │   ├── manage_hypotheses.py│ Hypothesis queue + archetype expansion.    │
│   │   ├── novelty_guard.py   │ History-aware duplicate detection.          │
│   │   ├── synthesize_decision.py│ Post-run verdict + auto-queue.          │
│   │   ├── update_state.py    │ Structured experiment state (YAML).         │
│   │   │                      └─────────────────────────────────────────────┘
│   │   │  ┌─ ANALYSIS / REPORTING ────────────────────────────┐
│   │   ├── show_metrics.py    │ Experiment metrics table + diff display.    │
│   │   ├── compare_runs.py    │ Side-by-side experiment comparison.         │
│   │   ├── generate_brief.py  │ Research briefing with failure clustering.  │
│   │   ├── generate_logbook.py│ HTML experiment logbook.                    │
│   │   ├── show_experiment_tree.py│ Dependency tree visualization.          │
│   │   ├── show_families.py   │ Per-family performance summaries.           │
│   │   ├── experiment_index.py│ TF-IDF semantic experiment search.          │
│   │   ├── show_environment.py│ Environment snapshot display.               │
│   │   │                      └─────────────────────────────────────────────┘
│   │   │  ┌─ INFRASTRUCTURE ──────────────────────────────────┐
│   │   ├── scaffold.py        │ Unified project scaffolding (ADR-0016).     │
│   │   ├── sweep.py           │ Hyperparameter sweep queue.                 │
│   │   ├── verify_placeholders.py│ Post-scaffold placeholder check.         │
│   │   ├── validate_stability.py│ Metric variance + auto-fix.              │
│   │   ├── statistical_compare.py│ Multi-run Mann-Whitney U tests.         │
│   │   ├── suggest_next.py    │ Bayesian surrogate suggestions.             │
│   │   ├── critique_hypothesis.py│ Hypothesis quality scoring.             │
│   │   ├── post-train-hook.sh │ PostToolUse hook — auto-logs metrics.       │
│   │   └── stop-hook.sh       │ Stop hook — convergence detection.          │
│   │                          └─────────────────────────────────────────────┘
│   └── tests/
│       └── conftest.py        Deterministic test fixtures (template for users).
│
├── tests/                     ← PLUGIN TEST SUITE (332 tests)
│   │                            Tests the plugin's template code, not user projects.
│   ├── test_evaluate.py       Measurement apparatus (20 tests)
│   ├── test_convergence.py    Convergence detection (19 tests)
│   ├── test_novelty_guard.py  Novelty guard (25 tests)
│   ├── test_log_experiment.py Experiment logging (14 tests)
│   ├── test_hypotheses.py     Hypothesis queue (14 tests)
│   ├── test_decisions.py      Decision packets (20 tests)
│   ├── test_prepare.py        Data preparation (13 tests)
│   ├── test_featurizers.py    Feature pipeline (13 tests)
│   ├── test_integration.py    Cross-module contracts (5 tests)
│   ├── test_turing_io.py      Shared data loaders (17 tests)
│   └── ...                    + archetypes, brief, families, scaffold, etc.
│
├── src/                       ← INSTALLER LAYER (5 files)
│   │                            npm deployment machinery. See ADR-0010.
│   ├── paths.js               Path resolution for global/project scopes.
│   ├── claude-md.js           CLAUDE.md managed section with idempotent markers.
│   ├── install.js             Deploys 14 commands, 2 agents, 8 configs.
│   ├── verify.js              Validates installation completeness.
│   └── postinstall.js         npm postinstall — setup instructions.
│
├── bin/                       ← CLI LAYER (2 files)
│   ├── cli.js                 Unified CLI: install | verify | init | help.
│   └── turing-init.sh         Direct scaffolding (delegates to scaffold.py).
│
├── docs/                      ← DOCUMENTATION LAYER
│   ├── ARCHITECTURE.md        This file.
│   └── adr/                   16 Architecture Decision Records.
│       ├── README.md          Lifecycle, index, principles.
│       ├── template.md        ADR template.
│       └── 0001-0016-*.md     Individual decisions.
│
├── .claude-plugin/
│   └── plugin.json            Plugin registration (14 commands, 2 agents).
│
├── package.json               npm package definition.
├── CONTRIBUTING.md            Command/script/config authoring conventions.
├── README.md                  Philosophical README.
├── ROADMAP.md                 Feature roadmap (phases 1-9 complete).
├── V1_RELEASE_BLOCKERS.md     Release readiness tracker.
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
│  Invokes /turing:* commands                             │
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

The `{{PLACEHOLDER}}` system ([ADR-0008](adr/0008-template-based-project-scaffolding.md)) affects every file in `templates/`. Six placeholders are defined in `config/defaults.yaml` and resolved by `commands/init.md` (Claude Code) or `bin/turing-init.sh` (CLI). Unreplaced placeholders are detectable by grepping for `{{`.

### Hook Integration

Two Claude Code hooks bridge the plugin into the runtime:
- **PostToolUse** → `scripts/post-train-hook.sh`: auto-logs metrics after training
- **Stop** → `scripts/stop-hook.sh`: convergence detection, exit code 2 halts `/loop`

Both hooks are configured in `.claude/settings.local.json` during `/turing:init`.

### Agent Memory

The researcher agent maintains persistent memory at `.claude/agent-memory/ml-researcher/MEMORY.md`. This is read at the START of each session and updated after EACH experiment. It bridges context across `/loop` iterations and separate Claude Code sessions.

## What This Document Does NOT Cover

- Individual ADR rationale → see `docs/adr/`
- Agent prompt engineering → see `agents/*.md`
- Template file contents → see `templates/` directly
- The philosophical foundations → see `README.md`
