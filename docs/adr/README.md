# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the Turing ML research harness. Each ADR documents a significant architectural decision — the context, the options considered, the decision made, and the consequences accepted.

## Lifecycle

```
                ┌──────────┐
                │ Proposed │
                └────┬─────┘
                     │
              ┌──────┴──────┐
              │   Review    │  ← challenge before accepting
              └──────┬──────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ Accepted │ │ Rejected │ │ Deferred │
   └────┬─────┘ └──────────┘ └─────┬────┘
        │                          │
        │    (trigger met)         │
        │◄─────────────────────────┘
        │
   ┌────┴──────────────┐
   ▼                   ▼
┌──────────────┐ ┌──────────────┐
│  Deprecated  │ │  Superseded  │
│              │ │  by ADR-NNNN │
└──────────────┘ └──────────────┘
```

### Status Definitions

| Status | Meaning |
|--------|---------|
| **Proposed** | Under consideration, not yet decided |
| **Accepted** | Decision in effect, codebase should conform |
| **Rejected** | Considered and explicitly not adopted |
| **Deferred** | Valid but not yet needed; has trigger conditions |
| **Deprecated** | Was accepted, no longer relevant |
| **Superseded** | Replaced by a newer ADR (linked) |

### Transition Rules

- Proposed → Accepted, Rejected, or Deferred (after review)
- Deferred → Accepted (when trigger conditions are met)
- Accepted → Deprecated or Superseded
- Rejected and Deprecated are terminal states
- You cannot accept a rejected ADR (create a new one)
- You cannot supersede a proposed ADR (decide on it first)

## Principles

1. **One decision per ADR.** If a document covers two independent decisions, split it.
2. **Immutable history.** Don't edit accepted ADRs to change the decision. Supersede them.
3. **Honest consequences.** Every decision has negative consequences. Document them.
4. **Evidence over opinion.** Link to benchmarks, prior art, or data when possible.
5. **Context is king.** The decision without context is trivia. The context without the decision is a story.

## Naming Convention

```
NNNN-kebab-case-title.md
```

Sequential numbering, zero-padded to 4 digits. The number never changes, even if the ADR is superseded.

## Index

| # | Title | Status | Category | Date |
|---|-------|--------|----------|------|
| [0001](0001-use-adrs-for-architectural-decisions.md) | Use ADRs for architectural decisions | Accepted | Process | 2026-03-31 |
| [0002](0002-separate-hypothesis-space-from-measurement-apparatus.md) | Separate hypothesis space from measurement apparatus | Accepted | Architecture Pattern | 2026-03-31 |
| [0003](0003-two-agent-architecture-with-least-privilege-boundaries.md) | Two-agent architecture with least-privilege capability boundaries | Accepted | Architecture Pattern | 2026-03-31 |
| [0004](0004-toml-config-dsl-for-domain-knowledge.md) | TOML config DSL for domain knowledge encoding | Accepted | Architecture Pattern | 2026-03-31 |
| [0005](0005-git-disciplined-experiment-lifecycle.md) | Git-disciplined experiment lifecycle with branch-per-experiment | Accepted | Process | 2026-03-31 |
| [0006](0006-patience-based-convergence-detection.md) | Patience-based convergence detection with stop hook | Accepted | Architecture Pattern | 2026-03-31 |
| [0007](0007-jsonl-append-only-experiment-logging.md) | JSONL append-only experiment logging with TSV summary | Accepted | Data Model | 2026-03-31 |
| [0008](0008-template-based-project-scaffolding.md) | Template-based project scaffolding with placeholder substitution | Accepted | Architecture Pattern | 2026-03-31 |
| [0009](0009-xgboost-default-with-pluggable-featurizers.md) | XGBoost as default model with pluggable featurizer pipeline | Accepted | Technology Choice | 2026-03-31 |
| [0010](0010-claude-code-plugin-distribution.md) | Claude Code plugin distribution with npm installer | Accepted | Deployment Strategy | 2026-03-31 |
| [0011](0011-establish-testing-strategy.md) | Establish testing strategy for plugin infrastructure | Accepted | Process | 2026-03-31 |
| [0012](0012-extract-convergence-detection-to-python.md) | Extract convergence detection from shell to testable Python | Accepted | Architecture Pattern | 2026-03-31 |
| [0013](0013-standardize-experiment-status-vocabulary.md) | Standardize experiment status vocabulary to match lifecycle.toml | Accepted | Architecture Pattern | 2026-03-31 |
| [0014](0014-enforce-placeholder-verification.md) | Enforce placeholder substitution verification after scaffolding | Accepted | Architecture Pattern | 2026-03-31 |
| [0015](0015-document-metric-output-contract.md) | Extract metric output format into documented contract | Accepted | Architecture Pattern | 2026-03-31 |
| [0016](0016-unify-scaffolding-implementation.md) | Unify scaffolding into a single implementation | Proposed | Architecture Pattern | 2026-03-31 |

## Commands

| Command | Purpose |
|---------|---------|
| `/blueprint:list` | Show this index with contextual suggestions |
| `/blueprint:new "topic"` | Create a new ADR |
| `/blueprint:review N` | Devil's advocate challenge before accepting |
| `/blueprint:search "term"` | Find decisions by topic |
| `/blueprint:audit` | Verify codebase follows accepted decisions |
