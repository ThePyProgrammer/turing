# ADR-0016: Unify Scaffolding into a Single Implementation

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-03-31 |
| **Author** | Prannaya Gupta |
| **Supersedes** | (none) |
| **Category** | Architecture Pattern |

## Context

Project scaffolding is implemented twice:

1. **`commands/init.md`** (Claude Code path) — 80 lines of markdown instructing an LLM to copy templates, replace placeholders, create venv, configure hooks, and create agent memory.
2. **`bin/turing-init.sh`** (CLI path) — 124 lines of bash that copies templates without substitution, does not configure hooks, and places MEMORY.md in the wrong location.

These two implementations must produce identical results but share no code. They have already diverged:

| Feature | `init.md` | `turing-init.sh` |
|---------|-----------|-------------------|
| Placeholder substitution | Yes (LLM-driven) | No (manual) |
| Hook configuration | Yes | No |
| Agent memory creation | Yes (correct location) | Copies to ML dir root |
| Venv creation | Yes | No |
| Placeholder verification | No | No |

Any future change to the scaffolding contract must be made in both places. This is the shotgun surgery anti-pattern.

## Options Considered

### Option 1: Single Python Scaffolding Script

Extract all scaffolding logic into `scripts/scaffold.py`. The Claude Code command calls `python scaffold.py --interactive` (prompts for values). The CLI calls `python scaffold.py --project-name X --metric Y ...`. One implementation, two interfaces.

Trade-offs: eliminates divergence. But requires Python to be available for CLI scaffolding (currently only bash).

### Option 2: Shared Template Manifest

Create `config/scaffold-manifest.yaml` listing all template files, their destination paths, and which placeholders they contain. Both paths read the manifest.

Trade-offs: ensures file list consistency. But each path still implements its own copy/substitute logic.

### Option 3: Deprecate CLI Path

Declare Claude Code as the only supported scaffolding method. Keep `turing-init.sh` as a minimal convenience that prints "use /turing:init in Claude Code for full setup."

Trade-offs: eliminates the problem by eliminating one path. But reduces accessibility for users who want CLI-only scaffolding.

### Option 4: Accept Divergence

Document that the CLI path is a minimal convenience and the Claude Code path is the full experience. Accept that they differ.

Trade-offs: honest about the status quo. But ensures the CLI path will fall further behind over time.

## Decision

**We will extract scaffolding logic into a single Python script** that both the Claude Code command and CLI invoke. This eliminates the dual-implementation problem and ensures all scaffolding features (substitution, verification, hooks, memory) are available in both paths.

## Rationale

The dual-path problem is a Conway's Law violation waiting to happen: if a second contributor adds a template file, they must update both paths in two different languages. A single Python script provides one place to maintain, one test surface, and consistent behavior regardless of invocation method.

Python is already a dependency of the scaffolded project (the templates are Python). Requiring Python for scaffolding does not add a new dependency — it leverages an existing one.

## Consequences

### Positive

- One implementation to maintain, test, and debug
- CLI and Claude Code users get identical scaffolding
- Placeholder substitution, hook configuration, and verification are available in both paths
- Future template changes require updating one file

### Negative

- CLI scaffolding now requires Python (previously only bash)
- `turing-init.sh` becomes a thin wrapper calling Python

### Neutral

- The scaffolding interface (project name, metric, data source, etc.) remains the same

## References

- Architecture Evaluation Report (2026-03-31) — bug surface and Conway's dimensions flagged dual-path divergence
- ADR-0008 — template-based project scaffolding
- ADR-0014 — placeholder verification (would be integrated into the unified script)
- `commands/init.md` — Claude Code scaffolding path
- `bin/turing-init.sh` — CLI scaffolding path
