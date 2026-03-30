# ADR-0004: TOML Config DSL for Domain Knowledge Encoding

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-31 |
| **Author** | Prannaya Gupta |
| **Supersedes** | (none) |
| **Category** | Architecture Pattern |

## Context

Helios encodes domain knowledge in two forms: agent prompts (natural language instructions) and structured data (experiment types, lifecycle states, failure classifications). When domain knowledge is embedded in agent prompts, it has several problems:

1. **Duplication** — the same categories appear in multiple agent definitions
2. **Inconsistency** — agents may classify the same thing differently
3. **Fragility** — changing a category requires editing multiple files
4. **Opacity** — the classification system is not discoverable without reading every prompt

The alternative is to extract domain knowledge into structured configuration files that agents reference. This follows the principle that **data outlives code** — when categories change, edit a config file, not an agent prompt.

## Options Considered

### Option 1: TOML Configuration Files

Extract domain knowledge into `config/lifecycle.toml` (experiment state machine) and `config/taxonomy.toml` (classification system). Agents reference these files rather than embedding categories in their prompts.

Trade-offs: adds configuration files. Agents need to read config files to understand categories. TOML is human-readable but less expressive than code.

### Option 2: YAML Configuration Files

Same approach but using YAML instead of TOML.

Trade-offs: YAML is more widely used in ML (config.yaml already exists for hyperparameters). But YAML has parsing ambiguities (the Norway problem), and using the same format for both hyperparameters and domain knowledge blurs the distinction between project-specific and system-wide configuration.

### Option 3: Embedded in Agent Prompts

Keep all domain knowledge in agent markdown files as natural language.

Trade-offs: simplest — no additional files. But categories are duplicated, inconsistent across agents, and not machine-parseable for validation.

### Option 4: JSON Schema

Use JSON Schema to define the domain vocabulary with formal validation.

Trade-offs: maximally precise and validatable. But verbose, hard to read, and overkill for the current scale (6 states, 6 experiment types, 8 failure modes).

## Decision

**We will use TOML configuration files for domain knowledge** because TOML provides the right balance of human readability and structured data, clearly distinguishes system-wide domain knowledge (TOML) from project-specific parameters (YAML), and can be read by both agents and automated tools.

## Rationale

The choice of TOML over YAML is deliberate: YAML is used for project-specific configuration (`config.yaml`, `sweep_config.yaml`) that changes per experiment. TOML is used for system-wide domain knowledge (`lifecycle.toml`, `taxonomy.toml`) that defines the vocabulary of the system. The format distinction signals the semantic distinction.

This follows the blueprint plugin's approach, which uses TOML for lifecycle, taxonomy, state, and relationships — establishing a precedent within this plugin ecosystem.

## Consequences

### Positive

- Single source of truth for domain categories — no duplication across agent prompts
- Machine-parseable: tools can validate experiment classifications against the taxonomy
- Human-readable: developers can understand the domain model without reading agent code
- Agents can reference authoritative categories instead of embedded lists

### Negative

- Agents must read additional files before operating
- TOML is less familiar to ML practitioners than YAML or JSON
- Two configuration formats in the project (YAML for experiments, TOML for domain knowledge)

### Neutral

- The config files are small (< 50 lines each) — the maintenance burden is minimal

## References

- [TOML Specification](https://toml.io/) — Tom's Obvious Minimal Language
- [Domain-Specific Languages](https://en.wikipedia.org/wiki/Domain-specific_language) — Fowler, 2010
- `config/lifecycle.toml` — experiment state machine
- `config/taxonomy.toml` — classification system
- blueprint plugin — precedent for TOML config DSL in Claude Code plugins
