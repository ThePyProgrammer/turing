# ADR-0008: Template-Based Project Scaffolding with Placeholder Substitution

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-31 |
| **Author** | Prannaya Gupta |
| **Supersedes** | (none) |
| **Category** | Architecture Pattern |

## Context

Helios is a plugin that scaffolds ML project infrastructure into user repositories. The scaffolded files need to be customized per project — metric names, data paths, task descriptions, and directory locations differ for every project. The scaffolding mechanism must balance two concerns:

1. **Customizability**: every project is different
2. **Consistency**: the autoresearch protocol requires specific file structures, naming, and behavior

The challenge is that Helios is a Claude Code plugin (markdown commands + agent definitions) without a runtime — it cannot execute arbitrary code at scaffold time. The scaffolding is done by the `/helios:init` command, which is an LLM agent following instructions.

## Options Considered

### Option 1: Template Files with Placeholder Substitution

Ship complete template files in `templates/`. Use `{{PLACEHOLDER}}` markers that the init command replaces with project-specific values. 6 defined placeholders: PROJECT_NAME, TARGET_METRIC, TASK_DESCRIPTION, ML_DIR, DATA_SOURCE, METRIC_DIRECTION.

Trade-offs: simple, predictable, inspectable. But the agent must do string replacement correctly across many files.

### Option 2: Code Generation

Generate files programmatically from a Node.js/Python script that takes configuration as input.

Trade-offs: more flexible — can conditionally include sections, generate different file sets. But requires a runtime, adds complexity, and the generated files are harder to inspect before deployment.

### Option 3: Interactive Notebook

Scaffold via a Jupyter notebook that the user fills in and executes.

Trade-offs: interactive, visual. But requires Jupyter, doesn't integrate with Claude Code, and isn't automatable.

### Option 4: Copy-and-Edit

Ship example projects. User copies and manually edits.

Trade-offs: no automation overhead. But error-prone — users miss placeholders, forget files, and skip configuration steps.

## Decision

**We will use template files with `{{PLACEHOLDER}}` substitution** because it is the simplest mechanism that works within the Claude Code plugin model, produces inspectable files, and requires no runtime beyond the agent's ability to find-and-replace text.

## Rationale

The template approach leverages the agent's native capability: reading files and replacing text. The init command reads templates from the plugin directory, copies them to the target, and replaces 6 placeholders. This is a task the agent can perform reliably.

The `{{DOUBLE_BRACE}}` syntax was chosen because it is visually distinct from code syntax in Python, YAML, bash, and markdown — making unreplaced placeholders obvious during code review.

Templates are shipped as complete, runnable files (not fragments) so they can be inspected and tested in isolation. The default XGBoost training pipeline works out of the box once data is provided and placeholders are replaced.

## Consequences

### Positive

- Templates are inspectable: `cat templates/train.py` shows exactly what will be scaffolded
- No build step or runtime required
- The agent can verify scaffolding by grepping for remaining `{{` markers
- Both CLI (`helios-init`) and Claude Code (`/helios:init`) use the same templates

### Negative

- Limited expressiveness: no conditional sections, no loops, no template inheritance
- 6 placeholders must be replaced correctly across 15+ files — error surface scales linearly
- Template files contain valid Python with `{{PLACEHOLDER}}` markers that break syntax highlighting

### Neutral

- Adding a new placeholder requires updating templates, defaults.yaml, init.md, and helios-init.sh

## References

- `templates/` directory — all template files
- `config/defaults.yaml` — placeholder definitions and descriptions
- `commands/init.md` — scaffolding instructions
- `bin/helios-init.sh` — CLI scaffolding script
