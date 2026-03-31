# Contributing to Turing

Conventions a second contributor needs to know. For *why* things are the way they are, see `docs/adr/`. This document covers *how* to add things correctly.

## Quick Start

```bash
git clone https://github.com/ThePyProgrammer/turing.git
cd turing

# Node.js (installer layer)
npm install

# Python (templates + tests)
pip install -e ".[test]"
pytest
```

Requirements: Node.js >= 18, Python >= 3.12.

## Adding a New Command

Commands are markdown skill files in `commands/`. Each command is a thin dispatcher -- no business logic, no state.

### Checklist

- [ ] Create `commands/<name>.md` with required frontmatter:
  ```yaml
  ---
  name: <name>
  description: One-line description of what this command does.
  disable-model-invocation: true
  argument-hint: "[args]"
  allowed-tools: Read, Write, Edit, Bash(...), Grep, Glob
  ---
  ```
  See `commands/train.md` for a complete example.

- [ ] Add routing entry to `commands/turing.md`:
  - Add a row to the **Routing Table** with user intent phrases
  - Add a row to the **Sub-commands** table with purpose and agent

- [ ] Add to `src/install.js` `SUB_COMMANDS` array (line ~22):
  ```js
  const SUB_COMMANDS = [
    "init", "train", ..., "<name>",
  ];
  ```

- [ ] Add to `src/verify.js` `EXPECTED_COMMANDS` array (line ~15):
  ```js
  "<name>/SKILL.md",
  ```

- [ ] Update `docs/ARCHITECTURE.md` codemap -- add the new command under the `commands/` section with a one-line description.

## Adding a New Script

Scripts live in `templates/scripts/` and are scaffolded into user projects by `/turing:init`.

### Standard Pattern

Every script follows this structure:

```python
#!/usr/bin/env python3
"""Module docstring explaining purpose and usage.

Usage:
    python scripts/<name>.py [--flag value]
"""

from __future__ import annotations

import argparse
# ... other imports ...


def main_logic(...):
    """Pure function containing the business logic."""
    ...


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="...")
    # ... args ...
    args = parser.parse_args()
    # ... call main_logic ...


if __name__ == "__main__":
    main()
```

Key points: shebang line, docstring with usage, `from __future__ import annotations`, argparse for CLI, logic separated from CLI wiring, `if __name__` guard.

### Checklist

- [ ] Create `templates/scripts/<name>.py` following the standard pattern above
- [ ] Add to `templates/scripts/scaffold.py` `TEMPLATE_DIRS["scripts"]` list (line ~55)
- [ ] Add tests in `tests/test_<name>.py`:
  - Import from `scripts.<name>` (pytest is configured with `pythonpath = ["templates"]`)
  - Test the pure logic functions, not the CLI wiring
  - Include docstring referencing the relevant ADR if applicable
- [ ] Update `docs/ARCHITECTURE.md` codemap under `templates/scripts/`

## Adding a New Config File

Config format is governed by [ADR-0004](docs/adr/0004-toml-config-dsl-for-domain-knowledge.md):

| Format | Use for | Examples |
|--------|---------|---------|
| **TOML** | System-wide domain knowledge (lifecycle states, taxonomies, relationships) | `lifecycle.toml`, `taxonomy.toml`, `state.toml` |
| **YAML** | Project-specific parameters (hyperparameters, data paths, defaults) | `defaults.yaml`, `experiment_archetypes.yaml` |

### Checklist

- [ ] Create `config/<name>.toml` or `config/<name>.yaml` per the format rules above
- [ ] Add to `src/install.js` `CONFIG_FILES` array (line ~75):
  ```js
  const CONFIG_FILES = [
    "defaults.yaml", "lifecycle.toml", ..., "<name>.toml",
  ];
  ```
- [ ] Add to `src/verify.js` `EXPECTED_CONFIG` array (line ~35):
  ```js
  "<name>.toml",
  ```
- [ ] Update `docs/ARCHITECTURE.md` codemap under `config/`

## Testing

```bash
pytest                     # run all tests
pytest tests/test_foo.py   # run one file
pytest -x                  # stop on first failure
```

### Setup

Defined in `pyproject.toml`:
- `testpaths = ["tests"]`
- `pythonpath = ["templates"]` -- template modules are importable as `from scripts.foo import bar`

### Conventions

- **File naming:** `tests/test_<module_name>.py` mirrors `templates/scripts/<module_name>.py`
- **Docstrings:** Test files start with a docstring explaining what ADR or invariant they verify
- **Fixtures:** Shared fixtures live in `tests/conftest.py` (`tmp_config`, `tmp_log`, `make_experiment_entry`, `write_experiments`)
- **Tiers** (from [ADR-0011](docs/adr/0011-establish-testing-strategy.md)):
  - Tier 1: Measurement apparatus, convergence detection
  - Tier 2: Experiment logging, sweep queue
  - Tier 3: Data pipeline, featurizers
  - Tier 4: Anti-pattern tests (invariant enforcement)
  - Tier 5: Installer and CLI
- **Anti-pattern tests:** `tests/test_anti_patterns.py` enforces structural invariants (e.g., evaluator agent must not have Write tool). Add invariant checks here when a new ADR introduces one.

## ADR Process

Architecture Decision Records live in `docs/adr/`. See [ADR-0001](docs/adr/0001-use-adrs-for-architectural-decisions.md) and the [ADR README](docs/adr/README.md) for full lifecycle details.

### When to Write an ADR

Write an ADR when you are making a decision that:
- Changes a public interface or file format
- Introduces a new dependency or tool
- Establishes a pattern other code must follow
- Has trade-offs worth documenting for future contributors

### How to Propose One

1. Copy `docs/adr/template.md` to `docs/adr/NNNN-<slug>.md` (next available number)
2. Set status to **Proposed**
3. Fill in Context, Options Considered, Decision, and Consequences
4. Open a PR -- the ADR will be reviewed before acceptance

### Principles

1. One decision per ADR
2. Immutable history -- supersede, don't edit accepted ADRs
3. Honest consequences -- every decision has downsides, document them
4. Evidence over opinion -- link to benchmarks or prior art

## Commit Message Format

This repo uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short description> (#issue)

Optional longer explanation.
```

### Types

| Type | When |
|------|------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `test` | Adding or updating tests |
| `docs` | Documentation only |
| `refactor` | Code change that neither fixes a bug nor adds a feature |

### Examples from This Repo

```
feat: add experiment archetype library (8 structured strategies)
fix: sync install.js with all 14 commands and 8 config files (blocker #1-2)
test: add 13 tests for prepare.py -- measurement apparatus foundation (blocker #5)
docs: update release blockers -- all MUST FIX items resolved
refactor: create shared turing_io module for duplicated data loaders
```

Keep the first line under 72 characters. Reference issue numbers where applicable.
