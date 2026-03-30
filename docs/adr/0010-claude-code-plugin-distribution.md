# ADR-0010: Claude Code Plugin Distribution with npm Installer

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-31 |
| **Author** | Prannaya Gupta |
| **Supersedes** | (none) |
| **Category** | Deployment Strategy |

## Context

Turing is a Claude Code plugin consisting of markdown commands, markdown agent definitions, TOML/YAML config, Python templates, and shell scripts. It must be distributable to users who may install it via:

1. Direct path: `claude plugin add /path/to/turing`
2. npm: `npm install -g claude-turing`
3. Claude Code marketplace (future)

The distribution must deploy commands to `~/.claude/commands/turing/`, agents to `~/.claude/agents/turing/`, and templates to an accessible location. It must also update the user's CLAUDE.md with a command reference.

## Options Considered

### Option 1: npm Package with Node.js Installer

Distribute as an npm package. Include `src/install.js` (deploys files to `~/.claude/`), `src/verify.js` (checks installation), and `src/postinstall.js` (shows setup instructions). CLI entry point via `bin/cli.sh`.

Trade-offs: npm is the standard distribution mechanism for Node.js tools. The installer can be run independently of npm. But requires Node.js.

### Option 2: Pure Shell Script Distribution

A single `install.sh` that copies files to the right locations.

Trade-offs: no Node.js dependency. But shell scripts are harder to make cross-platform (macOS vs Linux vs WSL), and lack the package management features of npm (versioning, updating, dependency resolution).

### Option 3: Python Package (pip/pipx)

Distribute as a Python package since the templates are Python.

Trade-offs: aligns with the ML ecosystem. But Turing is a Claude Code plugin, not a Python library — the primary consumers are Claude Code commands, not Python imports. pip distribution would confuse the packaging semantics.

### Option 4: Direct Git Clone Only

Users clone the repo and add the plugin via path.

Trade-offs: simplest. But no version management, no automated updates, and the user must remember the clone path.

## Decision

**We will distribute via npm with a Node.js installer and support direct path installation** because npm provides version management and global installation, the installer handles the mechanical deployment to `~/.claude/`, and direct path installation serves development and CI use cases.

## Rationale

The Claude Code plugin ecosystem is Node.js-based (`.claude-plugin/plugin.json`, `package.json`). npm is the natural distribution channel. The three-file installer (`install.js`, `verify.js`, `postinstall.js`) follows the pattern established by the blueprint plugin.

The installer manages a section in CLAUDE.md using idempotent markers (`<!-- turing:managed-start -->` / `<!-- turing:managed-end -->`), allowing repeated installations to update rather than duplicate the command reference.

`bin/cli.sh` provides a unified CLI (`claude-turing install/verify/init/help`) for npm consumers, while `bin/turing-init.sh` provides direct scaffolding for non-Claude-Code usage.

## Consequences

### Positive

- `npm install -g claude-turing` provides one-command global installation
- `claude-turing verify` confirms installation completeness
- CLAUDE.md section is managed with idempotent markers — safe to re-run
- Version management via npm (semver, `npm update`)
- Both global (`~/.claude/`) and local (`./.claude/`) installation supported

### Negative

- Requires Node.js >= 18
- npm packaging overhead (package.json, package-lock.json, node_modules)
- The installer is JavaScript, but the templates are Python — two languages in the project

### Neutral

- `.claude-plugin/plugin.json` serves both npm metadata and Claude Code plugin registration

## References

- [Claude Code Plugin System](https://docs.anthropic.com/en/docs/claude-code/plugins) — plugin specification
- `src/install.js` — installer implementation
- `src/verify.js` — verification script
- `src/postinstall.js` — npm postinstall hook
- `bin/cli.sh` — unified CLI entry point
- blueprint plugin — precedent for npm + installer pattern
