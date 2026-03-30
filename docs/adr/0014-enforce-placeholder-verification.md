# ADR-0014: Enforce Placeholder Substitution Verification After Scaffolding

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-31 |
| **Author** | Prannaya Gupta |
| **Supersedes** | (none) |
| **Category** | Architecture Pattern |

## Context

The `{{PLACEHOLDER}}` template system (ADR-0008) spans 15+ files with 6 distinct placeholders. There is no automated verification that all placeholders were replaced after scaffolding. A missed placeholder produces valid Python that silently does the wrong thing:

- `primary_metric = eval_cfg.get("primary_metric", "{{TARGET_METRIC}}")` — legal Python string, will never match any real metric
- `source: "{{DATA_SOURCE}}"` in config.yaml — legal YAML string, `prepare.py` will raise FileNotFoundError with an unhelpful message

The CLI path (`bin/turing-init.sh`) doesn't perform substitution at all — it copies raw templates and tells the user to replace manually. The Claude Code path (`commands/init.md`) instructs the LLM to replace placeholders, which is probabilistic.

ARCHITECTURE.md mentions "unreplaced placeholders are detectable by grepping for `{{`" but this grep is never automated.

## Options Considered

### Option 1: Post-Scaffolding Verification Script

Add `scripts/verify_placeholders.py` that greps all scaffolded files for `{{` and reports any unreplaced markers. Both scaffolding paths (`init.md` and `turing-init.sh`) call it as a final step. Fail loudly if any remain.

Trade-offs: simple, effective. Converts silent-wrong-behavior into loud-and-immediate failure.

### Option 2: Template Engine

Replace the ad-hoc `{{PLACEHOLDER}}` system with a proper template engine (Jinja2, Mustache). The engine validates that all required variables are provided before rendering.

Trade-offs: more robust. But adds a dependency, adds complexity, and changes the scaffolding architecture significantly.

### Option 3: Pre-Substitution Manifest

Add a `templates/manifest.yaml` listing every placeholder and which files contain it. The scaffolding scripts validate against the manifest.

Trade-offs: makes the placeholder system explicit and verifiable. But adds a manifest file that must be kept in sync with templates.

## Decision

**We will add a post-scaffolding verification script** that greps for unreplaced `{{` markers and fails loudly. Both scaffolding paths will call it as a final step. This is the simplest change that eliminates the silent failure mode.

## Rationale

The placeholder system is simple and works well — the problem is not the substitution mechanism but the lack of verification. A 20-line Python script that `grep -r '{{' .` over the scaffolded directory and exits non-zero if any matches are found converts every silent placeholder failure into an immediate, actionable error.

## Consequences

### Positive

- Unreplaced placeholders are caught immediately, not at runtime
- Both CLI and Claude Code paths get the same verification
- The verification script is itself testable

### Negative

- Adds one more script to `templates/scripts/`
- False positives if user code legitimately contains `{{` (e.g., Jinja templates) — mitigated by only checking known template files

### Neutral

- Does not change the substitution mechanism itself

## References

- Architecture Evaluation Report (2026-03-31) — bug surface dimension flagged silent placeholder failures
- ADR-0008 — template-based project scaffolding
- `commands/init.md` — Claude Code scaffolding path
- `bin/turing-init.sh` — CLI scaffolding path (does not substitute)
