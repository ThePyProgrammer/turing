# ADR-0001: Use ADRs for Architectural Decisions

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-31 |
| **Author** | Prannaya Gupta |
| **Supersedes** | (none) |
| **Category** | Process |

## Context

Turing is built on several non-obvious architectural decisions — the separation between hypothesis space and measurement apparatus, the two-agent capability boundary, the TOML config DSL, the patience-based convergence protocol. These decisions are load-bearing: changing any of them would require rethinking significant portions of the system.

Currently, these decisions are documented implicitly — scattered across the README, agent definitions, and code comments. A new contributor reading the codebase must reconstruct the rationale from fragments. Worse, there is no mechanism to detect when a code change violates one of these decisions, or to revisit a decision when circumstances change.

This is the standard failure mode of undocumented architecture: decisions made invisibly become permanent by accident rather than by choice.

## Options Considered

### Option 1: Architecture Decision Records (ADRs)

Lightweight, structured documents — one per decision — stored in `docs/adr/` with a formal lifecycle (Proposed → Accepted → Deprecated/Superseded). Each ADR records the context, options considered, decision, rationale, and consequences.

Trade-offs: requires discipline to maintain. Can become stale if not reviewed periodically.

### Option 2: Inline Code Comments

Document decisions where they're implemented — in the source files themselves.

Trade-offs: decisions are fragmented across files. No index. No lifecycle. No way to see the full decision landscape. Comments rot faster than standalone documents because they're invisible during code review.

### Option 3: Wiki / External Documentation

Use a wiki, Notion, or Confluence for architecture documentation.

Trade-offs: external to the repo, so it diverges from the code. Not version-controlled with the same rigor. Not auditable by automated tools.

### Option 4: Status Quo / Do Nothing

Continue with decisions embedded in README.md and code comments.

Trade-offs: new contributors must reconstruct rationale by reading everything. No formal mechanism for revisiting decisions. No way to detect violations.

## Decision

**We will use Architecture Decision Records** because they keep decisions co-located with the code they govern, provide a formal lifecycle for revisiting decisions when circumstances change, and enable automated compliance checking via `/blueprint:audit`.

## Rationale

ADRs were introduced by Michael Nygard in 2011 as a lightweight alternative to heavyweight architecture documentation. The key insight is that decisions — not diagrams or component lists — are the most valuable architectural artifact. A decision captures *why* the system is the way it is, not just *what* it is.

Turing has a small but dense decision surface: 10 decisions that deeply constrain the system's behavior. This is exactly the scale where ADRs provide maximum value — few enough to maintain rigorously, impactful enough to warrant formal documentation.

## Consequences

### Positive

- Every architectural decision has a single, discoverable location
- Decisions can be challenged, reviewed, and superseded through a formal process
- New contributors can read the ADR index to understand the system's constraints
- Automated tools (`/blueprint:audit`) can verify codebase compliance

### Negative

- Adds maintenance burden: new decisions must be documented, old ones reviewed
- Risk of stale ADRs if the process is abandoned

### Neutral

- Shifts documentation effort from scattered comments to structured records

## References

- [ADR GitHub Organization](https://adr.github.io/) — Nygard's original proposal
- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) — Nygard, 2011
- blueprint plugin — the ADR tooling used by this project
