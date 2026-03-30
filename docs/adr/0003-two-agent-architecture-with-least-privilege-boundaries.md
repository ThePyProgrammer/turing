# ADR-0003: Two-Agent Architecture with Least-Privilege Capability Boundaries

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-31 |
| **Author** | Prannaya Gupta |
| **Supersedes** | (none) |
| **Category** | Architecture Pattern |

## Context

Helios needs to perform two fundamentally different kinds of work:

1. **Research**: modify training code, run experiments, decide whether to keep or discard results. This requires Write, Edit, and Bash capabilities and is inherently risky — every modification could break the pipeline.

2. **Analysis**: read experiment logs, compare runs, assess convergence, provide insights. This requires only Read and Bash capabilities and is inherently safe — no code can be modified.

Combining both capabilities in a single agent creates an unnecessary risk surface. An agent asked to "analyze the last 5 experiments" should not be able to accidentally modify `train.py` while doing so. The principle of least privilege — each entity should have exactly the capabilities needed for its role and no more — suggests separating these roles.

## Options Considered

### Option 1: Two Specialized Agents

`@ml-researcher` with Read/Write/Edit/Bash/Grep/Glob (200 max turns) and `@ml-evaluator` with Read/Bash/Grep/Glob only (50 max turns). The researcher delegates analysis tasks to the evaluator.

Trade-offs: slight overhead from agent delegation. The evaluator cannot fix issues it discovers.

### Option 2: Single Agent with Mode Switching

One agent that operates in "research mode" (all tools) or "analysis mode" (read-only). Mode is set by the command that invoked it.

Trade-offs: simpler deployment. But mode enforcement is prompt-level — a confused agent might use write tools in analysis mode.

### Option 3: Single Agent, All Capabilities

One agent with all tools, relied upon to use them appropriately based on the task.

Trade-offs: simplest architecture. But no structural safety guarantee for analysis operations.

## Decision

**We will use two specialized agents with distinct capability sets** because the tool-level enforcement provides a structural guarantee that analysis operations cannot modify code, and this guarantee is more robust than instruction-level enforcement.

## Rationale

The two-agent architecture operationalizes ADR-0002 (hypothesis-measurement separation) at the agent level. The researcher can modify the hypothesis space but should delegate measurement and analysis to the evaluator, which structurally cannot modify anything.

The evaluator's read-only constraint has an epistemological benefit beyond safety: an analyst who cannot act on their observations makes more trustworthy observations. This is the same principle behind separating audit from operations in financial systems — the auditor's credibility depends on their inability to change the books.

Max turns are deliberately asymmetric: the researcher gets 200 turns for long experiment loops; the evaluator gets 50 turns for focused analysis tasks.

## Consequences

### Positive

- Analysis operations (status, compare) are structurally safe — the evaluator has no Write/Edit tools
- Each agent's prompt can be focused on its specific role without conflicting instructions
- The researcher can delegate analysis to the evaluator, keeping its own context focused on experiments
- Clear separation of concerns makes each agent simpler to maintain

### Negative

- Inter-agent delegation adds latency and context overhead
- The evaluator cannot fix issues it discovers — must report back to the researcher or user
- Two agent definitions to maintain instead of one

### Neutral

- Both agents share the same config files and project structure — no data duplication

## References

- [Principle of Least Privilege](https://en.wikipedia.org/wiki/Principle_of_least_privilege) — Saltzer & Schroeder, 1975
- [Separation of Duties](https://en.wikipedia.org/wiki/Separation_of_duties) — financial controls principle
- ADR-0002 — hypothesis-measurement separation (this is the agent-level implementation)
- `agents/ml-researcher.md` — researcher agent definition
- `agents/ml-evaluator.md` — evaluator agent definition
