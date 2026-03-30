# ADR-0002: Separate Hypothesis Space from Measurement Apparatus

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-31 |
| **Author** | Prannaya Gupta |
| **Supersedes** | (none) |
| **Category** | Architecture Pattern |

## Context

Helios automates the ML experiment loop: an AI agent iteratively modifies training code, runs experiments, and decides whether to keep or discard the results. The fundamental risk of this automation is that an agent with access to both the training code *and* the evaluation code can — deliberately or through optimization pressure — game its own metrics.

This is not a theoretical concern. In reinforcement learning, reward hacking is a well-documented phenomenon: agents find ways to maximize the reward signal without achieving the intended objective. In the Helios context, the equivalent risk is an agent that modifies `evaluate.py` to make its results look better — changing what "accuracy" means between experiments, overfitting the evaluation function, or introducing subtle bugs that inflate metrics.

The deeper epistemological principle is older than software: the entity that generates hypotheses must not be the entity that evaluates them. This is the foundation of the double-blind protocol in experimental science, the separation of prosecution and judiciary in law, and the principle of independent auditing in finance.

## Options Considered

### Option 1: Architectural Separation — Immutable Evaluation

Enforce a hard boundary: `prepare.py` and `evaluate.py` are READ-ONLY to the agent. `train.py` and `config.yaml` are the only modifiable files. The boundary is enforced by:
- Agent instructions (program.md, loop-protocol.md)
- File access documentation in every template header
- The evaluator agent having no Write/Edit tools at all

Trade-offs: reduces agent flexibility. The agent cannot fix bugs in evaluation code or adapt the evaluation to new requirements.

### Option 2: Soft Boundary — Trust the Agent

Allow the agent to modify all files but instruct it not to change evaluation code unless explicitly asked.

Trade-offs: simpler to implement. But instruction-following is probabilistic — the agent may violate the constraint under optimization pressure or through accumulated prompt drift.

### Option 3: Sandboxed Evaluation — Runtime Enforcement

Run evaluation in a separate sandbox/container where the agent has no write access. Enforce the boundary at the OS level, not the prompt level.

Trade-offs: technically robust but operationally complex. Requires container orchestration for what should be a lightweight CLI tool.

### Option 4: Checksummed Evaluation — Detect Modification

Allow the agent to modify evaluation files but compute checksums before and after each experiment. Flag any changes.

Trade-offs: detects violations after the fact but doesn't prevent them. The damage (invalid experiment comparisons) has already occurred.

## Decision

**We will enforce an architectural separation between hypothesis space (train.py, config.yaml) and measurement apparatus (prepare.py, evaluate.py)** because this is the load-bearing invariant that makes experiment comparisons valid, and prompt-level enforcement is sufficient for the current trust model where the agent operates under human supervision.

## Rationale

The separation maps directly to the structure of controlled experiments in science: the experimental variable (model code, hyperparameters) is the hypothesis; the measurement protocol (data splits, metrics computation) is the apparatus. Changing both simultaneously invalidates the experiment.

Prompt-level enforcement is a deliberate choice over runtime enforcement. Helios runs in Claude Code, where the agent operates under human oversight — the user can see every file modification. The architectural boundary serves as a cognitive guardrail for the agent and a verifiable invariant for the human. If Helios were ever deployed in a fully unsupervised setting, this decision should be revisited (see ADR-0003 on capability boundaries).

The specific enforcement points are:
1. Template headers: every file is labeled as `MEASUREMENT APPARATUS — READ-ONLY` or `HYPOTHESIS SPACE — AGENT-EDITABLE`
2. Agent instructions: `program.md` and `loop-protocol.md` state the constraint as non-negotiable
3. Access control table: `loop-protocol.md` contains a structured table mapping files to access levels
4. Evaluator agent: has no Write/Edit tools, providing a structural (not just instructional) guarantee for analysis tasks

## Consequences

### Positive

- All experiment comparisons are valid — the metric definition cannot change between experiments
- The invariant is simple enough to verify by inspection: check if prepare.py or evaluate.py were modified
- Aligns with established scientific methodology (double-blind protocols)
- The README can make strong claims about experimental integrity

### Negative

- The agent cannot fix legitimate bugs in evaluation code — requires human intervention
- Adapting the evaluation to new metrics or data formats requires manual changes to READ-ONLY files
- The boundary is enforced by convention/instruction, not by runtime access control — a sufficiently confused agent could violate it

### Neutral

- Feature engineering is a gray area: `featurizers.py` is READ-ONLY but the agent modifies how `train.py` uses it — the boundary is at the call site, not the implementation

## References

- [Double-Blind Protocol](https://en.wikipedia.org/wiki/Blinded_experiment) — experimental methodology
- [Reward Hacking](https://arxiv.org/abs/1711.09043) — Amodei et al., "Concrete Problems in AI Safety"
- [Goodhart's Law](https://en.wikipedia.org/wiki/Goodhart%27s_law) — "When a measure becomes a target, it ceases to be a good measure"
- Richard Feynman, "Cargo Cult Science" (Caltech, 1974) — on not fooling yourself
- `commands/rules/loop-protocol.md` — the access control matrix
- `templates/evaluate.py` header — "MEASUREMENT APPARATUS" label
- `templates/train.py` header — "HYPOTHESIS SPACE" label
