---
name: design
description: Generate a structured experiment design for a hypothesis. Reads experiment history, searches literature for methodology, produces a scored design document at experiments/designs/.
argument-hint: "<hypothesis-id or description>"
allowed-tools: Read, Write, Bash(python scripts/*:*, source .venv/bin/activate:*, mkdir:*), Grep, Glob, WebSearch, WebFetch
---

Front-load the thinking before the coding. Given a hypothesis, produce a structured experiment design grounded in methodology from the literature.

## Steps

### 1. Load Context

If `$ARGUMENTS` matches `hyp-NNN`, load the hypothesis:
```bash
source .venv/bin/activate && python scripts/manage_hypotheses.py show $ARGUMENTS
```

If freeform text, use it directly as the hypothesis description.

Read the current config and experiment state:
```bash
cat config.yaml
```
```bash
source .venv/bin/activate && python scripts/show_metrics.py --last 10 2>/dev/null || echo "No experiments yet"
```
```bash
cat experiment_state.yaml 2>/dev/null || echo "No experiment state yet"
```

### 2. Search for Methodology

Use `WebSearch` to find 2-3 papers or articles describing how to implement the proposed change effectively. Target:
- The specific technique in the hypothesis (e.g., "LightGBM dart boosting implementation best practices")
- Common pitfalls for this type of change
- Benchmark results showing expected improvement range

Use `WebFetch` on the most relevant results to extract specific methodology details: hyperparameter recommendations, training procedures, evaluation approaches.

### 3. Write the Design Document

Create `experiments/designs/<hyp-id>-design.md` (or `experiments/designs/adhoc-<date>-design.md` for freeform hypotheses):

```bash
mkdir -p experiments/designs
```

Write with this structure:

```markdown
# Experiment Design: <hypothesis summary>

## Hypothesis
<full description>

## Objective
<what we're testing, stated as a falsifiable claim>

## Method
<specific changes, grounded in literature findings>

## Literature Support
- <source 1>: <what it says about this approach>
- <source 2>: <relevant finding>

## Implementation Plan
### Changes to train.py
<concrete code changes needed>

### Changes to config.yaml (if any)
<hyperparameter values to set, with rationale from literature>

## Expected Outcome
- **Success:** <metric > threshold, specific number>
- **Failure:** <what would disprove the hypothesis>

## Risks
<specific pitfalls from literature, not generic "might not work">

## Estimated Runs
<how many iterations>
```

### 4. Self-Critique

Review the design:
- Is the implementation plan specific enough for the researcher agent to execute without ambiguity?
- Does the expected outcome have a concrete metric threshold?
- Are risks actionable?

Score each dimension 1-10 (feasibility, novelty, clarity). If any < 7, revise that section. Max 2 revision rounds.

### 5. Report

Display the design summary with scores and file location. The researcher agent can read the design during `/turing:train`.
