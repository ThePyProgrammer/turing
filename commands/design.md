---
name: design
description: Generate a structured experiment design for a queued hypothesis. Searches literature for methodology, produces a scored design document with implementation guidance, and saves to experiments/designs/.
disable-model-invocation: true
argument-hint: "<hypothesis-id or description>"
allowed-tools: Read, Write, Bash(python scripts/*:*, source .venv/bin/activate:*, mkdir:*), Grep, Glob, WebSearch, WebFetch
---

Front-load the thinking before the coding. Given a hypothesis, produce a structured experiment design grounded in methodology from the literature.

## Steps

### 1. Load the Hypothesis

If `$ARGUMENTS` matches `hyp-NNN`, load the hypothesis detail:
```bash
source .venv/bin/activate && python scripts/manage_hypotheses.py show $ARGUMENTS
```

If `$ARGUMENTS` is freeform text, use it directly as the hypothesis description.

Also read the current config for context:
```bash
cat config.yaml
```

### 2. Review Experiment History

Check what's been tried before:
```bash
source .venv/bin/activate && python scripts/show_metrics.py --last 10
```

Read the experiment state for context on what worked and what didn't:
```bash
cat experiment_state.yaml 2>/dev/null || echo "No experiment state yet"
```

### 3. Search for Methodology

Use `WebSearch` to find 2-3 papers or articles describing how to implement this type of change effectively. Search for:
- The specific technique mentioned in the hypothesis
- Best practices for this type of experiment
- Common pitfalls and how to avoid them

Example queries:
- "LightGBM vs XGBoost tabular data benchmark 2024"
- "feature engineering best practices time series classification"
- "regularization techniques prevent overfitting gradient boosting"

Use `WebFetch` on the most relevant results to extract specific methodology details.

### 4. Generate the Design Document

Create a structured design at `experiments/designs/<hyp-id>-design.md` (or `experiments/designs/adhoc-<timestamp>-design.md` for freeform hypotheses):

```markdown
# Experiment Design: <hypothesis summary>

## Hypothesis
<full hypothesis description>

## Objective
<what we're testing, stated as a falsifiable claim>

## Method
<specific changes to make, grounded in literature>

## Literature Support
- <paper/source 1>: <what it says about this approach>
- <paper/source 2>: <what it says>

## Implementation Plan
### Changes to train.py
<specific code changes needed, as concrete as possible>

### Changes to config.yaml (if any)
<hyperparameter changes>

## Expected Outcome
- **Success looks like:** <metric improvement, specific threshold>
- **Failure looks like:** <what would disprove the hypothesis>

## Risks
- <potential pitfalls identified from literature>

## Estimated Runs
<how many experiment iterations this design requires>

## Quality Score
- Feasibility: <0-10>
- Novelty: <0-10>
- Clarity: <0-10>
```

### 5. Self-Critique

Review the design you just wrote:
- Is the implementation plan specific enough that the researcher agent can execute it without ambiguity?
- Does the expected outcome have a concrete metric threshold?
- Are the risks actionable (not just "it might not work")?

If any dimension scores below 7/10, revise the relevant section. Maximum 2 revision rounds.

### 6. Report

```bash
mkdir -p experiments/designs
```

After writing the design file, display:
```
Experiment Design: <hypothesis summary>
===
Scores: Feasibility=X/10, Novelty=Y/10, Clarity=Z/10
Sources: N papers/articles consulted
File: experiments/designs/<filename>

The researcher agent can read this design during /turing:train.
```

## Integration

- Design files live in `experiments/designs/` and are readable by `@ml-researcher`
- The researcher agent can optionally read the design before editing `train.py`
- Update `program.md` reference: the HYPOTHESIZE step can check for designs
