---
name: report
description: Generate a markdown research report from experiment history — structured for sharing, archiving, or including in documentation. More detailed than a brief, less visual than a poster.
disable-model-invocation: true
argument-hint: "[--since YYYY-MM-DD] [--output path]"
allowed-tools: Read, Bash(python scripts/*:*, source .venv/bin/activate:*, mkdir:*), Grep, Glob
---

Generate a structured markdown research report summarizing the experiment campaign.

## Steps

### 1. Generate the Report

Use the logbook generator in markdown mode as the data backbone:

```bash
source .venv/bin/activate && python scripts/generate_logbook.py --format markdown
```

Also gather supplementary data:
```bash
source .venv/bin/activate && python scripts/generate_brief.py
cat experiment_state.yaml 2>/dev/null || true
cat RESEARCH_PLAN.md 2>/dev/null || true
```

### 2. Enhance with Analysis

The logbook generator produces raw data. Enhance it with your analysis to create a proper report. Add these sections that the script doesn't generate:

- **Executive Summary** (2-3 sentences): What was the task? What's the best result? Is it good enough?
- **Approach:** Describe the methodology — autoresearch loop, evaluation strategy, search strategy used
- **Key Findings:** Synthesize patterns from the experiment log:
  - Which model families outperformed others?
  - What hyperparameter ranges work vs don't?
  - Were there surprising results?
  - What failure patterns emerged?
- **Recommendations:** Based on the findings, what should be tried next? What should be avoided?
- **Limitations:** What wasn't explored? What constraints affected the results?

### 3. Output

If `$ARGUMENTS` contains `--output <path>`:
```bash
mkdir -p $(dirname <path>)
```
Write the report to the specified path.

Otherwise, display the report directly.

**Common usage:**
```
/turing:report --output reports/campaign-v1.md
/turing:report --since 2026-03-15 --output reports/week-12.md
```

## Report Structure

```markdown
# Research Report: <task description>
Generated: <date>

## Executive Summary
<2-3 sentences>

## Methodology
<approach, evaluation strategy, convergence criteria>

## Campaign Summary
<table: experiments, keep rate, best metric, timespan>

## Improvement Trajectory
<table: experiment-by-experiment metric progression>

## Key Findings
<synthesized patterns from experiment history>

## Model Comparison
<table: model families, experiments per family, best metric, keep rate>

## Hypothesis Analysis
<what was proposed, by whom, what worked>

## Recommendations
<concrete next steps>

## Limitations
<what wasn't tried, constraints>
```

## When to Use

- End of a research campaign for archiving
- Before a team review or status update
- To document findings for a paper or thesis
- To hand off a project to another researcher
