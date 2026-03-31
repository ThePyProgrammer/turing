---
name: brief
description: Generate a structured research intelligence report from experiment history — what's been learned, what's promising, what's exhausted, and what the human should consider next. Use --deep for literature-grounded suggestions.
disable-model-invocation: true
argument-hint: "[ml/project] [--deep]"
allowed-tools: Read, Bash(python scripts/*:*, source .venv/bin/activate:*), Grep, Glob, WebSearch, WebFetch
---

Generate a research briefing that a human can read in 2 minutes and immediately decide what to inject next.

## Project Detection

Before generating the briefing, detect which project to report on:

0. **Detect project directory:**
   - If `$ARGUMENTS` contains a path (e.g., `ml/coding`), use that as the project directory
   - Else if cwd contains `config.yaml` and `train.py`, use cwd
   - Else search for `ml/*/` subdirectories containing `config.yaml`
     - If exactly one found, use it
     - If multiple found, list them and ask the user which to report on
   - All subsequent commands run from the detected project directory

## Steps

1. **Generate the briefing:**
   ```bash
   source .venv/bin/activate && python scripts/generate_brief.py
   ```

2. **Self-critique the briefing** before presenting. Review the generated output and check:
   - **Recommendations specificity:** Are they concrete enough to act on? "Try a different model" is bad. "Try LightGBM with leaf-wise growth because exp-004 showed depth sensitivity" is good. If vague, rewrite them with specific model/hyperparameter suggestions grounded in the experiment data.
   - **Exhausted directions coverage:** Cross-reference the "Model Types Explored" section against `experiments/log.jsonl`. Are there discarded experiments missing from the summary? If so, add them.
   - **Convergence estimate grounding:** If the briefing says "close to convergence" or "further improvement possible", verify against the actual metric trajectory. Is the claim supported by the numbers?
   - **Metric accuracy:** Spot-check that the "Current Best" metrics match the actual log. Run `python scripts/show_metrics.py --last 1` if uncertain.

   If any section fails the check, regenerate just that section. Max 1 revision round — don't over-polish.

3. **Present the output** to the user. The briefing has 6 sections:
   - **Campaign Summary** — total experiments, keep rate, timespan
   - **Current Best** — model type, metrics, experiment ID, configuration
   - **Improvement Trajectory** — metric over time, rate of improvement
   - **Model Types Explored** — which approaches have been tried and their hit rates
   - **Hypothesis Queue** — pending and completed hypotheses
   - **Recommendations** — data-driven next steps

4. **If `$ARGUMENTS` contains `--deep`:** run the Literature-Grounded Suggestions step below.

5. **Prompt for action:**
   - "Want to inject a hypothesis? Use `/turing:try <idea>`"
   - "Want to continue training? Use `/turing:train`"
   - "Want literature-backed suggestions? Use `/turing:brief --deep`"

## Literature-Grounded Suggestions (--deep flag)

When `--deep` is requested, add a 7th section: **Literature-Grounded Suggestions**.

### Steps:

1. **Read context:** Read `config.yaml` and the briefing output to understand:
   - What task type this is (tabular classification, time series, etc.)
   - Which model families have been exhausted (from "Model Types Explored")
   - Where improvement has plateaued (from "Improvement Trajectory")
   - What failure patterns keep recurring

2. **Search literature** with `WebSearch` for techniques that address the specific stagnation:
   - If plateaued: "improve [task type] accuracy beyond [current metric] 2024"
   - If overfitting: "regularization techniques [model family] [task type]"
   - If all models tried: "state of the art [task type] benchmark 2024 2025"

3. **Distill 3-5 suggestions** from the literature, each with:
   - **Technique:** specific and actionable
   - **Source:** paper or article URL
   - **Why now:** how it addresses the specific stagnation point
   - **Impact estimate:** high/medium/low
   - **Complexity:** low/medium/high

4. **Queue suggestions** as hypotheses:
   ```bash
   source .venv/bin/activate && python scripts/manage_hypotheses.py add "<technique>: <rationale> (source: <citation>)" --priority medium --source literature
   ```

5. **Format as a section** appended to the briefing.

## Saving Briefs

```bash
mkdir -p briefs && python scripts/generate_brief.py > briefs/brief-$(date +%Y-%m-%d).md
```

## When to Use

- After a training session completes or converges
- Before injecting new hypotheses (to understand what's already been tried)
- When returning to a project after time away
- **With `--deep`:** when the agent seems stuck and you want evidence-based direction
