---
name: brief
description: Generate a structured research intelligence report from experiment history — what's been learned, what's promising, what's exhausted, and what the human should consider next. Closes the taste-leverage loop. Use --deep for literature-grounded suggestions.
disable-model-invocation: true
argument-hint: "[--deep]"
allowed-tools: Read, Bash(python scripts/*:*, source .venv/bin/activate:*), Grep, Glob, WebSearch, WebFetch
---

Generate a research briefing that a human can read in 2 minutes and immediately decide what to inject next.

## Steps

1. **Generate the briefing:**
   ```bash
   source .venv/bin/activate && python scripts/generate_brief.py
   ```

2. **Present the output** to the user. The briefing has 6 sections:
   - **Campaign Summary** — total experiments, keep rate, timespan
   - **Current Best** — model type, metrics, experiment ID, configuration
   - **Improvement Trajectory** — metric over time, rate of improvement
   - **Model Types Explored** — which approaches have been tried and their hit rates
   - **Hypothesis Queue** — pending and completed hypotheses
   - **Recommendations** — data-driven next steps

3. **If `$ARGUMENTS` contains `--deep`:** run the Literature-Grounded Suggestions step (see below).

4. **Prompt for action:**
   - "Want to inject a hypothesis? Use `/turing:try <idea>`"
   - "Want to continue training? Use `/turing:train`"
   - "Want to compare specific runs? Use `/turing:compare <a> <b>`"
   - "Want literature-backed suggestions? Use `/turing:brief --deep`"

## Literature-Grounded Suggestions (--deep flag)

When `--deep` is requested, add a 7th section to the briefing: **Literature-Grounded Suggestions**.

### Steps for --deep mode:

1. **Classify the task:**
   ```bash
   source .venv/bin/activate && python scripts/classify_task.py --config config.yaml --format json
   ```

2. **Identify stagnation points** from the briefing output:
   - Which model families have been exhausted?
   - Where has the improvement trajectory plateaued?
   - What failure patterns keep recurring?

3. **Search literature** using `WebSearch` for techniques that address the specific stagnation:
   - If plateaued on accuracy: search for "improve [task type] accuracy beyond [current metric]"
   - If overfitting: search for "regularization techniques [model family] [task type]"
   - If all model families tried: search for "state of the art [task type] 2024 2025"

4. **Distill 3-5 suggestions** from the literature, each with:
   - **Technique:** specific, actionable (not "try something different")
   - **Source:** paper or article that recommends this
   - **Why now:** how it addresses the specific stagnation point
   - **Impact estimate:** high/medium/low
   - **Implementation complexity:** low/medium/high

5. **Auto-queue suggestions** as hypotheses:
   ```bash
   source .venv/bin/activate && python scripts/manage_hypotheses.py add "<technique>: <rationale> (source: <citation>)" --priority medium --source literature
   ```

6. **Format as a section:**
   ```
   ## Literature-Grounded Suggestions

   Based on <N> papers/articles consulted for "<task description>":

   1. [HIGH impact, LOW complexity] <technique>
      Source: <citation>
      Addresses: <which stagnation point>
      → Queued as hyp-NNN

   2. [MEDIUM impact, MEDIUM complexity] ...
   ```

## Saving Briefs

To save a briefing for later reference:
```bash
mkdir -p briefs && python scripts/generate_brief.py > briefs/brief-$(date +%Y-%m-%d).md
```

## When to Use

- After a training session completes or converges
- Before injecting new hypotheses (to understand what's already been tried)
- When returning to a project after time away
- Before a research discussion with collaborators
- **With `--deep`:** when the agent seems stuck and you want evidence-based direction
