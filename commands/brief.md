---
name: brief
description: Generate a structured research intelligence report from experiment history — what's been learned, what's promising, what's exhausted, and what the human should consider next. Closes the taste-leverage loop.
disable-model-invocation: true
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

3. **Prompt for action:**
   - "Want to inject a hypothesis? Use `/turing:try <idea>`"
   - "Want to continue training? Use `/turing:train`"
   - "Want to compare specific runs? Use `/turing:compare <a> <b>`"

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
