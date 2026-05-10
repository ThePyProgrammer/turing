---
name: mode
description: Set the research strategy mode — explore (try new things), exploit (refine what works), or replicate (verify results). Drives novelty guard policy and agent behavior.
argument-hint: "<explore|exploit|replicate>"
---

Set the research mode for the current project. The mode determines how the novelty guard filters proposed experiments and how the agent prioritizes its work.

## Modes

| Mode | Novelty Guard Policy | Agent Behavior |
|------|---------------------|----------------|
| **explore** | Allow novel ideas, block repeats and follow-ups | Try fundamentally different approaches |
| **exploit** | Allow follow-ups and known successes, block repeats | Refine the current best configuration |
| **replicate** | Allow duplicate runs, block novel ideas | Re-run best experiments with different seeds |

## Steps

1. **Parse mode** from `$ARGUMENTS`. Must be one of: `explore`, `exploit`, `replicate`.

2. **Update experiment state:**
   ```bash
   source .venv/bin/activate
   python -c "
   import yaml
   from pathlib import Path
   path = Path('experiment_state.yaml')
   state = yaml.safe_load(path.read_text()) if path.exists() else {}
   state['research_mode'] = '$ARGUMENTS'
   path.write_text(yaml.dump(state, default_flow_style=False))
   print(f'Research mode set to: $ARGUMENTS')
   "
   ```

3. **Confirm** with guidance:
   - `explore`: "The agent will prioritize novel ideas and avoid follow-ups. Best when the current approach feels exhausted."
   - `exploit`: "The agent will refine the current best. Best when you have a promising direction."
   - `replicate`: "The agent will re-run experiments for statistical verification. Best before declaring a winner."

## Default

The default mode is `exploit` (refine what works). Change to `explore` when plateauing, `replicate` before final decisions.
