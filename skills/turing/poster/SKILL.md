---
name: poster
description: Generate a single-page HTML research poster summarizing the experiment campaign — best result, trajectory, key findings, and methodology. Adapted from posterskill's self-contained HTML architecture.
argument-hint: "[title override]"
allowed-tools: Read, Write, Edit, Bash(uv run python scripts/*:*, uv sync:*, mkdir:*, open:*), Grep, Glob
---

Generate a research poster summarizing the experiment campaign as a single self-contained HTML file. Adapted from [posterskill](https://github.com/ethanweber/posterskill)'s architecture — no build step, works when opened as `file://`.

## Steps

### 1. Gather Data

Read the experiment history and project context:

```bash
cat config.yaml
uv run python scripts/generate_brief.py
uv run python scripts/show_metrics.py --last 20
cat experiment_state.yaml 2>/dev/null || true
cat RESEARCH_PLAN.md 2>/dev/null || true
```

From this, extract:
- **Title:** from config task description (or `$ARGUMENTS` override)
- **Best result:** metric name, value, experiment ID
- **Improvement trajectory:** metric values over experiments
- **Key findings:** what model families worked, what didn't, what was surprising
- **Methodology:** the experiment loop, evaluation strategy, convergence criteria
- **Campaign stats:** total experiments, keep rate, time span

### 2. Generate the Poster HTML

Create `poster/index.html` — a self-contained HTML file with:

```bash
mkdir -p poster
```

**Structure the poster with these cards:**

| Card | Content |
|------|---------|
| **Header** | Title, "Autonomous ML Research Campaign", date range, best metric badge |
| **Objective** | Task description and success criteria from config |
| **Methodology** | The autoresearch loop: hypothesize → train → evaluate → decide. Mention immutable evaluation, git-disciplined rollback |
| **Trajectory** | Chart.js line chart of metric progression (embed data inline) |
| **Best Configuration** | Model type, hyperparameters, metric values from best experiment |
| **Key Findings** | 3-5 bullet points: what worked, what didn't, surprises |
| **Explored Approaches** | Table of model families tried with keep rates |
| **Campaign Stats** | Total experiments, keep rate, human vs agent hypotheses, convergence |

**Design principles (from posterskill):**
- Single self-contained HTML file, CDN dependencies only (Chart.js, Google Fonts)
- Print-optimized CSS (`@media print`, `@page` with poster dimensions)
- Card-based layout with colored top borders
- Clean typography (system fonts or Nunito from Google Fonts)
- Data embedded directly in the HTML as JSON — no external file dependencies

**Poster dimensions:** Default A1 landscape (841mm x 594mm). The user can print to PDF from their browser.

### 3. Self-Critique

Review the generated poster:
- Does the trajectory chart render correctly with the embedded data?
- Are the key findings specific and data-grounded (not generic)?
- Is the best configuration complete (model type + all relevant hyperparameters)?
- Would a collaborator understand the campaign from this single page?

Fix any issues found.

### 4. Present

```
Research poster generated at poster/index.html

Open in your browser to view. Print to PDF for sharing.
Best result: <metric>=<value> (<experiment_id>)
Campaign: <N> experiments, <keep_rate>% keep rate
```

Suggest: "Open `poster/index.html` in your browser. Use Ctrl+P / Cmd+P to save as PDF."

## Integration

- The poster reads from the same data sources as `/turing:brief` and `/turing:logbook`
- For a more detailed view, use `/turing:logbook` (full experiment-by-experiment narrative)
- For a quick summary, use `/turing:brief` (text-only intelligence report)
