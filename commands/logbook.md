---
name: logbook
description: Generate a research logbook showing the full experiment narrative — hypotheses proposed, experiments run, decisions made, and progress over time. Outputs HTML (with interactive chart) or markdown.
argument-hint: "[--since YYYY-MM-DD] [--format html|markdown] [--output path]"
allowed-tools: Read, Bash(python scripts/*:*, source .venv/bin/activate:*, mkdir:*), Grep, Glob
---

Generate a research logbook that captures the full narrative of the experiment campaign.

## Steps

1. **Generate the logbook:**
   ```bash
   source .venv/bin/activate && python scripts/generate_logbook.py
   ```

   **With options from `$ARGUMENTS`:**
   - `--since 2026-03-15` — only include events after this date
   - `--format markdown` — output as markdown instead of HTML
   - `--output logbook.html` — write to file instead of stdout

   **Common usage:**
   ```bash
   # HTML logbook with interactive trajectory chart
   source .venv/bin/activate && python scripts/generate_logbook.py --output logbook.html

   # Markdown for embedding in docs or READMEs
   source .venv/bin/activate && python scripts/generate_logbook.py --format markdown --output logbook.md

   # Last week's activity
   source .venv/bin/activate && python scripts/generate_logbook.py --since 2026-03-24 --output logbook.html
   ```

2. **Present the result:**
   - If HTML: tell the user to open the file in their browser. The logbook includes an interactive Chart.js trajectory visualization.
   - If markdown: display inline or note the output file location.

## What the Logbook Contains

- **Campaign summary:** total experiments, keep rate, best metric, hypothesis count
- **Improvement trajectory:** interactive line chart showing metric progression and best-so-far envelope
- **Experiment log:** every experiment with ID, description, metric value, status (kept/discarded), date
- **Hypothesis queue:** every hypothesis with source (human/agent/literature), status, priority

## When to Use

- To share progress with collaborators
- Before and after meetings to show what was tried
- To archive a completed research campaign
- To track progress over a specific time period
