---
title: "Experiment Archaeology"
description: "Long-term trend analysis, session restoration, archival, annotations, search, templates, and experiment replay."
---

# Experiment Archaeology

Commands for navigating and maintaining a large experiment history -- spotting long-term trends, restoring context after time away, cleaning up old artifacts, annotating experiments with human context, searching across hundreds of runs, saving reusable templates, and replaying old experiments with current infrastructure.

---

### `/turing:trend` -- Long-term trend analysis

See the arc of your research, not just the latest results. Analyzes improvement velocity over time windows, ranks model families by ROI, detects diminishing returns, and identifies phase transitions in your experiment history. Most useful after 100+ experiments when the strategic picture matters more than individual results.

**Syntax:** `/turing:trend [--window 30d] [--metric accuracy]`

**Examples:**
```
/turing:trend                        # Full trend analysis
/turing:trend --window 14d           # Last 2 weeks
```

---

### `/turing:flashback` -- Session context restoration

Come back to a project after a week and start working in 10 seconds instead of 30 minutes. Summarizes the current best model, last session's experiments, pending hypotheses, annotations, budget status, and a suggested next action. The "where was I?" command.

**Syntax:** `/turing:flashback [--days 7] [--last 10]`

**Examples:**
```
/turing:flashback                    # Default: last 7 days
/turing:flashback --days 14          # 2-week lookback
/turing:flashback --last 5           # Last 5 experiments
```

---

### `/turing:archive` -- Experiment lifecycle cleanup

Keep your project directory manageable after 200+ experiments. Compresses old artifacts, prunes checkpoints, and creates a queryable summary index. Protected experiments (Pareto-optimal, current best, recent, top-N by metric) are never archived. Reports archived count, preserved count, and space reclaimed.

**Syntax:** `/turing:archive [--older-than 30d] [--keep-best 10] [--dry-run]`

**Examples:**
```
/turing:archive --dry-run                    # Preview what would be archived
/turing:archive --older-than 30 --keep-best 10  # Archive old, keep top 10
/turing:archive                              # Default: 30 days, keep 10
```

---

### `/turing:annotate` -- Retrospective experiment annotations

Add context that experiment logs cannot capture. Human notes, tags, and observations that automated metrics miss: "This only worked because the data was pre-sorted." Annotations are stored alongside experiment data and surfaced by `/turing:flashback` and `/turing:search`.

**Syntax:** `/turing:annotate <exp-id> "note" [--tag fragile] | --list | --search "keyword"`

**Examples:**
```
/turing:annotate exp-042 "Fragile -- only works with specific preprocessing"
/turing:annotate exp-042 "Reviewer 2 requested this" --tag reviewer-requested
/turing:annotate --list
/turing:annotate --search "fragile"
```

---

### `/turing:search` -- Natural language experiment search

Find specific experiments in a large history with natural language queries and structured filters. Supports metric thresholds, status filters, family filters, and date ranges. Returns a ranked table of matching experiments.

**Syntax:** `/turing:search <query> [--filter "accuracy>0.85"] [--limit 10]`

**Examples:**
```
/turing:search "LightGBM high accuracy" --filter "accuracy>0.85"
/turing:search "failed neural net" --filter "status:discarded"
/turing:search "last week" --limit 5
```

---

### `/turing:template` -- Experiment template library

Turn your best experiment configs into reusable recipes that persist across projects. Save winning configurations from any experiment, list available templates, apply them to new projects, or share them as exports. Templates are stored at `~/.turing/templates/` for cross-project reuse.

**Syntax:** `/turing:template <save|list|apply|share> [--name name] [--from exp-id]`

**Examples:**
```
/turing:template save --from exp-042 --name "tabular-xgboost-v2"
/turing:template list
/turing:template apply tabular-xgboost-v2
```

---

### `/turing:replay` -- Experiment replay

Re-run a historical experiment with current infrastructure to test if old approaches do better now. Infrastructure changes -- new library versions, improved preprocessing, additional data -- may make previously failed approaches viable. Compares original vs replayed metrics and reports the delta.

**Syntax:** `/turing:replay <exp-id> [--with-current-data] [--with-current-preprocessing] [--list]`

**Examples:**
```
/turing:replay exp-023                              # Replay with current infrastructure
/turing:replay exp-023 --with-current-data          # Current data, old code
/turing:replay --list                               # List replayable experiments
```
