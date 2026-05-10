---
name: present
description: Presentation figure generation — training curves, comparison charts, ablation tables, Pareto plots, sensitivity heatmaps.
argument-hint: "[--figures training,comparison] [--style light|dark|poster]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Generate presentation-ready figure specifications from experiment data in seconds.

## Steps
1. **Sync environment:** `uv sync`
2. **Run:** `uv run python scripts/generate_figures.py $ARGUMENTS`
3. **Figure types:** training, comparison, ablation, pareto, sensitivity
4. **Styles:** light (papers), dark (demos), poster (large fonts)
5. **Saved output:** `paper/figures/`

## Examples
```
/turing:present                                  # All figures
/turing:present --figures training,comparison    # Specific figures
/turing:present --style dark                     # Dark theme
```
