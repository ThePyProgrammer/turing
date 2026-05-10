---
name: stitch
description: Pipeline composition — decompose ML pipelines into swappable stages. Show, swap, cache, and run stages independently.
disable-model-invocation: true
argument-hint: "<show|swap|cache|run> [stage] [--from exp-id]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Decompose your ML pipeline into stages that can be independently varied, cached, and reused across experiments.

## Steps

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Parse arguments from `$ARGUMENTS`:**
   - First argument is the action: `show`, `swap`, `cache`, `run`
   - `show` — display pipeline stages with hash and cache status
   - `swap <stage> --from <exp-id>` — replace a stage with one from another experiment
   - `cache` — save intermediate stage outputs to disk
   - `run` — execute pipeline, skipping cached stages

3. **Run pipeline manager:**
   ```bash
   python scripts/pipeline_manager.py $ARGUMENTS
   ```

4. **Report results:**
   - **show:** numbered stage list with description, content hash, and cache status
   - **swap:** what changed, old vs new stage config, updated pipeline
   - **cache:** per-stage cache paths and status
   - **run:** which stages will be skipped (cached) vs re-run

5. **Stage types:** preprocess, features, model, postprocess (configurable in `config.yaml` under `pipeline.stages`)

6. **Cache benefit:** when only the model stage changes, preprocessing and feature engineering are skipped — experiments run faster

7. **If no pipeline config:** falls back to default 4-stage pipeline

## Examples

```
/turing:stitch show                          # Display pipeline stages
/turing:stitch swap model --from exp-031     # Keep features, swap model
/turing:stitch cache                         # Cache intermediate outputs
/turing:stitch run                           # Run with cached stages
```
