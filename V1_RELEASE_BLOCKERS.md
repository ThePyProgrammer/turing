# v1.0.0 Release Blockers & Quality Issues

*Generated from architecture evaluation (2026-03-31). 5 agents, 5 dimensions.*

## Status: MUST-FIX BLOCKERS RESOLVED — SHOULD-FIX items remain

---

## MUST FIX (blocks v1.0.0 release)

### 1. Installer manifest divergence — commands
- **Issue:** `src/install.js` SUB_COMMANDS lists 8 commands; `commands/` directory has 14
- **Missing:** design, logbook, mode, poster, report, suggest
- **Impact:** npm users get a router that references 6 commands that don't exist
- **Files:** `src/install.js`, `src/verify.js`, `src/claude-md.js`
- **Status:** [x] FIXED

### 2. Installer manifest divergence — config
- **Issue:** `src/install.js` CONFIG_FILES lists 3 files; `config/` has 8
- **Missing:** experiment_archetypes.yaml, novelty_aliases.yaml, relationships.toml, state.toml, task_taxonomy.yaml
- **Impact:** Archetype expansion returns "none loaded", novelty aliasing silently degrades
- **Files:** `src/install.js`, `src/verify.js`
- **Status:** [x] FIXED

### 3. ARCHITECTURE.md codemap is stale
- **Issue:** Documents 6 commands, 3 configs, 7 scripts. Actual: 14 commands, 8 configs, ~22 scripts
- **Impact:** Contributors form wrong mental model, duplicate existing work
- **Files:** `docs/ARCHITECTURE.md`
- **Status:** [x] FIXED

### 4. plugin.json description stale
- **Issue:** Says "6 commands" — actual is 14
- **Files:** `.claude-plugin/plugin.json`
- **Status:** [x] FIXED

### 5. `prepare.py` has zero tests
- **Issue:** The MEASUREMENT APPARATUS foundation (ADR-0002) is untested
- **Missing tests:** load_data (CSV, JSONL, unsupported format, missing file, empty), create_splits (stratification, no overlap, deterministic, missing column), load_splits (missing file)
- **Impact:** Stratification bug invalidates every experiment comparison
- **Files:** Need new `tests/test_prepare.py`
- **Status:** [x] FIXED

### 6. `featurizers.py` has zero tests
- **Issue:** Feature pipeline the agent composes — untested
- **Missing tests:** NumericFeaturizer (auto-detect, explicit columns), CategoricalFeaturizer (one-hot, unseen category), CompositeFeaturizer (concatenation, empty input), fit_transform consistency
- **Impact:** Unseen category at transform time = production crash
- **Files:** Need new `tests/test_featurizers.py`
- **Status:** [x] FIXED

---

## SHOULD FIX (quality, not blocking)

### 7. `load_experiments()` duplicated 8 times
- **Issue:** Identical JSONL loader copy-pasted across 8 scripts. Plus 4x load_config(), 3x find_best()
- **Impact:** Schema change requires 8 edits, guaranteed to miss one
- **Fix:** Extract to shared `scripts/turing_io.py`
- **Status:** [x] FIXED

### 8. `EMPTY_STATE` shallow copy in update_state.py
- **Issue:** `dict(EMPTY_STATE)` shares list references with module-level constant
- **Impact:** Multiple calls in one process corrupt shared state
- **Fix:** Use `copy.deepcopy(EMPTY_STATE)` instead of `dict(EMPTY_STATE)`
- **Status:** [x] FIXED

### 9. `log_experiment.py` uses manual sys.argv parsing
- **Issue:** Only script not using argparse — manual while loop, no --help, no type validation
- **Impact:** Wrong argument order silently misinterpreted on every experiment
- **Fix:** Migrate to argparse
- **Status:** [x] FIXED

### 10. No integration test for metrics format/parse roundtrip
- **Issue:** format_metrics() -> parse_metrics_block() contract untested as integrated flow
- **Impact:** Format drift between producer and consumer breaks silently
- **Fix:** Add test_metrics_roundtrip to test_parse_metrics.py
- **Status:** [x] FIXED

### 11. No manifest consistency test
- **Issue:** No test validates install.js arrays match filesystem
- **Impact:** Manifest drift recurs every time a command/config is added
- **Fix:** Add test that compares manifests to directory contents
- **Status:** [x] FIXED

### 12. No CONTRIBUTING.md
- **Issue:** Command authoring conventions are implicit knowledge
- **Impact:** Second contributor adds commands wrong
- **Fix:** Document skill file format, frontmatter fields, script delegation pattern
- **Status:** [x] FIXED

---

## NICE TO HAVE (defer to v1.1)

### 13. TOML/YAML format compliance (ADR-0004)
- experiment_archetypes.yaml and task_taxonomy.yaml should be TOML per ADR-0004

### 14. commands/mode.md missing allowed-tools frontmatter

### 15. Nesting depth 8-9 in manage_hypotheses.py and update_state.py main()

### 16. templates/scripts/ flat directory (25 scripts, no sub-packages)

### 17. Status vocabulary as Python constants module (not string literals in 15+ files)
