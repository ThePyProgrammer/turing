# ADR-0015: Extract Metric Output Format into a Documented Contract with Single Parser

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-31 |
| **Author** | Prannaya Gupta |
| **Supersedes** | (none) |
| **Category** | Architecture Pattern |

## Context

`evaluate.py`'s `format_metrics()` outputs metrics between `---` delimiters in a `key:     value` format. Three independent consumers parse this format:

1. **The agent** via `grep -A 10 "^---" run.log | head -10`
2. **`post-train-hook.sh`** via `sed` + `awk -F:` + associative arrays + manual JSON construction
3. **`stop-hook.sh`** reads `log.jsonl` (which was written by parser #2)

The format is undocumented — no schema, no spec beyond "between `---` delimiters." The bash parser in `post-train-hook.sh` builds JSON by string concatenation, which breaks if metric values contain colons or are not bare numbers. Metadata keys (`model_type`, `train_seconds`) are mixed with metric keys in the same block, with separation logic hardcoded in both `evaluate.py` and `post-train-hook.sh`.

Three independent parsers for one format means three independent bug surfaces.

## Options Considered

### Option 1: Single Python Parser Module

Extract metric parsing into `scripts/parse_metrics.py` with a function `parse_run_log(path) -> dict`. All three consumers use this single parser: the agent calls it from Python, `post-train-hook.sh` calls it as a CLI tool, and `stop-hook.sh` reads the JSONL written by the parser.

Trade-offs: one parser, one bug surface. The bash hook becomes a thin wrapper calling Python.

### Option 2: JSON Output Instead of Custom Format

Change `format_metrics()` to output JSON instead of the custom `---` delimited format. JSON has standard parsers in every language.

Trade-offs: eliminates the parsing problem entirely. But changes the agent-facing output format, requiring updates to `program.md`, `train.md`, and `loop-protocol.md`.

### Option 3: Document the Format, Keep Three Parsers

Write a formal spec for the `---` delimited format. Keep three parsers but test them against the spec.

Trade-offs: least disruptive. But three parsers still means three maintenance points.

## Decision

**We will extract a single Python parser module** and have all consumers use it. The custom `---` delimited format remains (it is human-readable in `run.log`), but only one piece of code parses it.

## Rationale

The format itself is fine for its purpose — human-readable metric output in a log file. The problem is that three independent implementations parse it. A single parser eliminates two of the three bug surfaces and makes the remaining one testable.

## Consequences

### Positive

- One parser to test, one parser to maintain
- `post-train-hook.sh` eliminates 30 lines of fragile bash string manipulation
- The metric format has a canonical parser that defines what is valid

### Negative

- `post-train-hook.sh` gains a Python dependency (must activate venv to parse)
- Slightly more complex hook invocation

### Neutral

- The `---` delimited format remains unchanged in `run.log`

## References

- Architecture Evaluation Report (2026-03-31) — bug surface dimension flagged three independent parsers
- `templates/evaluate.py:88-114` — `format_metrics()` output format
- `templates/scripts/post-train-hook.sh:28-67` — bash metric parser
- `templates/program.md` — agent grep instructions
