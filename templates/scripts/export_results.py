"""Export experiment results to CSV, Markdown, LaTeX, or JSON.

Reads experiments/log.jsonl and renders the selected experiments in the
requested format — suitable for pasting into README files, academic papers,
or downstream data pipelines.

Typical usage:
    python scripts/export_results.py                                    # CSV, all experiments
    python scripts/export_results.py --format markdown --status kept    # Markdown, kept only
    python scripts/export_results.py --format latex --last 10 --sort accuracy
    python scripts/export_results.py --format csv --output results.csv
    python scripts/export_results.py --columns "experiment_id,accuracy,f1_weighted" --format markdown
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"

# Columns that are promoted to the top level from nested dicts in the raw
# JSONL entry.  Everything else is treated as a plain top-level key.
_CONFIG_KEYS = {"model_type", "hyperparams"}
_DEFAULT_COLUMNS = ["experiment_id", "status", "model_type", "description"]


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


def _flatten(entry: dict) -> dict[str, Any]:
    """Return a flat dict from one JSONL entry.

    Promotes ``config.model_type`` and all ``metrics.*`` keys to the top
    level so callers can reference them by bare name (e.g. "accuracy",
    "model_type").
    """
    flat: dict[str, Any] = {}

    # Top-level scalar fields
    for key in ("experiment_id", "timestamp", "status", "description",
                "git_commit", "parent_experiment", "hypothesis_id", "family",
                "model_path"):
        flat[key] = entry.get(key, "")

    # Tags as a semicolon-separated string so it fits in a single cell
    tags = entry.get("tags", [])
    flat["tags"] = ";".join(tags) if isinstance(tags, list) else str(tags or "")

    # config sub-keys
    cfg = entry.get("config") or {}
    flat["model_type"] = cfg.get("model_type", "")

    # metrics — each metric becomes its own column
    for k, v in (entry.get("metrics") or {}).items():
        flat[k] = v

    return flat


def _collect_all_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Return a deterministic column order across all rows.

    Order: default columns first (preserving the list), then all metric
    columns sorted alphabetically, then any remaining keys sorted.
    """
    seen: set[str] = set()
    ordered: list[str] = []

    for col in _DEFAULT_COLUMNS:
        if col not in seen:
            ordered.append(col)
            seen.add(col)

    # Collect metric keys (anything that is not in the fixed set)
    fixed = {
        "experiment_id", "timestamp", "status", "description",
        "git_commit", "parent_experiment", "hypothesis_id", "family",
        "model_path", "tags", "model_type",
    }
    metric_keys: set[str] = set()
    extra_keys: set[str] = set()
    for row in rows:
        for k in row:
            if k in seen:
                continue
            if k not in fixed:
                metric_keys.add(k)
            else:
                extra_keys.add(k)

    for k in sorted(metric_keys):
        if k not in seen:
            ordered.append(k)
            seen.add(k)
    for k in sorted(extra_keys):
        if k not in seen:
            ordered.append(k)
            seen.add(k)

    return ordered


# ---------------------------------------------------------------------------
# Filtering and sorting
# ---------------------------------------------------------------------------


def filter_experiments(
    experiments: list[dict],
    status: str,
    last: int | None,
    family: str | None,
) -> list[dict]:
    """Apply status, family, and recency filters to the raw experiment list."""
    result = experiments

    if status != "all":
        result = [e for e in result if e.get("status") == status]

    if family:
        result = [e for e in result if e.get("family") == family]

    if last is not None and last > 0:
        result = result[-last:]

    return result


def sort_experiments(
    rows: list[dict[str, Any]],
    sort_key: str | None,
    ascending: bool,
) -> list[dict[str, Any]]:
    """Sort rows by the given key.  Rows missing the key sort to the end."""
    if not sort_key:
        return rows

    def _key(row: dict[str, Any]) -> tuple[int, Any]:
        val = row.get(sort_key)
        if val is None or val == "":
            return (1, 0)
        try:
            return (0, float(val))
        except (TypeError, ValueError):
            return (0, str(val))

    return sorted(rows, key=_key, reverse=not ascending)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _cell(value: Any) -> str:
    """Convert a cell value to a display string."""
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        # Trim trailing zeros but keep up to 6 decimal places
        return f"{value:.6g}"
    return str(value)


def format_csv(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render rows as RFC 4180 CSV."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=columns,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({col: _cell(row.get(col)) for col in columns})
    return buf.getvalue()


def format_markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render rows as a GitHub-flavoured Markdown table."""
    if not rows:
        return "_No experiments match the requested filters._\n"

    # Compute column widths
    widths = {col: max(len(col), max((len(_cell(r.get(col))) for r in rows), default=0))
              for col in columns}

    def _pad(text: str, col: str) -> str:
        return text.ljust(widths[col])

    header = "| " + " | ".join(_pad(col, col) for col in columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in columns) + " |"
    lines = [header, sep]

    for row in rows:
        cells = " | ".join(_pad(_cell(row.get(col)), col) for col in columns)
        lines.append(f"| {cells} |")

    return "\n".join(lines) + "\n"


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in a cell value."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def format_latex(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render rows as a LaTeX tabular environment.

    Produces a self-contained snippet ready to drop into a paper's
    ``table`` float.  Numeric columns use right-alignment; text columns
    use left-alignment.
    """
    if not rows:
        return "% No experiments match the requested filters.\n"

    # Determine alignment: right-align columns whose values look numeric
    def _is_numeric_col(col: str) -> bool:
        for row in rows:
            val = row.get(col)
            if val is None or val == "":
                continue
            try:
                float(val)
                return True
            except (TypeError, ValueError):
                return False
        return False

    alignments = ["r" if _is_numeric_col(c) else "l" for c in columns]
    col_spec = " ".join(alignments)

    header_row = " & ".join(
        r"\textbf{" + _latex_escape(col.replace("_", r"\_")) + "}"
        for col in columns
    )

    data_lines = []
    for row in rows:
        cells = " & ".join(_latex_escape(_cell(row.get(col))) for col in columns)
        data_lines.append(f"  {cells} \\\\")

    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"  \hline",
        f"  {header_row} \\\\",
        r"  \hline",
        *data_lines,
        r"  \hline",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def format_json(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render rows as a JSON array containing only the selected columns."""
    filtered = [{col: row.get(col) for col in columns} for row in rows]
    return json.dumps(filtered, indent=2, default=str) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the configured argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Export experiment results from experiments/log.jsonl to CSV, "
            "Markdown, LaTeX, or JSON."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python scripts/export_results.py
  python scripts/export_results.py --format markdown --status kept
  python scripts/export_results.py --format latex --last 10 --sort accuracy
  python scripts/export_results.py --format csv --output results.csv
  python scripts/export_results.py --columns "experiment_id,accuracy,f1_weighted" --format markdown
""",
    )

    parser.add_argument(
        "--log",
        default=DEFAULT_LOG_PATH,
        metavar="PATH",
        help=f"Path to JSONL experiment log (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "markdown", "latex", "json"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--status",
        choices=["all", "kept", "discarded"],
        default="all",
        help="Filter by experiment status (default: all)",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        metavar="N",
        help="Include only the last N experiments (applied after status filter)",
    )
    parser.add_argument(
        "--family",
        default=None,
        metavar="NAME",
        help="Filter to a specific experiment family",
    )
    parser.add_argument(
        "--columns",
        default=None,
        metavar="COL1,COL2,...",
        help=(
            "Comma-separated list of columns to include. "
            "Defaults to experiment_id, status, model_type, all metrics, description."
        ),
    )
    parser.add_argument(
        "--sort",
        default=None,
        metavar="METRIC",
        help="Sort by this column, descending (higher-is-better default)",
    )
    parser.add_argument(
        "--sort-asc",
        default=None,
        metavar="METRIC",
        dest="sort_asc",
        help="Sort by this column, ascending (lower-is-better)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Write output to FILE instead of stdout",
    )

    return parser


def main() -> None:
    """CLI entry point for export_results."""
    parser = build_parser()
    args = parser.parse_args()

    # --sort and --sort-asc are mutually exclusive
    if args.sort and args.sort_asc:
        parser.error("--sort and --sort-asc are mutually exclusive")

    sort_key = args.sort or args.sort_asc
    ascending = bool(args.sort_asc)

    # Load config for metric names (used to build default column list)
    config = load_config()
    eval_cfg = config.get("evaluation", {})
    configured_metrics: list[str] = eval_cfg.get("metrics", [])

    # Load and filter experiments
    raw_experiments = load_experiments(args.log)

    if not raw_experiments:
        msg = f"No experiments found in {args.log}."
        if args.format in ("markdown",):
            print(f"_{msg}_")
        else:
            print(msg, file=sys.stderr)
        sys.exit(0)

    filtered = filter_experiments(raw_experiments, args.status, args.last, args.family)

    if not filtered:
        msg = "No experiments match the requested filters."
        if args.format == "markdown":
            print(f"_{msg}_")
        else:
            print(msg, file=sys.stderr)
        sys.exit(0)

    # Flatten each entry
    rows = [_flatten(e) for e in filtered]

    # Determine column list
    if args.columns:
        columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    else:
        # Build default: fixed header + configured metrics + description
        all_cols = _collect_all_columns(rows)
        # Promote configured metrics to appear before description when they
        # exist in the data.
        fixed_head = ["experiment_id", "status", "model_type"]
        fixed_tail = ["description"]
        metric_cols = [m for m in configured_metrics if m in {c for r in rows for c in r}]
        # Any additional metric columns not listed in config
        extra_metric_cols = [
            c for c in all_cols
            if c not in fixed_head
            and c not in fixed_tail
            and c not in metric_cols
            and c not in {"timestamp", "git_commit", "parent_experiment",
                          "hypothesis_id", "family", "model_path", "tags"}
        ]
        columns = fixed_head + metric_cols + extra_metric_cols + fixed_tail

    # Sort
    rows = sort_experiments(rows, sort_key, ascending)

    # Render
    formatters = {
        "csv": format_csv,
        "markdown": format_markdown,
        "latex": format_latex,
        "json": format_json,
    }
    output_text = formatters[args.format](rows, columns)

    # Write
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"Wrote {len(rows)} experiment(s) to {out_path}", file=sys.stderr)
    else:
        print(output_text, end="")


if __name__ == "__main__":
    main()
