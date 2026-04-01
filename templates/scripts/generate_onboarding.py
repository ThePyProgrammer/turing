#!/usr/bin/env python3
"""Project onboarding generator for new collaborators.

Reads config, experiments, annotations, and hypotheses to produce a
structured walkthrough: task description, what's been tried (grouped
by family), key decisions, where heading, how to start.

Usage:
    python scripts/generate_onboarding.py --audience researcher --depth full
    python scripts/generate_onboarding.py --audience stakeholder --depth brief --json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from scripts.turing_io import load_config, load_experiments, load_hypotheses

VALID_AUDIENCES = ["researcher", "engineer", "stakeholder"]
VALID_DEPTHS = ["brief", "full"]
DEFAULT_LOG = "experiments/log.jsonl"
DEFAULT_ANNOTATIONS = "experiments/annotations.yaml"
DEFAULT_HYPOTHESES = "hypotheses.yaml"


def _load_yaml_list(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with open(p) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def _load_yaml_dir(directory: str, glob: str) -> list[dict]:
    path = Path(directory)
    if not path.exists():
        return []
    items = []
    for f in sorted(path.glob(glob)):
        try:
            with open(f) as fh:
                d = yaml.safe_load(fh)
                if d and isinstance(d, dict):
                    items.append(d)
        except (yaml.YAMLError, OSError):
            continue
    return items


def _family_summary(exps: list[dict], metric: str, lower_is_better: bool) -> dict:
    total = len(exps)
    kept = [e for e in exps if e.get("status") == "kept"]
    best_val, best_id = None, None
    for e in kept:
        val = e.get("metrics", {}).get(metric)
        if val is None:
            continue
        if best_val is None or (lower_is_better and val < best_val) or (not lower_is_better and val > best_val):
            best_val, best_id = val, e.get("experiment_id")
    return {"total": total, "kept": len(kept), "keep_rate": round(len(kept) / total, 2) if total else 0,
            "best_metric": best_val, "best_experiment": best_id}


def _find_best(experiments: list[dict], metric: str, lower_is_better: bool) -> dict | None:
    best, best_val = None, float("inf") if lower_is_better else float("-inf")
    for e in experiments:
        if e.get("status") != "kept":
            continue
        val = e.get("metrics", {}).get(metric)
        if val is not None and ((lower_is_better and val < best_val) or (not lower_is_better and val > best_val)):
            best_val, best = val, e
    return best


def _extract_decisions(packets: list[dict], annotations: list[dict]) -> list[dict]:
    decisions = []
    for pkt in packets:
        if pkt.get("action") in ("promote", "abandon", "replicate"):
            decisions.append({"type": "decision", "experiment": pkt.get("experiment_id", "?"),
                              "action": pkt["action"], "reason": pkt.get("reason", ""),
                              "date": pkt.get("timestamp", "")[:10]})
    key_tags = {"decision", "key", "important", "milestone"}
    for ann in annotations:
        if set(t.lower() for t in ann.get("tags", [])) & key_tags:
            decisions.append({"type": "annotation", "experiment": ann.get("experiment_id", "?"),
                              "text": ann.get("text", ""), "date": ann.get("date", "")[:10]})
    decisions.sort(key=lambda d: d.get("date", ""), reverse=True)
    return decisions


def _project_direction(hypotheses: list[dict], experiments: list[dict]) -> dict:
    queued = [h for h in hypotheses if h.get("status") == "queued"]
    promising = [h for h in hypotheses if h.get("status") == "promising"]
    recent = experiments[-5:] if len(experiments) >= 5 else experiments
    recent_kept = sum(1 for e in recent if e.get("status") == "kept")
    if not experiments:
        phase = "not_started"
    elif not queued and not promising:
        phase = "exhausted"
    elif recent_kept == 0 and len(recent) >= 3:
        phase = "plateaued"
    elif promising:
        phase = "promising_leads"
    else:
        phase = "active_exploration"
    return {"phase": phase, "queued": queued[:5], "promising": promising[:3],
            "n_queued": len(queued), "n_promising": len(promising)}


def format_onboarding_report(config, experiments, families, best, decisions,
                             direction, annotations, seeds, audience, depth,
                             metric, lower_is_better) -> str:
    d = "lower" if lower_is_better else "higher"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = ["# Project Onboarding", "", f"*Generated {now} for audience: {audience}*", "", "---", "",
         "## 1. What This Project Does", ""]
    data_cfg, eval_cfg = config.get("data", {}), config.get("evaluation", {})
    L.append(f"**Task:** {config.get('task_description', 'N/A')}")
    L.append(f"**Dataset:** {data_cfg.get('source', 'unknown')}")
    L.append(f"**Primary metric:** `{metric}` ({d} is better)")
    extra = [m for m in eval_cfg.get("metrics", []) if m != metric]
    if extra:
        L.append(f"**Additional metrics:** {', '.join(f'`{m}`' for m in extra)}")
    if depth == "full" and audience != "stakeholder":
        sr = data_cfg.get("split_ratios", {})
        if sr:
            L.append(f"**Data splits:** {' / '.join(f'{k}: {int(v*100)}%' for k, v in sr.items())}")
        if data_cfg.get("target_column"):
            L.append(f"**Target column:** `{data_cfg['target_column']}`")
    L.extend(["", "## 2. What's Been Tried", ""])
    total, kept_n = len(experiments), sum(1 for e in experiments if e.get("status") == "kept")
    if total == 0:
        L.append("No experiments yet. Start with `/turing:train`.")
    else:
        L.append(f"**{total} experiments**, **{kept_n} kept** ({round(kept_n/total*100)}% keep rate).")
        L.append("")
        if best:
            ms = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in best.get("metrics", {}).items())
            L.extend([f"**Champion:** `{best.get('experiment_id','?')}` "
                       f"({best.get('config',{}).get('model_type','?')}) — {ms}", ""])
        L.extend(["### By Family", "", "| Family | Exps | Kept | Best | Status |",
                   "|--------|------|------|------|--------|"])
        for name, s in sorted(families.items()):
            bv = f"{s['best_metric']:.4f}" if s["best_metric"] is not None else "---"
            st = "Exhausted" if s["keep_rate"] == 0 and s["total"] >= 3 else (
                 "Productive" if s["keep_rate"] >= 0.5 else "Mixed")
            L.append(f"| {name} | {s['total']} | {s['kept']} | {bv} | {st} |")
        L.append("")
        if depth == "full" and audience in ("researcher", "engineer"):
            for name, s in sorted(families.items()):
                if not s["total"]:
                    continue
                L.append(f"#### {name}")
                if s["best_experiment"]:
                    L.append(f"- Best: `{s['best_experiment']}` ({metric}={s['best_metric']:.4f})")
                L.append(f"- {s['kept']}/{s['total']} kept ({s['keep_rate']:.0%})")
                fam_ids = {e.get("experiment_id") for e in experiments if (e.get("family") or "untagged") == name}
                notes = [a for a in annotations if a.get("experiment_id") in fam_ids]
                for n in notes[:3]:
                    L.append(f"  - {n.get('text','')[:80]}")
                L.append("")
    L.extend(["## 3. Key Decisions", ""])
    if not decisions:
        L.append("No major decisions recorded yet.")
    else:
        lim = 5 if depth == "brief" else 15
        for dec in decisions[:lim]:
            if dec["type"] == "decision":
                L.append(f"- **{dec['date']}** `{dec['experiment']}`: **{dec['action']}** — {dec['reason']}")
            else:
                L.append(f"- **{dec['date']}** `{dec['experiment']}`: {dec['text'][:100]}")
        if len(decisions) > lim:
            L.append(f"  *...and {len(decisions)-lim} more*")
    L.extend(["", "## 4. Where We're Heading", ""])
    phases = {"not_started": "Project has not started experiments yet.",
              "exhausted": "All hypotheses tested. Need fresh ideas.",
              "plateaued": "Recent experiments not improving. Consider pivoting.",
              "promising_leads": "Promising directions identified and being pursued.",
              "active_exploration": "Actively exploring hypothesis space."}
    L.extend([phases.get(direction["phase"], "Unknown phase."), ""])
    if direction["n_queued"]:
        L.append(f"**{direction['n_queued']} hypotheses queued:**")
        for h in direction["queued"]:
            p = " (HIGH)" if h.get("priority") == "high" else ""
            L.append(f"- {h.get('id','?')}: {h.get('description','?')}{p}")
        L.append("")
    if direction["n_promising"]:
        L.append(f"**{direction['n_promising']} promising lead(s):**")
        for h in direction["promising"]:
            L.append(f"- {h.get('id','?')}: {h.get('description','?')}")
        L.append("")
    sensitive = [s for s in seeds if s.get("seed_sensitive")]
    if sensitive and audience != "stakeholder":
        L.append("**Seed sensitivity warnings:**")
        for s in sensitive:
            L.append(f"- `{s.get('experiment_id','?')}`: CV={s.get('cv_percent',0):.1f}%")
        L.append("")
    L.extend(["## 5. How to Get Started", ""])
    cmds = {"researcher": [
        "1. Read `config.yaml` for task and evaluation setup",
        "2. `/turing:status` — current experiment state",
        "3. `/turing:brief` — full research intelligence report",
        "4. Review `hypotheses.yaml` for queued ideas",
        "5. `/turing:try \"your hypothesis\"` — inject ideas",
        "6. `/turing:train` — run next experiment",
    ], "engineer": [
        "1. `pip install -r requirements.txt`",
        "2. Review `config.yaml` for data paths",
        "3. `/turing:status` — where things stand",
        "4. Check `train.py` for current model",
        "5. `/turing:train` — execute experiments",
    ], "stakeholder": [
        "1. `/turing:brief` — high-level summary",
        "2. Check champion performance above",
        "3. Review 'Where We're Heading' for next steps",
    ]}
    L.extend(cmds.get(audience, []))
    L.extend(["", "---", f"*Generated by `/turing:onboard` — {audience}, {depth}*"])
    return "\n".join(L)


def save_onboarding_report(content: str, path: str = "ONBOARDING.md") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def generate_onboarding(config_path="config.yaml", log_path=DEFAULT_LOG,
                         hypotheses_path=DEFAULT_HYPOTHESES,
                         annotations_path=DEFAULT_ANNOTATIONS,
                         audience="researcher", depth="full") -> dict:
    """Generate full onboarding report. Returns dict with report and metadata."""
    config = load_config(config_path)
    metric = config.get("evaluation", {}).get("primary_metric", "accuracy")
    lower = config.get("evaluation", {}).get("lower_is_better", False)
    experiments = load_experiments(log_path)
    hypotheses = load_hypotheses(hypotheses_path)
    annotations = _load_yaml_list(annotations_path)
    packets = _load_yaml_dir("experiments/decisions", "*.yaml")
    seeds = _load_yaml_dir("experiments/seed_studies", "*-seeds.yaml")
    fam_groups = {}
    for e in experiments:
        fam_groups.setdefault(e.get("family") or "untagged", []).append(e)
    families = {n: _family_summary(exps, metric, lower) for n, exps in fam_groups.items()}
    best = _find_best(experiments, metric, lower)
    decisions = _extract_decisions(packets, annotations)
    direction = _project_direction(hypotheses, experiments)
    report = format_onboarding_report(config, experiments, families, best, decisions,
                                       direction, annotations, seeds, audience, depth, metric, lower)
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "audience": audience,
            "depth": depth, "total_experiments": len(experiments),
            "project_phase": direction["phase"], "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate project onboarding for new collaborators")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--hypotheses", default=DEFAULT_HYPOTHESES)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--audience", default="researcher", choices=VALID_AUDIENCES)
    parser.add_argument("--depth", default="full", choices=VALID_DEPTHS)
    parser.add_argument("--output", default="ONBOARDING.md")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()
    result = generate_onboarding(args.config, args.log, args.hypotheses,
                                  args.annotations, args.audience, args.depth)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        saved = save_onboarding_report(result["report"], args.output)
        print(result["report"])
        print(f"\nSaved to {saved}", file=sys.stderr)


if __name__ == "__main__":
    main()
