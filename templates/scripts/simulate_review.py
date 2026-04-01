#!/usr/bin/env python3
"""Simulated peer review for ML experiment campaigns.

Checks for: missing baselines, missing error bars, missing ablation,
overclaimed results, missing SOTA comparison, calibration, computational
cost. Generates structured review with strengths/weaknesses/questions,
each weakness linked to a /turing: fix command. Scores 1-10.

Usage:
    python scripts/simulate_review.py
    python scripts/simulate_review.py --venue neurips --harsh
    python scripts/simulate_review.py --venue icml --json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG = "experiments/log.jsonl"
VALID_VENUES = ["neurips", "icml", "general"]
SEVERITY_WEIGHTS = {"critical": 3.0, "major": 2.0, "minor": 1.0, "nitpick": 0.3}


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


def _load_yaml_list(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with open(p) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def _w(id, sev, title, detail, fix, venues=None):
    """Shorthand weakness constructor."""
    return {"id": id, "severity": sev, "title": title, "detail": detail,
            "fix_command": fix, "venue_relevance": venues or ["neurips", "icml", "general"]}


# --- Review checks ---

def check_baselines(experiments, config):
    types = {e.get("config", {}).get("model_type", "") for e in experiments if e.get("status") == "kept"}
    baselines = {"logistic_regression", "linear_regression", "dummy", "majority_class", "random", "baseline"}
    if types and not (types & baselines):
        return _w("missing-baselines", "major", "No simple baseline comparison",
                   f"Model types: {', '.join(sorted(types))}. No simple baseline to calibrate expectations.",
                   '/turing:try "Add logistic regression baseline"')

def check_error_bars(experiments, seeds):
    kept = [e for e in experiments if e.get("status") == "kept"]
    if not kept:
        return None
    studied = {s.get("experiment_id") for s in seeds}
    unstudied = [e for e in kept if e.get("experiment_id") not in studied]
    if len(unstudied) == len(kept):
        return _w("no-error-bars", "critical", "No error bars on any result",
                   f"{len(kept)} kept experiment(s) with no seed studies. Single-seed results not publishable.",
                   "/turing:seed")
    elif unstudied:
        ids = ", ".join(e.get("experiment_id", "?") for e in unstudied[:5])
        return _w("partial-error-bars", "minor", "Some experiments lack error bars",
                   f"{len(unstudied)}/{len(kept)} lack seed studies: {ids}.", "/turing:seed")

def check_ablation(experiments, ablations):
    if len([e for e in experiments if e.get("status") == "kept"]) >= 2 and not ablations:
        return _w("no-ablation", "major", "No ablation study",
                   "No ablation studies found. Component contributions unclear.", "/turing:ablate")

def check_overclaimed(experiments, seeds, metric, lower_is_better):
    sensitive = [s for s in seeds if s.get("seed_sensitive")]
    if not sensitive:
        return None
    details = "; ".join(f"{s.get('experiment_id','?')}: CV={s.get('cv_percent',0):.1f}%" for s in sensitive)
    return _w("overclaimed-results", "major", "Seed-sensitive results may be overclaimed",
               f"{len(sensitive)} experiment(s) show high seed sensitivity: {details}. "
               "Report mean +/- std instead of point estimates.", "/turing:seed")

def check_sota(experiments, config, annotations):
    kw = {"sota", "state-of-the-art", "benchmark", "leaderboard", "published"}
    for ann in annotations:
        text = ann.get("text", "").lower()
        if any(k in text for k in kw) or any(k in [t.lower() for t in ann.get("tags", [])] for k in kw):
            return None
    if config.get("evaluation", {}).get("reference_score") or config.get("evaluation", {}).get("sota_score"):
        return None
    return _w("no-sota-comparison", "minor", "No SOTA or external benchmark comparison",
               "No reference to published results. Add reference score or annotate with SOTA values.",
               '/turing:try "Add SOTA comparison from literature"', ["neurips", "icml"])

def check_calibration(cal_results, experiments):
    kept = [e for e in experiments if e.get("status") == "kept"]
    if not kept:
        return None
    if not cal_results:
        return _w("no-calibration", "minor", "No calibration analysis",
                   "Model calibration not assessed. Reviewers expect ECE or reliability diagrams.",
                   '/turing:try "Add calibration analysis (ECE)"', ["neurips", "icml"])
    poor = [r for r in cal_results if r.get("ece", 0) > 0.1]
    if poor:
        return _w("poor-calibration", "minor", "Model poorly calibrated",
                   f"{len(poor)} model(s) have ECE > 0.1. Consider temperature scaling.",
                   '/turing:try "Apply temperature scaling"')

def check_compute_cost(experiments):
    kept = [e for e in experiments if e.get("status") == "kept"]
    if not kept:
        return None
    has_time = any(e.get("metrics", {}).get("train_seconds") is not None for e in kept)
    has_env = any(e.get("environment") for e in kept)
    issues = []
    if not has_time:
        issues.append("No training time reported")
    if not has_env:
        issues.append("No hardware info recorded")
    if issues:
        return _w("no-compute-cost", "major" if not has_time else "minor",
                   "Computational cost not reported", "; ".join(issues) + ". "
                   "Reporting compute cost is expected at all major venues.",
                   '/turing:try "Profile training and report compute cost"')

def check_diversity(experiments):
    types = {e.get("config", {}).get("model_type", "") for e in experiments if e.get("status") == "kept"}
    kept_n = sum(1 for e in experiments if e.get("status") == "kept")
    if len(types) == 1 and kept_n >= 3:
        return _w("low-diversity", "minor", "Only one model family explored",
                   f"All {kept_n} kept experiments use {list(types)[0]}. Alternatives not explored.",
                   '/turing:try "Explore alternative model architecture"')

def check_leakage(experiments, annotations):
    kw = {"leakage", "leak", "contamination", "suspicious", "too high", "too good"}
    flagged = [a for a in annotations
               if any(k in a.get("text", "").lower() for k in kw) or "leakage" in [t.lower() for t in a.get("tags", [])]]
    if flagged:
        return _w("leakage-concern", "critical", "Data leakage flagged in annotations",
                   f"{len(flagged)} annotation(s) mention potential leakage. Must investigate before submission.",
                   '/turing:try "Investigate and rule out data leakage"')

def check_reproducibility(experiments, config):
    issues = []
    if config.get("data", {}).get("random_state") is None:
        issues.append("No random state in config")
    rd = Path("experiments/reproductions")
    if len(experiments) >= 5 and not (rd.exists() and any(rd.glob("*.yaml"))):
        issues.append("No reproduction checks run")
    if issues:
        return _w("reproducibility-gaps", "minor", "Reproducibility not fully verified",
                   "; ".join(issues) + ".", "/turing:reproduce")


# --- Strengths & questions ---

def identify_strengths(experiments, seeds, ablations, config, metric, lower_is_better):
    S = []
    kept = [e for e in experiments if e.get("status") == "kept"]
    types = set(e.get("config", {}).get("model_type", "") for e in kept)
    if len(kept) >= 5:
        S.append(f"Thorough experimentation: {len(kept)} successful experiments across {len(types)} type(s).")
    stable = [s for s in seeds if not s.get("seed_sensitive")]
    if stable:
        S.append(f"Seed studies: {len(stable)}/{len(seeds)} experiments show stable results.")
    if ablations:
        S.append(f"Ablation analysis provided ({len(ablations)} study/ies).")
    families = set(e.get("family") for e in kept if e.get("family"))
    if len(families) >= 3:
        S.append(f"Systematic exploration of {len(families)} research directions.")
    if len(experiments) >= 5 and len(kept) / len(experiments) >= 0.4:
        S.append(f"High experiment efficiency: {len(kept)/len(experiments):.0%} keep rate.")
    return S or ["Experiments have been initiated on this problem."]


def generate_questions(weaknesses, experiments, config, venue):
    Q, wids = [], {w["id"] for w in weaknesses}
    qmap = {"missing-baselines": "How does performance compare to a simple baseline?",
            "no-error-bars": "Can you provide confidence intervals over multiple seeds?",
            "overclaimed-results": "Can you provide confidence intervals over multiple seeds?",
            "no-ablation": "What is the contribution of each component?",
            "no-sota-comparison": "How do results compare to published state-of-the-art?",
            "no-compute-cost": "What are the computational requirements (GPU hours)?"}
    for wid, q in qmap.items():
        if wid in wids:
            Q.append(q)
    if venue == "neurips":
        Q.append("What is the broader impact? Are there negative societal implications?")
    elif venue == "icml":
        Q.append("Is there theoretical justification, or is this purely empirical?")
    return Q


def compute_score(strengths, weaknesses, harsh):
    score = 6.0 + min(len(strengths) * 0.4, 2.0)
    for w in weaknesses:
        score -= SEVERITY_WEIGHTS.get(w["severity"], 1.0)
    if harsh:
        score -= 1.0
    return max(1, min(10, round(score)))


# --- Formatting ---

def format_review_report(strengths, weaknesses, questions, score, venue, harsh,
                          config, experiments, metric):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    kept = [e for e in experiments if e.get("status") == "kept"]
    types = set(e.get("config", {}).get("model_type", "") for e in kept)
    labels = {1: "Strong Reject", 2: "Reject", 3: "Reject", 4: "Weak Reject",
              5: "Borderline Reject", 6: "Borderline Accept", 7: "Weak Accept",
              8: "Accept", 9: "Strong Accept", 10: "Strong Accept"}
    crit = sum(1 for w in weaknesses if w["severity"] == "critical")
    maj = sum(1 for w in weaknesses if w["severity"] == "major")
    mn = sum(1 for w in weaknesses if w["severity"] == "minor")
    L = ["# Simulated Peer Review", "",
         f"*Generated {now}*",
         f"*Venue: {venue.upper()} | Mode: {'HARSH' if harsh else 'Standard'} | "
         f"Score: {score}/10 ({labels.get(score, '?')})*", "", "---", "",
         "## Summary", "",
         f"This work presents experiments on {config.get('task_description', 'the given task')} "
         f"with {len(kept)} successful experiment(s) across {len(types)} model type(s). "
         f"Primary metric: `{metric}`.", "",
         "## Score", "", f"**{score}/10** — {labels.get(score, '?')}", "",
         f"- Strengths: {len(strengths)}",
         f"- Weaknesses: {len(weaknesses)} ({crit} critical, {maj} major, {mn} minor)", "",
         "## Strengths", ""]
    for i, s in enumerate(strengths, 1):
        L.append(f"**S{i}.** {s}")
    L.extend(["", "## Weaknesses", ""])
    for i, w in enumerate(weaknesses, 1):
        L.extend([f"**W{i}. [{w['severity'].upper()}] {w['title']}**", "",
                   w["detail"], "", f"*Fix:* `{w['fix_command']}`", ""])
    if not weaknesses:
        L.extend(["No significant weaknesses identified.", ""])
    L.extend(["## Questions for Authors", ""])
    for i, q in enumerate(questions, 1):
        L.append(f"**Q{i}.** {q}")
    critical_major = [w for w in weaknesses if w["severity"] in ("critical", "major")]
    if critical_major:
        L.extend(["", "## Recommended Action Plan", "", "Address before submission:", ""])
        for p, w in enumerate(critical_major, 1):
            L.append(f"{p}. **[{w['severity'].upper()}]** {w['title']}: `{w['fix_command']}`")
    L.extend(["", "## Verdict", ""])
    if score >= 7:
        L.append("Approaching publication quality. Address minor issues and consider submission.")
    elif score >= 5:
        L.append("Borderline. Significant improvements needed. Follow the action plan.")
    else:
        L.append("Not ready. Major methodology gaps. Focus on critical and major weaknesses.")
    L.extend(["", "---", "*Simulated review by `/turing:review` — not a substitute for actual peer review.*"])
    return "\n".join(L)


def save_review_report(result: dict, output_dir="experiments/reviews") -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = p / f"review-{ts}.yaml"
    with open(out, "w") as f:
        yaml.dump({"timestamp": result["timestamp"], "venue": result["venue"],
                    "harsh": result["harsh"], "score": result["score"],
                    "weaknesses": [{"id": w["id"], "severity": w["severity"],
                                     "title": w["title"], "fix_command": w["fix_command"]}
                                    for w in result["weaknesses"]]},
                   f, default_flow_style=False, sort_keys=False)
    return out


# --- Orchestration ---

def simulate_review(venue="general", harsh=False, config_path="config.yaml",
                     log_path=DEFAULT_LOG) -> dict:
    """Run full simulated review pipeline."""
    config = load_config(config_path)
    metric = config.get("evaluation", {}).get("primary_metric", "accuracy")
    lower = config.get("evaluation", {}).get("lower_is_better", False)
    experiments = load_experiments(log_path)
    if not experiments:
        return {"error": "No experiments found. Run /turing:train first.",
                "timestamp": datetime.now(timezone.utc).isoformat()}
    seeds = _load_yaml_dir("experiments/seed_studies", "*-seeds.yaml")
    ablations = _load_yaml_dir("experiments/ablations", "*-ablation.yaml")
    cal = _load_yaml_dir("experiments/calibration", "*.yaml")
    annotations = _load_yaml_list("experiments/annotations.yaml")
    checks = [check_baselines(experiments, config), check_error_bars(experiments, seeds),
              check_ablation(experiments, ablations), check_overclaimed(experiments, seeds, metric, lower),
              check_sota(experiments, config, annotations), check_calibration(cal, experiments),
              check_compute_cost(experiments), check_diversity(experiments),
              check_leakage(experiments, annotations), check_reproducibility(experiments, config)]
    weaknesses = [c for c in checks if c and venue in c.get("venue_relevance", ["general"])]
    sev_order = {"critical": 0, "major": 1, "minor": 2, "nitpick": 3}
    weaknesses.sort(key=lambda w: sev_order.get(w["severity"], 9))
    strengths = identify_strengths(experiments, seeds, ablations, config, metric, lower)
    questions = generate_questions(weaknesses, experiments, config, venue)
    score = compute_score(strengths, weaknesses, harsh)
    report = format_review_report(strengths, weaknesses, questions, score, venue, harsh,
                                   config, experiments, metric)
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "venue": venue, "harsh": harsh,
            "score": score, "strengths": strengths, "weaknesses": weaknesses,
            "questions": questions, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate peer review of experiment campaign")
    parser.add_argument("--venue", default="general", choices=VALID_VENUES)
    parser.add_argument("--harsh", action="store_true", help="Stricter review criteria")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()
    result = simulate_review(args.venue, args.harsh, args.config, args.log)
    if "error" in result and "report" not in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(result["report"])
        saved = save_review_report(result)
        print(f"\nReview saved to {saved}", file=sys.stderr)


if __name__ == "__main__":
    main()
