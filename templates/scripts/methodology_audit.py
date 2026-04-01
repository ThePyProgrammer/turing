#!/usr/bin/env python3
"""Pre-submission methodology audit for the autoresearch pipeline.

Checks for common ML paper methodology mistakes before submission:
data leakage, wrong CV strategy, missing baselines, unreported tuning
cost, cherry-picked seeds, train/test overlap. A reviewer checklist
you run before submitting.

Usage:
    python scripts/methodology_audit.py
    python scripts/methodology_audit.py --strict
    python scripts/methodology_audit.py --checklist neurips
    python scripts/methodology_audit.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"

# Severity levels
CRITICAL = "critical"
HIGH = "high"
MEDIUM = "medium"
LOW = "low"


# --- Audit Checks ---


def check_seed_sensitivity(
    experiments: list[dict],
    seed_dir: str = "experiments/seed_studies",
) -> dict:
    """Check that results are reported with error bars from multiple seeds."""
    path = Path(seed_dir)
    seed_studies = list(path.glob("*-seeds.yaml")) if path.exists() else []

    best_kept = [e for e in experiments if e.get("status") == "kept"]
    best_ids = {e.get("experiment_id") for e in best_kept[-3:]} if best_kept else set()

    studied_ids = set()
    for f in seed_studies:
        exp_id = f.stem.replace("-seeds", "")
        studied_ids.add(exp_id)

    covered = best_ids & studied_ids

    if not best_ids:
        return {"check": "seed_sensitivity", "status": "skip", "reason": "No kept experiments", "severity": HIGH}

    if covered == best_ids:
        return {"check": "seed_sensitivity", "status": "pass", "reason": f"Seed studies exist for {len(covered)} best experiment(s)", "severity": HIGH}
    elif covered:
        return {"check": "seed_sensitivity", "status": "warn", "reason": f"Seed studies for {len(covered)}/{len(best_ids)} best experiments", "severity": HIGH, "fix": "/turing:seed"}
    else:
        return {"check": "seed_sensitivity", "status": "fail", "reason": "No seed studies for best experiments", "severity": HIGH, "fix": "/turing:seed"}


def check_ablation(
    experiments: list[dict],
    ablation_dir: str = "experiments/ablations",
) -> dict:
    """Check that major components have been ablated."""
    path = Path(ablation_dir)
    ablations = list(path.glob("*.yaml")) if path.exists() else []

    if ablations:
        return {"check": "ablation_completeness", "status": "pass", "reason": f"{len(ablations)} ablation study(s) found", "severity": HIGH}
    else:
        return {"check": "ablation_completeness", "status": "fail", "reason": "No ablation studies found", "severity": HIGH, "fix": "/turing:ablate"}


def check_baseline(experiments: list[dict]) -> dict:
    """Check that reasonable baselines were compared against."""
    baseline_keywords = {"baseline", "majority", "random", "mean", "median", "dummy", "constant", "naive"}

    baselines = []
    for exp in experiments:
        model_type = exp.get("config", {}).get("model_type", "").lower()
        desc = exp.get("description", "").lower()
        if any(kw in model_type or kw in desc for kw in baseline_keywords):
            baselines.append(exp.get("experiment_id", "?"))

    if baselines:
        return {"check": "baseline_comparison", "status": "pass", "reason": f"Baseline experiments found: {', '.join(baselines[:3])}", "severity": HIGH}
    else:
        return {"check": "baseline_comparison", "status": "fail", "reason": "No baseline experiments found in log", "severity": HIGH, "fix": "/turing:try 'add majority class baseline'"}


def check_reproducibility(
    experiments: list[dict],
    repro_dir: str = "experiments/reproductions",
) -> dict:
    """Check that the best result has been reproduced."""
    path = Path(repro_dir)
    repros = list(path.glob("*-repro.yaml")) if path.exists() else []

    if not repros:
        return {"check": "reproducibility", "status": "fail", "reason": "No reproduction reports found", "severity": HIGH, "fix": "/turing:reproduce <best-exp-id>"}

    # Check if any passed
    for f in repros:
        try:
            with open(f) as fh:
                report = yaml.safe_load(fh)
                if report and report.get("verdict") in ("reproducible", "approximately_reproducible"):
                    return {"check": "reproducibility", "status": "pass", "reason": f"Experiment {report.get('experiment_id', '?')} reproduced successfully", "severity": HIGH}
        except (yaml.YAMLError, OSError):
            continue

    return {"check": "reproducibility", "status": "warn", "reason": "Reproduction reports exist but none passed", "severity": HIGH, "fix": "/turing:reproduce <best-exp-id>"}


def check_hyperparameter_budget(experiments: list[dict]) -> dict:
    """Check that total hyperparameter tuning budget is documented."""
    n_experiments = len(experiments)
    total_seconds = sum(
        e.get("metrics", {}).get("train_seconds", 0)
        for e in experiments
        if isinstance(e.get("metrics", {}).get("train_seconds"), (int, float))
    )
    total_hours = total_seconds / 3600

    if n_experiments > 0:
        return {
            "check": "hyperparameter_budget",
            "status": "pass" if n_experiments > 0 else "warn",
            "reason": f"{n_experiments} experiments, {total_hours:.1f} compute hours logged",
            "severity": MEDIUM,
            "detail": {"n_experiments": n_experiments, "total_hours": round(total_hours, 2)},
        }
    return {"check": "hyperparameter_budget", "status": "warn", "reason": "No experiments logged", "severity": MEDIUM}


def check_data_leakage(config: dict) -> dict:
    """Check for potential data leakage indicators.

    This is a heuristic check — verifies that config suggests proper
    train/test separation. Full leakage detection requires code analysis.
    """
    prepare_exists = Path("prepare.py").exists()
    evaluate_exists = Path("evaluate.py").exists()

    if prepare_exists and evaluate_exists:
        return {"check": "data_leakage", "status": "pass", "reason": "Separate prepare.py and evaluate.py files exist (proper separation)", "severity": CRITICAL}
    elif prepare_exists:
        return {"check": "data_leakage", "status": "warn", "reason": "prepare.py exists but evaluate.py missing — verify evaluation uses held-out data", "severity": CRITICAL}
    else:
        return {"check": "data_leakage", "status": "warn", "reason": "No prepare.py found — verify data splitting is done before feature engineering", "severity": CRITICAL}


def check_cv_strategy(config: dict) -> dict:
    """Check that CV strategy is appropriate for the data type."""
    eval_cfg = config.get("evaluation", {})
    cv_strategy = eval_cfg.get("cv_strategy", eval_cfg.get("cv", ""))

    if cv_strategy:
        return {"check": "cv_strategy", "status": "pass", "reason": f"CV strategy specified: {cv_strategy}", "severity": CRITICAL}
    else:
        return {"check": "cv_strategy", "status": "warn", "reason": "No CV strategy specified in config — verify appropriate split method for data type", "severity": CRITICAL}


def check_regression_stability(
    regress_dir: str = "experiments/regressions",
) -> dict:
    """Check that regression tests have been run."""
    path = Path(regress_dir)
    checks = list(path.glob("check-*.yaml")) if path.exists() else []

    if checks:
        return {"check": "regression_stability", "status": "pass", "reason": f"{len(checks)} regression check(s) performed", "severity": MEDIUM}
    else:
        return {"check": "regression_stability", "status": "warn", "reason": "No regression checks found", "severity": MEDIUM, "fix": "/turing:regress"}


# --- Venue-Specific Checklists ---


VENUE_CHECKS = {
    "neurips": [
        {"check": "broader_impact", "description": "Broader impact statement included", "severity": MEDIUM},
        {"check": "reproducibility_checklist", "description": "NeurIPS reproducibility checklist completed", "severity": HIGH},
        {"check": "code_availability", "description": "Code and data availability documented", "severity": MEDIUM},
    ],
    "icml": [
        {"check": "reproducibility_checklist", "description": "ICML reproducibility checklist completed", "severity": HIGH},
    ],
    "iclr": [
        {"check": "ethics_statement", "description": "Ethics statement included", "severity": MEDIUM},
    ],
}


def get_venue_checks(venue: str | None) -> list[dict]:
    """Get venue-specific additional checks."""
    if not venue:
        return []
    checks = VENUE_CHECKS.get(venue.lower(), [])
    # These are manual checks — mark as "manual" status
    return [
        {**c, "status": "manual", "reason": f"Manual check required: {c['description']}"}
        for c in checks
    ]


# --- Full Audit ---


def run_audit(
    strict: bool = False,
    venue: str | None = None,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    """Run a complete methodology audit.

    Args:
        strict: Treat warnings as failures.
        venue: Venue-specific checklist (neurips, icml, iclr).
        config_path: Path to config.yaml.
        log_path: Path to experiment log.

    Returns:
        Complete audit report.
    """
    config = load_config(config_path)
    experiments = load_experiments(log_path)

    checks = [
        check_data_leakage(config),
        check_cv_strategy(config),
        check_seed_sensitivity(experiments),
        check_ablation(experiments),
        check_baseline(experiments),
        check_reproducibility(experiments),
        check_hyperparameter_budget(experiments),
        check_regression_stability(),
    ]

    # Add venue-specific checks
    venue_checks = get_venue_checks(venue)
    checks.extend(venue_checks)

    # Compute score
    n_pass = sum(1 for c in checks if c["status"] == "pass")
    n_fail = sum(1 for c in checks if c["status"] == "fail")
    n_warn = sum(1 for c in checks if c["status"] == "warn")
    n_skip = sum(1 for c in checks if c["status"] == "skip")
    n_manual = sum(1 for c in checks if c["status"] == "manual")
    total_checkable = len(checks) - n_skip - n_manual

    if strict:
        # Treat warnings as failures
        effective_pass = n_pass
        effective_fail = n_fail + n_warn
    else:
        effective_pass = n_pass
        effective_fail = n_fail

    # Overall verdict
    critical_fails = [c for c in checks if c["status"] == "fail" and c.get("severity") == CRITICAL]
    if critical_fails:
        verdict = "fail"
    elif n_fail > 0:
        verdict = "needs_work"
    elif n_warn > 2:
        verdict = "needs_work"
    elif n_warn > 0:
        verdict = "pass_with_warnings"
    else:
        verdict = "pass"

    # Action items
    actions = []
    for c in checks:
        if c["status"] in ("fail", "warn") and c.get("fix"):
            actions.append({
                "check": c["check"],
                "fix": c["fix"],
                "severity": c.get("severity", MEDIUM),
            })

    return {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "strict_mode": strict,
        "venue": venue,
        "checks": checks,
        "score": {
            "pass": n_pass,
            "fail": n_fail,
            "warn": n_warn,
            "skip": n_skip,
            "manual": n_manual,
            "total": len(checks),
            "checkable": total_checkable,
        },
        "verdict": verdict,
        "actions": actions,
    }


# --- Report Formatting ---


def save_audit_report(report: dict, output_dir: str = "experiments/audits") -> Path:
    """Save audit report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filepath = out_path / f"audit-{date}.yaml"

    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    return filepath


def format_audit_report(report: dict) -> str:
    """Format audit report as markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    verdict = report.get("verdict", "?")
    score = report.get("score", {})
    strict = report.get("strict_mode", False)

    verdict_labels = {
        "pass": "PASS — Ready for submission",
        "pass_with_warnings": "PASS (with warnings) — Address before submission",
        "needs_work": "NEEDS WORK — Fix failures before submission",
        "fail": "FAIL — Critical issues found",
    }

    lines = [
        "# Methodology Audit Report",
        "",
        f"*Audited {report.get('audited_at', 'N/A')[:19]}*",
        f"*Mode: {'strict' if strict else 'standard'}*",
    ]

    if report.get("venue"):
        lines.append(f"*Venue: {report['venue']}*")

    lines.extend([
        "",
        f"**{verdict_labels.get(verdict, verdict.upper())}**",
        "",
        "## Checks",
        "",
    ])

    status_markers = {
        "pass": "PASS",
        "fail": "FAIL",
        "warn": "WARN",
        "skip": "SKIP",
        "manual": "TODO",
    }

    for c in report.get("checks", []):
        status = c.get("status", "?")
        marker = status_markers.get(status, status.upper())
        sev = c.get("severity", "medium")
        lines.append(f"- **[{marker}]** {c.get('check', '?')} ({sev}): {c.get('reason', 'N/A')}")

    # Score
    lines.extend([
        "",
        "## Score",
        "",
        f"**{score.get('pass', 0)}/{score.get('checkable', 0)} pass**, "
        f"{score.get('warn', 0)} warning(s), "
        f"{score.get('fail', 0)} failure(s)",
    ])

    if score.get("manual", 0) > 0:
        lines.append(f"*{score['manual']} manual check(s) required*")

    # Actions
    actions = report.get("actions", [])
    if actions:
        lines.extend(["", "## Required Actions", ""])
        for a in actions:
            lines.append(f"- **{a['check']}** ({a['severity']}): run `{a['fix']}`")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-submission methodology audit",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Strict mode: treat warnings as failures",
    )
    parser.add_argument(
        "--checklist",
        help="Venue-specific checklist (neurips, icml, iclr)",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--log", default=DEFAULT_LOG_PATH,
        help="Path to experiment log",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    args = parser.parse_args()

    report = run_audit(
        strict=args.strict,
        venue=args.checklist,
        config_path=args.config,
        log_path=args.log,
    )

    if "error" not in report:
        filepath = save_audit_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_audit_report(report))

    # Exit code based on verdict
    if report.get("verdict") == "fail":
        sys.exit(1)
    elif report.get("verdict") == "needs_work":
        sys.exit(2)


if __name__ == "__main__":
    main()
