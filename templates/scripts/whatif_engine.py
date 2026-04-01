#!/usr/bin/env python3
"""What-if analysis engine for the autoresearch pipeline.

Routes hypothetical questions to existing estimators: scaling laws,
ablation, sensitivity, ensemble, pruning, and stitch. Returns an
estimate with confidence without running new experiments.

Usage:
    python scripts/whatif_engine.py "what if I had 2x more data"
    python scripts/whatif_engine.py "what if I removed class 3"
    python scripts/whatif_engine.py "what if I combined exp-031 with exp-042"
    python scripts/whatif_engine.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"

# Question patterns mapped to estimator routes
ROUTE_PATTERNS = [
    {
        "name": "scaling",
        "patterns": [
            r"(?:more|less|2x|3x|4x|5x|10x|half|double|triple)\s+(?:data|samples|training data|examples)",
            r"(?:data|dataset)\s+(?:size|scaling|increase|decrease)",
            r"(?:scale|scaled)\s+(?:up|down)\s+(?:data|the data)",
        ],
        "source": "scaling law fit from `/turing:scale`",
        "verify_cmd": "/turing:scale --extrapolate",
        "data_dir": "experiments/scaling",
    },
    {
        "name": "ablation",
        "patterns": [
            r"(?:remov\w*|drop\w*|exclud\w*|without|ablat\w*)\s+(?:class|feature|component|column|variable)",
            r"(?:class|feature|component)\s+(?:\d+|[A-Za-z_]+)\s+(?:remov\w*|drop\w*|exclud\w*)",
        ],
        "source": "ablation study from `/turing:ablate`",
        "verify_cmd": "/turing:ablate",
        "data_dir": "experiments/ablations",
    },
    {
        "name": "stitch",
        "patterns": [
            r"(?:combine|stitch|swap|mix)\s+.*(?:from|of)\s+exp-\d+\s+(?:with|and)\s+.*exp-\d+",
            r"(?:pipeline|stage)\s+(?:from|of)\s+exp-\d+",
        ],
        "source": "pipeline composition from `/turing:stitch`",
        "verify_cmd": "/turing:stitch",
        "data_dir": "experiments/cache",
    },
    {
        "name": "sensitivity",
        "patterns": [
            r"(?:different|change|modify|adjust|set)\s+(?:hyperparameter|learning.?rate|lr|depth|estimators|epochs|batch.?size)",
            r"(?:learning.?rate|lr|depth|max_depth|n_estimators|epochs|batch.?size)\s+(?:was|were|to|=|of)\s+[\d.]+",
        ],
        "source": "sensitivity interpolation from `/turing:sensitivity`",
        "verify_cmd": "/turing:sensitivity",
        "data_dir": "experiments/sensitivity",
    },
    {
        "name": "ensemble",
        "patterns": [
            r"(?:ensembl\w*|combine|blend\w*|stack\w*|vot\w*)\s+(?:(?:these|the|top|best)\s+)*(?:models|experiments)",
            r"(?:voting|stacking|blending)\s+(?:of|with|from)",
        ],
        "source": "prediction correlation from `/turing:ensemble`",
        "verify_cmd": "/turing:ensemble",
        "data_dir": "experiments/ensembles",
    },
    {
        "name": "pruning",
        "patterns": [
            r"(?:prune|pruning|sparsity)\s+(?:to|at|of)?\s*\d+",
            r"\d+%?\s*(?:sparsity|sparse|pruned)",
        ],
        "source": "pruning sweep interpolation from `/turing:prune`",
        "verify_cmd": "/turing:prune",
        "data_dir": "experiments/pruning",
    },
    {
        "name": "budget",
        "patterns": [
            r"(?:spend|spent|budget|allocate|invest)\s+.*(?:budget|remaining).*(?:on|in|for)",
            r"(?:remaining|left)\s+(?:budget|experiments|compute)",
            r"(?:budget)\s+(?:on|for|between)",
        ],
        "source": "budget allocation from `/turing:budget`",
        "verify_cmd": "/turing:budget",
        "data_dir": None,
    },
]


# --- Question Parsing ---


def classify_question(question: str) -> dict:
    """Classify a what-if question into a route.

    Args:
        question: Natural language question.

    Returns:
        Route dict with name, source, verify_cmd, or unknown route.
    """
    q_lower = question.lower().strip()

    for route in ROUTE_PATTERNS:
        for pattern in route["patterns"]:
            if re.search(pattern, q_lower):
                return {
                    "route": route["name"],
                    "source": route["source"],
                    "verify_cmd": route["verify_cmd"],
                    "data_dir": route["data_dir"],
                    "matched_pattern": pattern,
                }

    return {
        "route": "unknown",
        "source": None,
        "verify_cmd": None,
        "data_dir": None,
        "matched_pattern": None,
    }


def extract_multiplier(question: str) -> float | None:
    """Extract a data multiplier from the question (e.g., '2x' -> 2.0)."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*x\s+(?:more|the)?\s*(?:data|samples)", question.lower())
    if match:
        return float(match.group(1))

    if re.search(r"(?:double|twice)\s+(?:the\s+)?(?:data|samples)", question.lower()):
        return 2.0
    if re.search(r"(?:triple)\s+(?:the\s+)?(?:data|samples)", question.lower()):
        return 3.0
    if re.search(r"(?:half)\s+(?:the\s+)?(?:data|samples)", question.lower()):
        return 0.5

    return None


def extract_experiment_ids(question: str) -> list[str]:
    """Extract experiment IDs from the question."""
    return re.findall(r"exp-(\d+)", question.lower())


def extract_target_value(question: str) -> float | None:
    """Extract a target numeric value (e.g., sparsity percentage, hyperparameter value)."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", question)
    if match:
        return float(match.group(1))
    return None


# --- Estimators ---


def estimate_scaling(
    question: str,
    experiments: list[dict],
    primary_metric: str,
    scaling_dir: str = "experiments/scaling",
) -> dict:
    """Estimate metric change from data scaling.

    Uses existing scaling law data if available, otherwise extrapolates
    from experiment history.
    """
    multiplier = extract_multiplier(question)
    if multiplier is None:
        return {"error": "Could not parse data multiplier from question"}

    # Try loading existing scaling data
    scaling_path = Path(scaling_dir)
    scaling_results = []
    if scaling_path.exists():
        for f in scaling_path.glob("*.yaml"):
            with open(f) as fh:
                data = yaml.safe_load(fh)
                if isinstance(data, dict) and "fit" in data:
                    scaling_results.append(data)

    if scaling_results:
        # Use the most recent scaling fit
        fit = scaling_results[-1].get("fit", {})
        a = fit.get("a", 0)
        b = fit.get("b", 0)
        c = fit.get("c", 0)
        r_squared = fit.get("r_squared", 0)

        current_metric = _get_best_metric(experiments, primary_metric)
        predicted = a * (multiplier ** b) + c

        confidence = _r_squared_to_confidence(r_squared)
        return {
            "estimate": round(predicted, 4),
            "current": current_metric,
            "delta": round(predicted - current_metric, 4) if current_metric else None,
            "confidence": confidence,
            "confidence_detail": f"R²={r_squared:.3f} on scaling curve",
            "multiplier": multiplier,
        }

    # Fallback: rough extrapolation from experiment count
    current_metric = _get_best_metric(experiments, primary_metric)
    if current_metric is None:
        return {"error": "No experiments with metrics found"}

    # Conservative log-based estimate
    import math
    delta_estimate = current_metric * 0.01 * math.log2(multiplier)
    predicted = current_metric + delta_estimate

    return {
        "estimate": round(predicted, 4),
        "current": current_metric,
        "delta": round(delta_estimate, 4),
        "confidence": "LOW",
        "confidence_detail": "No scaling data — using conservative log extrapolation",
        "multiplier": multiplier,
    }


def estimate_ablation(
    question: str,
    experiments: list[dict],
    primary_metric: str,
    ablation_dir: str = "experiments/ablations",
) -> dict:
    """Estimate impact of removing a class/feature from ablation data."""
    ablation_path = Path(ablation_dir)
    if not ablation_path.exists() or not list(ablation_path.glob("*.yaml")):
        return {"error": "No ablation data available. Run `/turing:ablate` first."}

    # Load the most recent ablation study
    ablation_files = sorted(ablation_path.glob("*.yaml"))
    with open(ablation_files[-1]) as f:
        ablation_data = yaml.safe_load(f)

    if not isinstance(ablation_data, dict):
        return {"error": "Malformed ablation data"}

    components = ablation_data.get("components", ablation_data.get("results", []))
    if not components:
        return {"error": "No component ablation results found"}

    current_metric = _get_best_metric(experiments, primary_metric)

    # Find the component mentioned in the question
    q_lower = question.lower()
    matched = None
    for comp in components if isinstance(components, list) else []:
        name = comp.get("component", comp.get("name", "")).lower()
        if name and name in q_lower:
            matched = comp
            break

    if matched:
        impact = matched.get("impact", matched.get("metric_delta", 0))
        return {
            "estimate": round(current_metric + impact, 4) if current_metric else None,
            "current": current_metric,
            "delta": round(impact, 4),
            "component": matched.get("component", matched.get("name")),
            "confidence": "HIGH",
            "confidence_detail": "Direct ablation data available",
        }

    # General summary if no specific match
    avg_impact = sum(
        abs(c.get("impact", c.get("metric_delta", 0)))
        for c in (components if isinstance(components, list) else [])
    ) / max(len(components) if isinstance(components, list) else 1, 1)

    return {
        "estimate": None,
        "current": current_metric,
        "delta": None,
        "confidence": "LOW",
        "confidence_detail": f"No exact match — average component impact is ±{avg_impact:.4f}",
        "available_components": [
            c.get("component", c.get("name")) for c in (components if isinstance(components, list) else [])
        ],
    }


def estimate_sensitivity(
    question: str,
    experiments: list[dict],
    primary_metric: str,
    sensitivity_dir: str = "experiments/sensitivity",
) -> dict:
    """Estimate hyperparameter change impact from sensitivity data."""
    sens_path = Path(sensitivity_dir)
    if not sens_path.exists() or not list(sens_path.glob("*.yaml")):
        return {"error": "No sensitivity data available. Run `/turing:sensitivity` first."}

    sens_files = sorted(sens_path.glob("*.yaml"))
    with open(sens_files[-1]) as f:
        sens_data = yaml.safe_load(f)

    if not isinstance(sens_data, dict):
        return {"error": "Malformed sensitivity data"}

    sensitivities = sens_data.get("sensitivities", [])
    current_metric = _get_best_metric(experiments, primary_metric)

    q_lower = question.lower()
    for sens in sensitivities:
        param = sens.get("param", "").lower()
        if param and param.replace("_", " ") in q_lower.replace("_", " "):
            return {
                "estimate": None,
                "current": current_metric,
                "param": sens.get("param"),
                "sensitivity_level": sens.get("level", "UNKNOWN"),
                "metric_range": [sens.get("metric_min"), sens.get("metric_max")],
                "best_value": sens.get("best_value"),
                "confidence": "MED" if sens.get("level") in ("HIGH", "MED") else "LOW",
                "confidence_detail": f"Sensitivity level: {sens.get('level')}, range: {sens.get('metric_min')}-{sens.get('metric_max')}",
            }

    return {
        "error": "Parameter not found in sensitivity data",
        "available_params": [s.get("param") for s in sensitivities],
    }


def estimate_ensemble(
    question: str,
    experiments: list[dict],
    primary_metric: str,
    ensemble_dir: str = "experiments/ensembles",
) -> dict:
    """Estimate ensemble improvement from prior ensemble data."""
    ens_path = Path(ensemble_dir)
    current_metric = _get_best_metric(experiments, primary_metric)

    if ens_path.exists() and list(ens_path.glob("*.yaml")):
        ens_files = sorted(ens_path.glob("*.yaml"))
        with open(ens_files[-1]) as f:
            ens_data = yaml.safe_load(f)
        if isinstance(ens_data, dict):
            best_method = ens_data.get("best_method", {})
            ens_metric = best_method.get("metric", best_method.get(primary_metric))
            if ens_metric is not None:
                return {
                    "estimate": round(ens_metric, 4),
                    "current": current_metric,
                    "delta": round(ens_metric - current_metric, 4) if current_metric else None,
                    "method": best_method.get("method", "unknown"),
                    "confidence": "HIGH",
                    "confidence_detail": "Prior ensemble result available",
                }

    # Conservative estimate: ensembles typically improve 1-3%
    if current_metric is not None:
        delta = current_metric * 0.015
        return {
            "estimate": round(current_metric + delta, 4),
            "current": current_metric,
            "delta": round(delta, 4),
            "confidence": "LOW",
            "confidence_detail": "No prior ensemble data — using typical 1.5% improvement estimate",
        }

    return {"error": "No experiments or ensemble data available"}


def estimate_pruning(
    question: str,
    experiments: list[dict],
    primary_metric: str,
    pruning_dir: str = "experiments/pruning",
) -> dict:
    """Estimate pruning impact from prior pruning sweep data."""
    target_sparsity = extract_target_value(question)
    prune_path = Path(pruning_dir)

    if not prune_path.exists() or not list(prune_path.glob("*.yaml")):
        return {"error": "No pruning data available. Run `/turing:prune` first."}

    prune_files = sorted(prune_path.glob("*.yaml"))
    with open(prune_files[-1]) as f:
        prune_data = yaml.safe_load(f)

    if not isinstance(prune_data, dict):
        return {"error": "Malformed pruning data"}

    current_metric = _get_best_metric(experiments, primary_metric)
    sweep = prune_data.get("sweep_results", prune_data.get("results", []))

    if target_sparsity is not None and sweep:
        # Interpolate from sweep
        sparsities = [s.get("sparsity", 0) for s in sweep]
        metrics = [s.get("metric", s.get(primary_metric, 0)) for s in sweep]

        if sparsities and metrics:
            predicted = _linear_interpolate(sparsities, metrics, target_sparsity)
            if predicted is not None:
                return {
                    "estimate": round(predicted, 4),
                    "current": current_metric,
                    "delta": round(predicted - current_metric, 4) if current_metric else None,
                    "target_sparsity": target_sparsity,
                    "confidence": "MED",
                    "confidence_detail": f"Interpolated from {len(sweep)} pruning sweep points",
                }

    # Return available data
    return {
        "estimate": None,
        "current": current_metric,
        "confidence": "LOW",
        "confidence_detail": "Could not interpolate — check pruning sweep data",
        "available_sparsities": [s.get("sparsity") for s in sweep] if sweep else [],
    }


def estimate_stitch(
    question: str,
    experiments: list[dict],
    primary_metric: str,
) -> dict:
    """Estimate pipeline stitch impact."""
    exp_ids = extract_experiment_ids(question)
    if len(exp_ids) < 2:
        return {"error": "Need at least 2 experiment IDs to estimate stitch (e.g., exp-031 and exp-042)"}

    current_metric = _get_best_metric(experiments, primary_metric)

    # Look up the referenced experiments
    ref_metrics = []
    for eid in exp_ids:
        full_id = f"exp-{eid}"
        for exp in experiments:
            if exp.get("experiment_id") == full_id:
                m = exp.get("metrics", {}).get(primary_metric)
                if m is not None:
                    ref_metrics.append({"id": full_id, "metric": m})
                break

    if len(ref_metrics) < 2:
        return {
            "error": f"Could not find metrics for referenced experiments",
            "found": ref_metrics,
        }

    # Conservative estimate: best of the two + small bonus
    best = max(m["metric"] for m in ref_metrics)
    delta = abs(ref_metrics[0]["metric"] - ref_metrics[1]["metric"]) * 0.3
    predicted = best + delta

    return {
        "estimate": round(predicted, 4),
        "current": current_metric,
        "delta": round(predicted - current_metric, 4) if current_metric else None,
        "experiments": ref_metrics,
        "confidence": "LOW",
        "confidence_detail": "Estimated from individual metrics — actual stitch may differ",
    }


def estimate_budget(
    question: str,
    experiments: list[dict],
    primary_metric: str,
) -> dict:
    """Estimate budget allocation impact."""
    current_metric = _get_best_metric(experiments, primary_metric)
    total_exps = len(experiments)
    kept = sum(1 for e in experiments if e.get("status") == "kept")

    return {
        "estimate": None,
        "current": current_metric,
        "total_experiments": total_exps,
        "kept_ratio": round(kept / total_exps, 2) if total_exps > 0 else 0,
        "confidence": "LOW",
        "confidence_detail": "Budget allocation requires simulation — use `/turing:simulate` for prediction",
    }


# --- Helpers ---


def _get_best_metric(experiments: list[dict], metric: str) -> float | None:
    """Get the best metric value from all experiments."""
    values = []
    for exp in experiments:
        v = exp.get("metrics", {}).get(metric)
        if v is not None:
            values.append(v)
    return max(values) if values else None


def _r_squared_to_confidence(r_squared: float) -> str:
    """Map R² to confidence level."""
    if r_squared >= 0.95:
        return "HIGH"
    elif r_squared >= 0.80:
        return "MED"
    else:
        return "LOW"


def _linear_interpolate(xs: list[float], ys: list[float], target_x: float) -> float | None:
    """Simple linear interpolation."""
    if not xs or not ys or len(xs) != len(ys):
        return None

    pairs = sorted(zip(xs, ys))
    xs_sorted = [p[0] for p in pairs]
    ys_sorted = [p[1] for p in pairs]

    if target_x <= xs_sorted[0]:
        return ys_sorted[0]
    if target_x >= xs_sorted[-1]:
        return ys_sorted[-1]

    for i in range(len(xs_sorted) - 1):
        if xs_sorted[i] <= target_x <= xs_sorted[i + 1]:
            t = (target_x - xs_sorted[i]) / (xs_sorted[i + 1] - xs_sorted[i])
            return ys_sorted[i] + t * (ys_sorted[i + 1] - ys_sorted[i])

    return None


# --- Main Pipeline ---


ESTIMATOR_MAP = {
    "scaling": estimate_scaling,
    "ablation": estimate_ablation,
    "sensitivity": estimate_sensitivity,
    "ensemble": estimate_ensemble,
    "pruning": estimate_pruning,
    "stitch": estimate_stitch,
    "budget": estimate_budget,
}


def whatif_analysis(
    question: str,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    """Run what-if analysis for a hypothetical question.

    Args:
        question: Natural language what-if question.
        config_path: Path to config.yaml.
        log_path: Path to experiment log.

    Returns:
        What-if analysis result with estimate, confidence, and recommendation.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")

    experiments = load_experiments(log_path)

    classification = classify_question(question)
    route = classification["route"]

    if route == "unknown":
        return {
            "question": question,
            "route": "unknown",
            "error": "Cannot classify question — no matching estimator found",
            "available_routes": [r["name"] for r in ROUTE_PATTERNS],
            "suggestion": "Try phrasing as: 'what if I had Nx more data', "
                          "'what if I removed <component>', "
                          "'what if I changed <hyperparameter>'",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    estimator = ESTIMATOR_MAP.get(route)
    if estimator is None:
        return {"error": f"No estimator for route: {route}"}

    # Build kwargs based on estimator signature
    kwargs = {
        "question": question,
        "experiments": experiments,
        "primary_metric": primary_metric,
    }

    # Add data_dir if the estimator accepts it
    data_dir = classification.get("data_dir")
    if data_dir and route in ("scaling", "ablation", "sensitivity", "ensemble", "pruning"):
        dir_param_names = {
            "scaling": "scaling_dir",
            "ablation": "ablation_dir",
            "sensitivity": "sensitivity_dir",
            "ensemble": "ensemble_dir",
            "pruning": "pruning_dir",
        }
        if route in dir_param_names:
            kwargs[dir_param_names[route]] = data_dir

    result = estimator(**kwargs)

    # Build recommendation
    recommendation = _generate_recommendation(result, route, classification)

    return {
        "question": question,
        "route": route,
        "source": classification["source"],
        "verify_cmd": classification["verify_cmd"],
        "result": result,
        "recommendation": recommendation,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _generate_recommendation(result: dict, route: str, classification: dict) -> str:
    """Generate a human-readable recommendation from the result."""
    if "error" in result:
        return f"Cannot estimate — {result['error']}"

    estimate = result.get("estimate")
    current = result.get("current")
    delta = result.get("delta")
    confidence = result.get("confidence", "UNKNOWN")

    if estimate is None or current is None:
        return f"Insufficient data for a point estimate. Run `{classification.get('verify_cmd', 'the source command')}` first."

    if delta is not None:
        if abs(delta) < 0.001:
            return "Marginal gain. Likely not worth the effort."
        elif delta > 0.01:
            return f"Promising (+{delta:.4f}). Worth investigating further."
        elif delta > 0:
            return f"Small gain (+{delta:.4f}). Consider other approaches first."
        else:
            return f"Negative impact ({delta:.4f}). Avoid this direction."

    return f"Estimate available ({confidence} confidence). Verify with `{classification.get('verify_cmd')}`."


# --- Report Formatting ---


def save_whatif_report(report: dict, output_dir: str = "experiments/whatif") -> Path:
    """Save what-if analysis report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filepath = out_path / f"whatif-{ts}.yaml"
    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_whatif_report(report: dict) -> str:
    """Format what-if report as readable markdown."""
    if "error" in report and "result" not in report:
        return f"ERROR: {report['error']}"

    lines = ["# What-If Analysis", ""]
    lines.append(f"**Question:** {report.get('question', 'N/A')}")
    lines.append(f"**Route:** {report.get('route', 'unknown')}")
    lines.append(f"**Source:** {report.get('source', 'N/A')}")
    lines.append("")

    result = report.get("result", {})

    if "error" in result:
        lines.append(f"**Error:** {result['error']}")
        if "available_routes" in report:
            lines.append("")
            lines.append("Available routes: " + ", ".join(report["available_routes"]))
        if "suggestion" in report:
            lines.append("")
            lines.append(f"**Suggestion:** {report['suggestion']}")
    else:
        current = result.get("current")
        estimate = result.get("estimate")
        delta = result.get("delta")
        confidence = result.get("confidence", "UNKNOWN")

        if current is not None:
            lines.append(f"**Current best:** {current}")
        if estimate is not None:
            lines.append(f"**Estimated:** {estimate}")
        if delta is not None:
            sign = "+" if delta >= 0 else ""
            lines.append(f"**Delta:** {sign}{delta}")
        lines.append(f"**Confidence:** {confidence}")

        detail = result.get("confidence_detail")
        if detail:
            lines.append(f"**Detail:** {detail}")

    lines.append("")
    rec = report.get("recommendation")
    if rec:
        lines.append(f"**Recommendation:** {rec}")

    verify = report.get("verify_cmd")
    if verify:
        lines.append(f"**To verify:** run `{verify}`")

    lines.append("")
    lines.append(f"*Generated: {report.get('generated_at', 'N/A')}*")
    return "\n".join(lines)


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="What-if analysis engine — answer hypotheticals from existing data"
    )
    parser.add_argument("question", nargs="?", help="What-if question to analyze")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    if not args.question:
        parser.error("Please provide a what-if question")

    report = whatif_analysis(
        question=args.question,
        config_path=args.config,
        log_path=args.log,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_whatif_report(report))

    # Save report
    saved = save_whatif_report(report)
    if not args.json:
        print(f"\nSaved: {saved}")


if __name__ == "__main__":
    main()
