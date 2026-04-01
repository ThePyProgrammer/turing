#!/usr/bin/env python3
"""Automated feature selection and generation for the autoresearch pipeline.

Runs multiple feature importance methods (mutual information, permutation,
L1, tree-based), computes consensus ranking, detects redundancy, and
generates candidate interaction features.

Usage:
    python scripts/feature_intelligence.py
    python scripts/feature_intelligence.py --method all
    python scripts/feature_intelligence.py --method importance --top-k 15
    python scripts/feature_intelligence.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from scripts.turing_io import load_config

DEFAULT_TOP_K = 20
REDUNDANCY_THRESHOLD = 0.95


# --- Importance Methods ---


def mutual_information_ranking(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
) -> list[dict]:
    """Rank features by mutual information with the target."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Detect task type from target
    unique_vals = len(np.unique(y))
    if unique_vals <= 20:  # Classification heuristic
        scores = mutual_info_classif(X, y, random_state=42)
    else:
        scores = mutual_info_regression(X, y, random_state=42)

    ranked = sorted(
        [{"feature": feature_names[i], "score": round(float(scores[i]), 6), "rank": 0}
         for i in range(len(scores))],
        key=lambda x: x["score"], reverse=True,
    )
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    return ranked


def l1_ranking(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
) -> list[dict]:
    """Rank features by L1 regularization coefficient magnitude."""
    from sklearn.linear_model import Lasso, LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    unique_vals = len(np.unique(y))
    if unique_vals <= 20:
        model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=1000)
        model.fit(X_scaled, y)
        coefs = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_.ravel())
    else:
        model = Lasso(alpha=0.01, max_iter=1000)
        model.fit(X_scaled, y)
        coefs = np.abs(model.coef_)

    ranked = sorted(
        [{"feature": feature_names[i], "score": round(float(coefs[i]), 6), "rank": 0}
         for i in range(len(coefs))],
        key=lambda x: x["score"], reverse=True,
    )
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    return ranked


def tree_importance_ranking(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
) -> list[dict]:
    """Rank features by tree-based importance."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    unique_vals = len(np.unique(y))
    if unique_vals <= 20:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X, y)
    importances = model.feature_importances_

    ranked = sorted(
        [{"feature": feature_names[i], "score": round(float(importances[i]), 6), "rank": 0}
         for i in range(len(importances))],
        key=lambda x: x["score"], reverse=True,
    )
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    return ranked


# --- Consensus ---


def compute_consensus(
    rankings: dict[str, list[dict]],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Compute consensus ranking across multiple methods.

    A feature's consensus score = number of methods that place it in top-K.
    """
    n_methods = len(rankings)
    feature_scores = {}

    for method_name, ranking in rankings.items():
        top_features = {r["feature"] for r in ranking[:top_k]}
        for feat in top_features:
            if feat not in feature_scores:
                feature_scores[feat] = {"feature": feat, "methods": {}, "consensus": 0}
            feature_scores[feat]["methods"][method_name] = next(
                (r["rank"] for r in ranking if r["feature"] == feat), None
            )
            feature_scores[feat]["consensus"] += 1

    # Add features not in any top-K
    all_features = set()
    for ranking in rankings.values():
        for r in ranking:
            all_features.add(r["feature"])

    for feat in all_features:
        if feat not in feature_scores:
            feature_scores[feat] = {
                "feature": feat,
                "methods": {m: next((r["rank"] for r in rk if r["feature"] == feat), None) for m, rk in rankings.items()},
                "consensus": 0,
            }

    result = sorted(feature_scores.values(), key=lambda x: (-x["consensus"], x["feature"]))

    for r in result:
        r["consensus_str"] = f"{r['consensus']}/{n_methods}"
        if r["consensus"] == n_methods:
            r["consensus_str"] += " ★"
        elif r["consensus"] == 0:
            r["consensus_str"] += " — DROP"

    return result


# --- Redundancy Detection ---


def detect_redundancy(
    X: np.ndarray,
    feature_names: list[str] | None = None,
    threshold: float = REDUNDANCY_THRESHOLD,
) -> list[dict]:
    """Detect highly correlated feature pairs."""
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    n = X.shape[1]
    if n < 2:
        return []

    corr = np.corrcoef(X.T)
    redundant = []

    for i in range(n):
        for j in range(i + 1, n):
            c = abs(corr[i, j])
            if not np.isnan(c) and c > threshold:
                redundant.append({
                    "feature_a": feature_names[i],
                    "feature_b": feature_names[j],
                    "correlation": round(float(c), 4),
                })

    return sorted(redundant, key=lambda x: -x["correlation"])


# --- Feature Generation ---


def generate_interaction_features(
    top_features: list[str],
    max_interactions: int = 10,
) -> list[dict]:
    """Generate candidate interaction features from top consensus features."""
    candidates = []

    for i, fa in enumerate(top_features[:5]):
        for fb in top_features[i + 1:6]:
            if len(candidates) >= max_interactions:
                break
            candidates.append({"name": f"{fa}_x_{fb}", "type": "product", "features": [fa, fb]})
            candidates.append({"name": f"{fa}_div_{fb}", "type": "ratio", "features": [fa, fb]})

    return candidates[:max_interactions]


# --- Full Pipeline ---


def feature_analysis(
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    method: str = "all",
    top_k: int = DEFAULT_TOP_K,
    config_path: str = "config.yaml",
) -> dict:
    """Run feature intelligence analysis."""
    config = load_config(config_path)

    if X is None or y is None:
        return {"error": "Provide X and y arrays for feature analysis",
                "note": "Run with --data train.npz to analyze features"}

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    rankings = {}

    if method in ("all", "importance"):
        rankings["mutual_info"] = mutual_information_ranking(X, y, feature_names)
        rankings["l1"] = l1_ranking(X, y, feature_names)
        rankings["tree"] = tree_importance_ranking(X, y, feature_names)

    if not rankings:
        return {"error": f"Unknown method: {method}"}

    consensus = compute_consensus(rankings, top_k)
    redundant = detect_redundancy(X, feature_names)

    top_consensus_features = [c["feature"] for c in consensus if c["consensus"] > 0][:top_k]
    interactions = generate_interaction_features(top_consensus_features)

    drop_candidates = [c for c in consensus if c["consensus"] == 0]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_features": X.shape[1],
        "top_k": top_k,
        "rankings": {k: v[:top_k] for k, v in rankings.items()},
        "consensus": consensus[:top_k * 2],
        "drop_candidates": drop_candidates,
        "n_drop": len(drop_candidates),
        "redundant_pairs": redundant,
        "interaction_candidates": interactions,
        "recommendation": f"Drop {len(drop_candidates)} features with 0/{len(rankings)} consensus ({len(drop_candidates)/X.shape[1]*100:.0f}% of features)" if drop_candidates else "All features contribute to at least one method",
    }

    return report


# --- Report Formatting ---


def save_feature_report(report: dict, output_dir: str = "experiments/features") -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filepath = out_path / f"features-{date}.yaml"
    clean = json.loads(json.dumps(report, default=str))
    with open(filepath, "w") as f:
        yaml.dump(clean, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_feature_report(report: dict) -> str:
    if "error" in report:
        return f"ERROR: {report['error']}\n{report.get('note', '')}"

    lines = ["# Feature Intelligence", "",
             f"*Generated {report.get('generated_at', 'N/A')[:19]}*",
             f"**{report.get('n_features', 0)} features analyzed, top-{report.get('top_k', 20)}**", ""]

    # Consensus table
    consensus = report.get("consensus", [])
    if consensus:
        methods = set()
        for c in consensus:
            methods.update(c.get("methods", {}).keys())
        method_names = sorted(methods)

        header = "| Feature |" + "|".join(f" {m} Rank " for m in method_names) + "| Consensus |"
        sep = "|---------|" + "|".join("-------" for _ in method_names) + "|-----------|"
        lines.extend(["## Consensus Ranking", "", header, sep])
        for c in consensus[:15]:
            ranks = "|".join(f" {c['methods'].get(m, '—')} " for m in method_names)
            lines.append(f"| {c['feature']} |{ranks}| {c['consensus_str']} |")
        lines.append("")

    # Redundancy
    redundant = report.get("redundant_pairs", [])
    if redundant:
        lines.extend(["## Redundant Pairs", ""])
        for r in redundant[:5]:
            lines.append(f"- **{r['feature_a']}** ↔ **{r['feature_b']}**: r={r['correlation']}")
        lines.append("")

    # Interactions
    interactions = report.get("interaction_candidates", [])
    if interactions:
        lines.extend(["## Candidate Interactions", ""])
        for i in interactions[:5]:
            lines.append(f"- `{i['name']}` ({i['type']}: {' × '.join(i['features'])})")
        lines.append("")

    # Recommendation
    lines.extend(["## Recommendation", "", report.get("recommendation", "")])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated feature selection")
    parser.add_argument("--method", choices=["all", "importance", "selection", "generation"], default="all")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = feature_analysis(method=args.method, top_k=args.top_k, config_path=args.config)

    if "error" not in report:
        filepath = save_feature_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_feature_report(report))


if __name__ == "__main__":
    main()
