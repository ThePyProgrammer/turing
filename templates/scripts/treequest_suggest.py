#!/usr/bin/env python3
"""Tree-search-guided hypothesis exploration for the autoresearch pipeline.

Uses TreeQuest's AB-MCTS (Adaptive Branching Monte Carlo Tree Search) to
explore the space of experiment hypotheses. Each tree node is a hypothesis
description + structured config. The generation function produces refinements
of a parent hypothesis, and the scoring function uses the critique engine
(novelty × feasibility × impact) as the reward signal.

This is the search-driven complement to suggest_next.py's surrogate model:
instead of fitting a Random Forest over hyperparameter space, we search
the space of *ideas* using MCTS with the critique score as reward.

Requires: pip install "treequest[all]"

Usage:
    python scripts/treequest_suggest.py \\
        --log experiments/log.jsonl \\
        --config config.yaml \\
        --top 5 \\
        --iterations 30 \\
        --strategy abmcts-a
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from scripts.critique_hypothesis import critique_hypothesis
from scripts.turing_io import load_experiments, load_config


# ---------------------------------------------------------------------------
# Node representation
# ---------------------------------------------------------------------------

@dataclass
class HypothesisNode:
    """A node in the hypothesis search tree.

    Each node represents a concrete experiment hypothesis with both a
    human-readable description and optional structured fields (model type,
    hyperparameters, feature changes) that can be passed to the hypothesis
    queue.
    """
    description: str
    model_type: str | None = None
    hyperparameters: dict | None = None
    feature_changes: dict | None = None
    parent_description: str | None = None
    depth: int = 0
    critique_scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for logging and queue integration."""
        return {
            "description": self.description,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
            "feature_changes": self.feature_changes,
            "parent_description": self.parent_description,
            "depth": self.depth,
            "critique_scores": self.critique_scores,
        }

    @staticmethod
    def from_dict(d: dict) -> "HypothesisNode":
        return HypothesisNode(
            description=d["description"],
            model_type=d.get("model_type"),
            hyperparameters=d.get("hyperparameters"),
            feature_changes=d.get("feature_changes"),
            parent_description=d.get("parent_description"),
            depth=d.get("depth", 0),
            critique_scores=d.get("critique_scores", {}),
        )


# ---------------------------------------------------------------------------
# Critique-based scoring
# ---------------------------------------------------------------------------

def score_hypothesis(
    node: HypothesisNode,
    log_path: str = "experiments/log.jsonl",
    config_path: str = "config.yaml",
) -> float:
    """Score a hypothesis node using the critique engine.

    Returns a float in [0, 10] — the weighted combination of
    novelty (30%), feasibility (30%), and expected impact (40%).
    """
    result = critique_hypothesis(
        description=node.description,
        log_path=log_path,
        config_path=config_path,
    )
    node.critique_scores = {
        "overall": result["overall_score"],
        "novelty": result["novelty"]["score"],
        "feasibility": result["feasibility"]["score"],
        "impact": result["impact"]["score"],
        "verdict": result["verdict"],
    }
    return result["overall_score"]


# ---------------------------------------------------------------------------
# Seed hypothesis generation
# ---------------------------------------------------------------------------

def generate_seed_hypotheses(
    config: dict,
    experiments: list[dict],
) -> list[HypothesisNode]:
    """Generate initial seed hypotheses from config and experiment history.

    These form the root nodes of the search tree. Each represents a
    distinct direction worth exploring.
    """
    seeds: list[HypothesisNode] = []
    current_model = config.get("model", {}).get("type", "xgboost")
    metric = config.get("evaluation", {}).get("primary_metric", "accuracy")

    # Seed 1: alternative model families
    model_alternatives = {
        "xgboost": ["LightGBM with dart boosting", "CatBoost with ordered boosting",
                     "Random Forest with extra-trees"],
        "lightgbm": ["XGBoost with hist method", "CatBoost with ordered boosting",
                      "Random Forest with extra-trees"],
        "catboost": ["XGBoost with hist method", "LightGBM with GOSS sampling",
                      "Random Forest with extra-trees"],
        "random_forest": ["XGBoost with hist method", "LightGBM with dart boosting",
                          "CatBoost with ordered boosting"],
    }
    alternatives = model_alternatives.get(current_model.lower(), [
        "XGBoost with hist method", "LightGBM with dart boosting",
    ])
    for alt in alternatives:
        seeds.append(HypothesisNode(
            description=f"Switch to {alt} for {metric} optimization",
            model_type=alt.split(" with ")[0].lower().replace(" ", ""),
        ))

    # Seed 2: regularization exploration
    seeds.append(HypothesisNode(
        description=f"Increase regularization — add L2 penalty and reduce max_depth to combat potential overfitting",
        hyperparameters={"reg_lambda": 1.0, "max_depth": 4},
    ))

    # Seed 3: feature engineering
    seeds.append(HypothesisNode(
        description="Add polynomial interaction features for the top-5 most important numeric columns",
        feature_changes={"add": ["polynomial_interactions"]},
    ))

    # Seed 4: learning rate schedule
    seeds.append(HypothesisNode(
        description=f"Use low learning rate (0.01) with high n_estimators (2000) and early stopping for {metric}",
        hyperparameters={"learning_rate": 0.01, "n_estimators": 2000},
    ))

    # Seed 5: based on experiment history — what's been working?
    kept = [e for e in experiments if e.get("status") == "kept"]
    if kept:
        last_kept = kept[-1]
        last_desc = last_kept.get("description", "")
        if last_desc:
            seeds.append(HypothesisNode(
                description=f"Refine the approach from '{last_desc}' — try a more aggressive variant with doubled learning rate",
                parent_description=last_desc,
            ))

    return seeds


# ---------------------------------------------------------------------------
# Perturbation-based child generation (non-LLM fallback)
# ---------------------------------------------------------------------------

_PERTURBATION_STRATEGIES = [
    "increase learning rate by 2x",
    "decrease learning rate by 2x",
    "double n_estimators",
    "halve max_depth",
    "double max_depth",
    "add L1 regularization (reg_alpha=1.0)",
    "add L2 regularization (reg_lambda=1.0)",
    "increase subsample ratio to 0.9",
    "decrease subsample ratio to 0.6",
    "add column sampling (colsample_bytree=0.7)",
    "switch to dart boosting",
    "switch to GOSS sampling",
    "add polynomial features",
    "add target encoding for categorical columns",
    "remove low-importance features (bottom 20%)",
    "try log-transform on skewed numeric features",
    "add min_child_weight constraint",
    "increase early stopping patience",
]


def generate_children(
    parent: HypothesisNode,
    n_children: int = 3,
    rng_seed: int = 42,
) -> list[HypothesisNode]:
    """Generate child hypotheses by perturbing a parent.

    Uses deterministic perturbation strategies. Each child is a refinement
    or variation of the parent hypothesis.
    """
    import hashlib

    # Deterministic but parent-dependent selection
    parent_hash = int(hashlib.sha256(parent.description.encode()).hexdigest(), 16)
    start_idx = (parent_hash + rng_seed) % len(_PERTURBATION_STRATEGIES)

    children = []
    for i in range(n_children):
        idx = (start_idx + i * 7) % len(_PERTURBATION_STRATEGIES)  # stride of 7 for diversity
        strategy = _PERTURBATION_STRATEGIES[idx]
        child = HypothesisNode(
            description=f"{parent.description}; additionally {strategy}",
            model_type=parent.model_type,
            hyperparameters=dict(parent.hyperparameters) if parent.hyperparameters else None,
            feature_changes=dict(parent.feature_changes) if parent.feature_changes else None,
            parent_description=parent.description,
            depth=parent.depth + 1,
        )
        children.append(child)

    return children


# ---------------------------------------------------------------------------
# TreeQuest integration
# ---------------------------------------------------------------------------

def run_treequest_search(
    seeds: list[HypothesisNode],
    log_path: str = "experiments/log.jsonl",
    config_path: str = "config.yaml",
    iterations: int = 30,
    top_k: int = 5,
    strategy: str = "abmcts-a",
    children_per_node: int = 3,
) -> list[HypothesisNode]:
    """Run TreeQuest MCTS search over the hypothesis space.

    Args:
        seeds: Initial hypothesis nodes (tree roots).
        log_path: Path to experiment log for critique scoring.
        config_path: Path to config for critique scoring.
        iterations: Number of MCTS iterations.
        top_k: Number of best hypotheses to return.
        strategy: TreeQuest algorithm — "abmcts-a" or "abmcts-m".
        children_per_node: Branching factor for child generation.

    Returns:
        Top-K hypothesis nodes ranked by critique score.
    """
    try:
        import treequest
    except ImportError:
        print(
            "TreeQuest not installed. Install with: pip install 'treequest[all]'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Select algorithm
    if strategy == "abmcts-m":
        algo = treequest.ABMCTSM()
    else:
        algo = treequest.ABMCTSA()

    # Track all scored nodes for final ranking
    all_scored: list[HypothesisNode] = []

    def generation_fn(parent_state: HypothesisNode | None) -> tuple[HypothesisNode, float]:
        """TreeQuest generation function.

        Given a parent node (or None for root), generate a child and score it.
        """
        if parent_state is None:
            # Pick a seed
            idx = len(all_scored) % len(seeds)
            node = seeds[idx]
        else:
            children = generate_children(
                parent_state,
                n_children=1,
                rng_seed=len(all_scored),
            )
            node = children[0]

        score = score_hypothesis(node, log_path, config_path)
        all_scored.append(node)
        return node, score

    # Initialize and run the tree search
    algo.init_tree()
    for i in range(iterations):
        try:
            algo.step(generation_fn)
        except Exception as e:
            print(f"Warning: iteration {i} failed: {e}", file=sys.stderr)
            continue

    # Rank all explored nodes by critique score
    all_scored.sort(key=lambda n: n.critique_scores.get("overall", 0), reverse=True)

    # Deduplicate by description similarity
    seen_descriptions: set[str] = set()
    unique_results: list[HypothesisNode] = []
    for node in all_scored:
        # Simple dedup: normalize and check
        normalized = node.description.lower().strip()
        if normalized not in seen_descriptions:
            seen_descriptions.add(normalized)
            unique_results.append(node)
        if len(unique_results) >= top_k:
            break

    return unique_results


# ---------------------------------------------------------------------------
# Fallback: greedy search without TreeQuest
# ---------------------------------------------------------------------------

def run_greedy_search(
    seeds: list[HypothesisNode],
    log_path: str = "experiments/log.jsonl",
    config_path: str = "config.yaml",
    iterations: int = 30,
    top_k: int = 5,
    children_per_node: int = 3,
) -> list[HypothesisNode]:
    """Greedy best-first search fallback when TreeQuest is not installed.

    Expands the highest-scoring node at each step, keeping a priority
    queue of candidates. Less sophisticated than MCTS but requires no
    external dependency.
    """
    import heapq

    # Score seeds
    scored_seeds: list[tuple[float, int, HypothesisNode]] = []
    for i, seed in enumerate(seeds):
        score = score_hypothesis(seed, log_path, config_path)
        # Negate score for min-heap (we want max)
        heapq.heappush(scored_seeds, (-score, i, seed))

    frontier = scored_seeds
    all_explored: list[HypothesisNode] = list(seeds)
    counter = len(seeds)

    for _ in range(iterations):
        if not frontier:
            break

        # Expand best node
        neg_score, _, best = heapq.heappop(frontier)
        children = generate_children(best, n_children=children_per_node, rng_seed=counter)

        for child in children:
            score = score_hypothesis(child, log_path, config_path)
            counter += 1
            heapq.heappush(frontier, (-score, counter, child))
            all_explored.append(child)

    # Rank and deduplicate
    all_explored.sort(key=lambda n: n.critique_scores.get("overall", 0), reverse=True)

    seen: set[str] = set()
    results: list[HypothesisNode] = []
    for node in all_explored:
        normalized = node.description.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            results.append(node)
        if len(results) >= top_k:
            break

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_results(
    results: list[HypothesisNode],
    metric_name: str,
    strategy_used: str,
    total_explored: int,
) -> str:
    """Format search results for terminal display."""
    lines = [
        f"TreeQuest Hypothesis Exploration ({strategy_used})",
        "=" * 60,
        f"Nodes explored: {total_explored}",
        f"Top {len(results)} hypotheses by critique score:",
        "",
    ]

    for i, node in enumerate(results, 1):
        scores = node.critique_scores
        overall = scores.get("overall", 0)
        verdict = scores.get("verdict", "?")
        novelty = scores.get("novelty", 0)
        feasibility = scores.get("feasibility", 0)
        impact = scores.get("impact", 0)

        lines.append(f"  {i}. [{verdict.upper()}] (score: {overall}/10)")
        lines.append(f"     {node.description}")
        lines.append(f"     Novelty: {novelty}  Feasibility: {feasibility}  Impact: {impact}")
        if node.depth > 0:
            lines.append(f"     Depth: {node.depth} (refined from parent)")
        lines.append("")

    return "\n".join(lines)


def results_to_json(results: list[HypothesisNode]) -> list[dict]:
    """Serialize results for machine consumption."""
    return [node.to_dict() for node in results]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tree-search-guided hypothesis exploration",
    )
    parser.add_argument("--log", default="experiments/log.jsonl",
                        help="Path to experiment log")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to project config")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of top hypotheses to return")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Number of search iterations")
    parser.add_argument("--strategy", default="abmcts-a",
                        choices=["abmcts-a", "abmcts-m", "greedy"],
                        help="Search strategy (abmcts-a, abmcts-m, or greedy fallback)")
    parser.add_argument("--children", type=int, default=3,
                        help="Children per node expansion")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--seeds-only", action="store_true",
                        help="Only show generated seeds, don't run search")
    args = parser.parse_args()

    config = load_config(args.config)
    experiments = load_experiments(args.log)
    metric = config.get("evaluation", {}).get("primary_metric", "accuracy")

    # Generate seeds
    seeds = generate_seed_hypotheses(config, experiments)

    if args.seeds_only:
        if args.json:
            print(json.dumps([s.to_dict() for s in seeds], indent=2))
        else:
            print(f"Generated {len(seeds)} seed hypotheses:")
            for i, s in enumerate(seeds, 1):
                print(f"  {i}. {s.description}")
        return

    # Run search
    if args.strategy == "greedy":
        results = run_greedy_search(
            seeds, args.log, args.config,
            iterations=args.iterations,
            top_k=args.top,
            children_per_node=args.children,
        )
        strategy_label = "greedy best-first"
    else:
        try:
            import treequest  # noqa: F401
            results = run_treequest_search(
                seeds, args.log, args.config,
                iterations=args.iterations,
                top_k=args.top,
                strategy=args.strategy,
                children_per_node=args.children,
            )
            strategy_label = f"TreeQuest {args.strategy.upper()}"
        except ImportError:
            print("TreeQuest not installed, falling back to greedy search.", file=sys.stderr)
            results = run_greedy_search(
                seeds, args.log, args.config,
                iterations=args.iterations,
                top_k=args.top,
                children_per_node=args.children,
            )
            strategy_label = "greedy best-first (fallback)"

    # Output
    if args.json:
        print(json.dumps(results_to_json(results), indent=2))
    else:
        total = args.iterations + len(seeds)
        print(format_results(results, metric, strategy_label, total))


if __name__ == "__main__":
    main()
