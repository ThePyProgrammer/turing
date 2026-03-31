#!/usr/bin/env python3
"""History-aware novelty guard for the autoresearch pipeline.

Checks a proposed experiment description against prior experiments
to classify it as novel, known_success, incremental_followup,
repeat_failure, or duplicate_run. Applies a mode policy (explore,
exploit, replicate) to determine whether to proceed.

Prevents the agent from wasting iterations re-trying things it has
already tried, especially across /loop sessions where context is lost.

The matcher is intentionally heuristic — rule-based token matching
with configurable alias tables, not embedding search. Fast, inspectable,
and dependency-free.

Usage:
    python scripts/novelty_guard.py check \\
        --description "increase max_depth to 8" \\
        --log experiments/log.jsonl \\
        --mode exploit

    python scripts/novelty_guard.py check \\
        --description "switch to LightGBM" \\
        --log experiments/log.jsonl \\
        --mode explore
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

TOKEN_RE = re.compile(r"[a-z0-9_]+")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

DEFAULT_ALIASES_PATH = "config/novelty_aliases.yaml"
# Fallback if running from a scaffolded project without the config dir
PLUGIN_ALIASES_PATHS = [
    Path(__file__).parent.parent.parent / "config" / "novelty_aliases.yaml",
    Path(__file__).parent.parent / "config" / "novelty_aliases.yaml",
]

NOVELTY_CLASSES = ("duplicate_run", "repeat_failure", "incremental_followup", "known_success", "novel")


def load_aliases(path: str | None = None) -> dict:
    """Load alias configuration from YAML."""
    candidates = [Path(path)] if path else [Path(DEFAULT_ALIASES_PATH)] + PLUGIN_ALIASES_PATHS
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f) or {}
    return {}


def normalize_text(text: str, aliases: dict) -> str:
    """Normalize experiment description for comparison.

    1. Lowercase
    2. Apply phrase aliases (multi-word → token)
    3. Tokenize
    4. Remove stopwords
    5. Apply token aliases (synonym → canonical)
    6. Sort and deduplicate
    """
    text = text.lower().strip()

    # Apply phrase aliases first (before tokenization breaks multi-word phrases)
    for phrase, replacement in aliases.get("phrase_aliases", {}).items():
        text = text.replace(phrase.lower(), replacement.lower())

    # Tokenize
    tokens = TOKEN_RE.findall(text)

    # Remove stopwords
    stopwords = set(aliases.get("stopwords", []))
    tokens = [t for t in tokens if t not in stopwords and (len(t) > 1 or t in ("up", "nn"))]

    # Apply token aliases
    token_map = aliases.get("token_aliases", {})
    tokens = [token_map.get(t, t) for t in tokens]

    return " ".join(sorted(set(tokens)))


def extract_numbers(text: str) -> set[str]:
    """Extract all numbers from text for numeric overlap scoring."""
    return set(NUMBER_RE.findall(text.lower()))


def token_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity between normalized token sets."""
    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def number_overlap(text_a: str, text_b: str) -> float:
    """Fraction of numbers in text_a that also appear in text_b."""
    nums_a = extract_numbers(text_a)
    nums_b = extract_numbers(text_b)
    if not nums_a:
        return 0.0
    return len(nums_a & nums_b) / len(nums_a)


def concept_overlap(text_a: str, text_b: str, aliases: dict) -> float:
    """Fraction of concept categories shared between two texts."""
    patterns = aliases.get("concept_patterns", {})
    if not patterns:
        return 0.0

    def concepts_for(text: str) -> set[str]:
        found = set()
        tokens = set(text.split())
        for concept, keywords in patterns.items():
            if any(kw in tokens or kw in text for kw in keywords):
                found.add(concept)
        return found

    concepts_a = concepts_for(text_a)
    concepts_b = concepts_for(text_b)
    if not concepts_a or not concepts_b:
        return 0.0
    intersection = concepts_a & concepts_b
    union = concepts_a | concepts_b
    return len(intersection) / len(union)


def similarity_score(
    text_a: str,
    text_b: str,
    raw_a: str,
    raw_b: str,
    aliases: dict,
) -> float:
    """Blended similarity score combining token, number, and concept overlap.

    Returns float in [0, 1]. Higher = more similar.
    """
    tok_sim = token_similarity(text_a, text_b)
    num_sim = number_overlap(raw_a, raw_b)
    con_sim = concept_overlap(text_a, text_b, aliases)

    # Weighted blend: tokens matter most, concepts second, numbers third
    return 0.5 * tok_sim + 0.3 * con_sim + 0.2 * num_sim


def classify_against_history(
    proposed_normalized: str,
    proposed_raw: str,
    history: list[dict],
    aliases: dict,
    threshold: float = 0.6,
) -> tuple[str, float, dict | None]:
    """Classify a proposed experiment against history.

    Returns (classification, best_score, best_match_record).

    Classifications:
    - duplicate_run: score >= 0.9 (nearly identical)
    - repeat_failure: score >= threshold and best match was discarded
    - incremental_followup: score >= threshold * 0.8 and best match was kept
    - known_success: score >= threshold and best match was kept
    - novel: score < threshold (nothing similar enough)
    """
    best_score = 0.0
    best_match = None

    for record in history:
        desc = record.get("description", "")
        normalized = normalize_text(desc, aliases)
        score = similarity_score(proposed_normalized, normalized, proposed_raw, desc, aliases)
        if score > best_score:
            best_score = score
            best_match = record

    if best_score >= 0.9:
        return "duplicate_run", best_score, best_match
    if best_score >= threshold:
        status = best_match.get("status", "") if best_match else ""
        if status == "discarded":
            return "repeat_failure", best_score, best_match
        elif status == "kept":
            return "known_success", best_score, best_match
    if best_score >= threshold * 0.8 and best_match:
        status = best_match.get("status", "")
        if status == "kept":
            return "incremental_followup", best_score, best_match

    return "novel", best_score, best_match


def apply_mode_policy(
    classification: str,
    mode: str,
    aliases: dict,
) -> tuple[str, str]:
    """Apply research mode policy to a novelty classification.

    Returns (decision, reason) where decision is "allow", "block", or "caution".
    """
    policies = aliases.get("mode_policies", {})
    mode_policy = policies.get(mode, {})

    if not mode_policy:
        return "allow", f"no policy defined for mode '{mode}'"

    decision = mode_policy.get(classification, "allow")
    reasons = {
        ("explore", "novel"): "novel enough for exploration",
        ("explore", "duplicate_run"): "duplicate work is not exploration",
        ("explore", "repeat_failure"): "repeating a failed idea is not exploratory",
        ("explore", "incremental_followup"): "follow-up work belongs in exploit mode",
        ("explore", "known_success"): "reusing a known success belongs in exploit mode",
        ("exploit", "novel"): "novel work is fine but not targeted exploitation",
        ("exploit", "duplicate_run"): "exact duplicates belong in replicate mode",
        ("exploit", "repeat_failure"): "prior failures are poor exploitation candidates",
        ("exploit", "incremental_followup"): "close to a known success — good exploitation",
        ("exploit", "known_success"): "refinement of proven approach",
        ("replicate", "novel"): "nothing close enough in history to replicate",
        ("replicate", "duplicate_run"): "intentional replication run",
        ("replicate", "known_success"): "replicating a proven result",
    }

    reason = reasons.get((mode, classification), f"{classification} under {mode} mode")
    return decision, reason


def check_novelty(
    description: str,
    log_path: str,
    mode: str = "exploit",
    aliases_path: str | None = None,
    threshold: float = 0.6,
) -> dict:
    """Main entry point: check a proposed experiment against history.

    Returns dict with: classification, score, decision, reason, top_match.
    """
    aliases = load_aliases(aliases_path)
    normalized = normalize_text(description, aliases)

    # Load experiment history
    history = []
    path = Path(log_path)
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    if not history:
        return {
            "classification": "novel",
            "score": 0.0,
            "decision": "allow",
            "reason": "no prior experiments — everything is novel",
            "top_match": None,
        }

    classification, score, top_match = classify_against_history(
        normalized, description, history, aliases, threshold,
    )
    decision, reason = apply_mode_policy(classification, mode, aliases)

    result = {
        "classification": classification,
        "score": round(score, 4),
        "decision": decision,
        "reason": reason,
        "top_match": {
            "experiment_id": top_match.get("experiment_id", "?"),
            "description": top_match.get("description", ""),
            "status": top_match.get("status", ""),
        } if top_match else None,
    }

    return result


def format_result(result: dict) -> str:
    """Format novelty check result for display."""
    lines = [
        f"Novelty: {result['classification']} (score: {result['score']:.2f})",
        f"Decision: {result['decision']} — {result['reason']}",
    ]
    if result.get("top_match"):
        m = result["top_match"]
        lines.append(f"Nearest: {m['experiment_id']} ({m['status']}) — {m['description'][:60]}")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Novelty guard for experiment proposals")
    subparsers = parser.add_subparsers(dest="command")

    check = subparsers.add_parser("check", help="Check a proposed experiment")
    check.add_argument("--description", required=True, help="Experiment description")
    check.add_argument("--log", default="experiments/log.jsonl", help="Experiment log path")
    check.add_argument("--mode", default="exploit", choices=["explore", "exploit", "replicate"])
    check.add_argument("--aliases", default=None, help="Path to novelty_aliases.yaml")
    check.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold")
    check.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.command == "check":
        result = check_novelty(
            description=args.description,
            log_path=args.log,
            mode=args.mode,
            aliases_path=args.aliases,
            threshold=args.threshold,
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(format_result(result))

        if result["decision"] == "block":
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
