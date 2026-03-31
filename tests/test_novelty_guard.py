"""Tests for the novelty guard (novelty_guard.py).

Phase 6.1: Prevents the agent from wasting iterations on duplicate work.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.novelty_guard import (
    apply_mode_policy,
    check_novelty,
    classify_against_history,
    concept_overlap,
    extract_numbers,
    normalize_text,
    number_overlap,
    similarity_score,
    token_similarity,
)

SAMPLE_ALIASES = {
    "phrase_aliases": {"learning rate": "lr", "batch size": "batch_size"},
    "token_aliases": {"increase": "up", "decrease": "down", "reduce": "down"},
    "stopwords": ["a", "an", "the", "to", "and", "with", "try", "use"],
    "concept_patterns": {
        "lr": ["lr", "learning_rate"],
        "architecture": ["depth", "width", "layers"],
        "regularization": ["dropout", "weight_decay"],
    },
    "mode_policies": {
        "explore": {"novel": "allow", "duplicate_run": "block", "repeat_failure": "block"},
        "exploit": {"novel": "caution", "incremental_followup": "allow", "repeat_failure": "block", "duplicate_run": "block"},
        "replicate": {"novel": "block", "duplicate_run": "allow"},
    },
}


# --- normalize_text ---

def test_normalize_lowercases():
    result = normalize_text("INCREASE Learning Rate", SAMPLE_ALIASES)
    assert "up" in result  # "increase" → "up"
    assert "lr" in result   # "learning rate" → "lr"


def test_normalize_removes_stopwords():
    result = normalize_text("try to use the model with a learning rate", SAMPLE_ALIASES)
    assert "try" not in result.split()
    assert "the" not in result.split()


def test_normalize_applies_phrase_aliases():
    result = normalize_text("change the learning rate to 0.01", SAMPLE_ALIASES)
    assert "lr" in result


def test_normalize_applies_token_aliases():
    result = normalize_text("reduce max_depth", SAMPLE_ALIASES)
    assert "down" in result


# --- similarity functions ---

def test_token_similarity_identical():
    assert token_similarity("lr up depth", "lr up depth") == 1.0


def test_token_similarity_disjoint():
    assert token_similarity("lr up", "dropout down") == 0.0


def test_token_similarity_partial():
    score = token_similarity("lr up depth", "lr down width")
    assert 0 < score < 1


def test_number_overlap_matching():
    assert number_overlap("set to 0.01 and 100", "use 0.01 for 100 trees") == 1.0


def test_number_overlap_none():
    assert number_overlap("increase depth", "increase depth") == 0.0


def test_concept_overlap_same_domain():
    score = concept_overlap("lr up", "lr down", SAMPLE_ALIASES)
    assert score == 1.0  # both are "lr" concept


def test_concept_overlap_different_domain():
    score = concept_overlap("lr up", "dropout down", SAMPLE_ALIASES)
    assert score < 1.0


# --- classify_against_history ---

def _make_hist(exp_id: str, desc: str, status: str = "kept") -> dict:
    return {"experiment_id": exp_id, "description": desc, "status": status}


def test_classify_novel_empty_history():
    cls, score, match = classify_against_history("something new", "something new", [], SAMPLE_ALIASES)
    assert cls == "novel"
    assert score == 0.0


def test_classify_duplicate():
    history = [_make_hist("exp-001", "increase learning rate to 0.01")]
    proposed_norm = normalize_text("increase learning rate to 0.01", SAMPLE_ALIASES)
    cls, score, _ = classify_against_history(proposed_norm, "increase learning rate to 0.01", history, SAMPLE_ALIASES)
    assert cls == "duplicate_run"
    assert score >= 0.9


def test_classify_repeat_failure():
    history = [_make_hist("exp-001", "increase max_depth to 8", status="discarded")]
    proposed_norm = normalize_text("increase max_depth to 8", SAMPLE_ALIASES)
    cls, score, _ = classify_against_history(proposed_norm, "increase max_depth to 8", history, SAMPLE_ALIASES, threshold=0.5)
    assert cls in ("duplicate_run", "repeat_failure")


def test_classify_known_success():
    history = [_make_hist("exp-001", "try lightgbm with dart boosting", status="kept")]
    proposed_norm = normalize_text("try lightgbm with dart boosting", SAMPLE_ALIASES)
    cls, score, _ = classify_against_history(proposed_norm, "try lightgbm with dart boosting", history, SAMPLE_ALIASES)
    # Exact same text scores high — classified as duplicate, known_success, or incremental_followup (all are "matched to kept")
    assert cls in ("duplicate_run", "known_success", "incremental_followup")
    assert score >= 0.5


def test_classify_novel_different():
    history = [_make_hist("exp-001", "increase learning rate")]
    proposed_norm = normalize_text("add polynomial features and try random forest", SAMPLE_ALIASES)
    cls, score, _ = classify_against_history(proposed_norm, "add polynomial features and try random forest", history, SAMPLE_ALIASES)
    assert cls == "novel"


# --- apply_mode_policy ---

def test_explore_allows_novel():
    decision, _ = apply_mode_policy("novel", "explore", SAMPLE_ALIASES)
    assert decision == "allow"


def test_explore_blocks_duplicate():
    decision, _ = apply_mode_policy("duplicate_run", "explore", SAMPLE_ALIASES)
    assert decision == "block"


def test_exploit_allows_followup():
    decision, _ = apply_mode_policy("incremental_followup", "exploit", SAMPLE_ALIASES)
    assert decision == "allow"


def test_exploit_blocks_failure():
    decision, _ = apply_mode_policy("repeat_failure", "exploit", SAMPLE_ALIASES)
    assert decision == "block"


def test_replicate_allows_duplicate():
    decision, _ = apply_mode_policy("duplicate_run", "replicate", SAMPLE_ALIASES)
    assert decision == "allow"


def test_replicate_blocks_novel():
    decision, _ = apply_mode_policy("novel", "replicate", SAMPLE_ALIASES)
    assert decision == "block"


# --- check_novelty (integration) ---

def test_check_novelty_empty_log(tmp_path: Path):
    result = check_novelty("try something new", str(tmp_path / "empty.jsonl"))
    assert result["classification"] == "novel"
    assert result["decision"] == "allow"


def test_check_novelty_with_history(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    entries = [
        {"experiment_id": "exp-001", "description": "increase max_depth to 8", "status": "discarded"},
    ]
    with open(log, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    result = check_novelty("increase max_depth to 8", str(log), mode="exploit")
    assert result["decision"] == "block"  # repeat_failure or duplicate_run in exploit mode


def test_check_novelty_returns_all_fields(tmp_path: Path):
    result = check_novelty("test", str(tmp_path / "empty.jsonl"))
    assert "classification" in result
    assert "score" in result
    assert "decision" in result
    assert "reason" in result
    assert "top_match" in result
