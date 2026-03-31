"""Tests for decision packet synthesis (synthesize_decision.py).

Phase 6.2: Post-run verdict synthesis.
"""

from __future__ import annotations

from scripts.synthesize_decision import (
    classify_outcome,
    compute_delta,
    recommend_action,
    synthesize_packet,
)


def _exp(exp_id: str, status: str = "kept", accuracy: float = 0.85, family: str | None = None) -> dict:
    return {
        "experiment_id": exp_id,
        "status": status,
        "metrics": {"accuracy": accuracy},
        "description": f"Test {exp_id}",
        "family": family,
    }


# --- compute_delta ---

def test_delta_improvement():
    delta = compute_delta(0.90, 0.85, lower_is_better=False)
    assert delta > 0

def test_delta_regression():
    delta = compute_delta(0.80, 0.85, lower_is_better=False)
    assert delta < 0

def test_delta_lower_is_better():
    delta = compute_delta(0.10, 0.15, lower_is_better=True)
    assert delta > 0  # lower is improvement

def test_delta_zero_champion():
    assert compute_delta(0.5, 0.0, lower_is_better=False) == 1.0


# --- classify_outcome ---

def test_new_champion():
    exp = _exp("exp-002", "kept", 0.90)
    champ = _exp("exp-001", "kept", 0.85)
    outcome, delta = classify_outcome(exp, champ, "accuracy", False)
    assert outcome == "new_champion"
    assert delta > 0

def test_marginal_improvement():
    exp = _exp("exp-002", "kept", 0.8505)
    champ = _exp("exp-001", "kept", 0.85)
    outcome, delta = classify_outcome(exp, champ, "accuracy", False)
    assert outcome == "marginal_improvement"

def test_lateral():
    exp = _exp("exp-002", "discarded", 0.850)
    champ = _exp("exp-001", "kept", 0.851)
    outcome, _ = classify_outcome(exp, champ, "accuracy", False)
    assert outcome == "lateral"

def test_regression():
    exp = _exp("exp-002", "discarded", 0.80)
    champ = _exp("exp-001", "kept", 0.85)
    outcome, _ = classify_outcome(exp, champ, "accuracy", False)
    assert outcome == "regression"

def test_crash():
    exp = {"experiment_id": "exp-002", "status": "crash", "metrics": {}, "description": "crashed"}
    outcome, _ = classify_outcome(exp, None, "accuracy", False)
    assert outcome == "crash"

def test_first_experiment_no_champion():
    exp = _exp("exp-001", "kept", 0.85)
    outcome, _ = classify_outcome(exp, None, "accuracy", False)
    assert outcome == "new_champion"


# --- recommend_action ---

def test_promote_for_new_champion():
    action, _ = recommend_action("new_champion", _exp("exp-001"))
    assert action == "promote"

def test_followup_for_marginal():
    action, _ = recommend_action("marginal_improvement", _exp("exp-001"))
    assert action == "branch_followup"

def test_abandon_for_regression():
    action, _ = recommend_action("regression", _exp("exp-001"))
    assert action == "abandon"

def test_fix_for_crash():
    action, _ = recommend_action("crash", _exp("exp-001"))
    assert action == "fix_and_retry"


# --- synthesize_packet ---

def test_packet_has_all_fields():
    exp = _exp("exp-002", "kept", 0.90, family="optimizer-sweep")
    champ = _exp("exp-001", "kept", 0.85)
    packet = synthesize_packet(exp, champ, "accuracy", False)
    assert packet["experiment_id"] == "exp-002"
    assert packet["outcome"] == "new_champion"
    assert packet["action"] == "promote"
    assert packet["delta"] is not None
    assert packet["current_metric"] == 0.90
    assert packet["champion_metric"] == 0.85
    assert packet["champion_id"] == "exp-001"
    assert packet["family"] == "optimizer-sweep"

def test_packet_no_champion():
    exp = _exp("exp-001", "kept", 0.85)
    packet = synthesize_packet(exp, None, "accuracy", False)
    assert packet["champion_id"] is None
    assert packet["outcome"] == "new_champion"


# --- auto_queue_followup ---

from scripts.synthesize_decision import auto_queue_followup
from scripts.manage_hypotheses import load_queue


def test_auto_queue_branch_followup(tmp_path):
    """branch_followup should auto-queue a medium-priority agent hypothesis."""
    qpath = str(tmp_path / "hypotheses.yaml")
    packet = {
        "experiment_id": "exp-005",
        "action": "branch_followup",
        "description": "increase depth to 6",
        "family": "architecture",
    }
    hyp_id = auto_queue_followup(packet, qpath)
    assert hyp_id is not None
    assert hyp_id.startswith("hyp-")

    queue = load_queue(qpath)
    assert len(queue) == 1
    assert queue[0]["source"] == "agent"
    assert queue[0]["priority"] == "medium"
    assert "exp-005" in queue[0]["description"]


def test_auto_queue_fix_and_retry(tmp_path):
    """fix_and_retry should auto-queue a high-priority agent hypothesis."""
    qpath = str(tmp_path / "hypotheses.yaml")
    packet = {
        "experiment_id": "exp-007",
        "action": "fix_and_retry",
        "description": "OOM on large batch size",
        "family": None,
    }
    hyp_id = auto_queue_followup(packet, qpath)
    assert hyp_id is not None

    queue = load_queue(qpath)
    assert queue[0]["priority"] == "high"
    assert "exp-007" in queue[0]["description"]


def test_auto_queue_promote_does_nothing(tmp_path):
    """promote action should NOT auto-queue anything."""
    qpath = str(tmp_path / "hypotheses.yaml")
    packet = {"experiment_id": "exp-010", "action": "promote", "description": "new best", "family": None}
    hyp_id = auto_queue_followup(packet, qpath)
    assert hyp_id is None
    assert load_queue(qpath) == []


def test_auto_queue_abandon_does_nothing(tmp_path):
    """abandon action should NOT auto-queue anything."""
    qpath = str(tmp_path / "hypotheses.yaml")
    packet = {"experiment_id": "exp-011", "action": "abandon", "description": "regression", "family": None}
    hyp_id = auto_queue_followup(packet, qpath)
    assert hyp_id is None
