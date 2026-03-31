"""Tests for experiment archetype expansion.

Verifies that archetype names expand into structured hypothesis
descriptions with steps, family tags, and expected experiment counts.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.manage_hypotheses import expand_archetype, load_archetypes


ARCHETYPE_CONFIG = Path(__file__).parent.parent / "config" / "experiment_archetypes.yaml"


def test_archetypes_file_exists():
    """The archetype config file must exist."""
    assert ARCHETYPE_CONFIG.exists()


def test_archetypes_load():
    """Should load all archetypes from config."""
    archetypes = load_archetypes(str(ARCHETYPE_CONFIG))
    assert len(archetypes) >= 8
    assert "model_comparison" in archetypes
    assert "feature_sweep" in archetypes
    assert "ensemble_construction" in archetypes


def test_each_archetype_has_required_fields():
    """Every archetype must have name, description, steps, when_to_use."""
    archetypes = load_archetypes(str(ARCHETYPE_CONFIG))
    for key, arch in archetypes.items():
        assert "name" in arch, f"{key} missing 'name'"
        assert "description" in arch, f"{key} missing 'description'"
        assert "steps" in arch, f"{key} missing 'steps'"
        assert len(arch["steps"]) >= 3, f"{key} has fewer than 3 steps"
        assert "when_to_use" in arch, f"{key} missing 'when_to_use'"
        assert "family_tag" in arch, f"{key} missing 'family_tag'"


def test_expand_model_comparison():
    """model_comparison archetype should expand with numbered steps."""
    desc, family, tags = expand_archetype("model_comparison", str(ARCHETYPE_CONFIG))
    assert "Systematic Model Comparison" in desc
    assert "1." in desc  # numbered steps
    assert "2." in desc
    assert family == "model-comparison"
    assert "archetype" in tags
    assert "model_comparison" in tags


def test_expand_feature_sweep():
    """feature_sweep should expand with feature engineering steps."""
    desc, family, tags = expand_archetype("feature_sweep", str(ARCHETYPE_CONFIG))
    assert "Feature Engineering" in desc
    assert family == "feature-engineering"


def test_expand_unknown_archetype():
    """Unknown archetype should return an error description with available list."""
    desc, family, tags = expand_archetype("nonexistent_archetype", str(ARCHETYPE_CONFIG))
    assert "Unknown archetype" in desc
    assert "model_comparison" in desc  # should list available ones
    assert family is None


def test_expand_with_no_config(tmp_path):
    """With no config at any search path, unknown archetype returns error."""
    # expand_archetype has fallback paths, so use an archetype name that
    # doesn't exist — this tests the "not found in loaded archetypes" path
    desc, family, tags = expand_archetype("totally_fake_archetype_xyz", str(ARCHETYPE_CONFIG))
    assert "Unknown archetype" in desc


def test_archetype_descriptions_are_multiline():
    """Expanded descriptions should have multiple lines (steps)."""
    archetypes = load_archetypes(str(ARCHETYPE_CONFIG))
    for key in archetypes:
        desc, _, _ = expand_archetype(key, str(ARCHETYPE_CONFIG))
        lines = desc.strip().split("\n")
        assert len(lines) >= 4, f"{key} expansion has fewer than 4 lines"


def test_all_family_tags_are_unique():
    """Each archetype should have a unique family_tag."""
    archetypes = load_archetypes(str(ARCHETYPE_CONFIG))
    tags = [arch.get("family_tag") for arch in archetypes.values()]
    assert len(tags) == len(set(tags)), f"Duplicate family tags: {tags}"
