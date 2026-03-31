"""Tests for the feature engineering pipeline (featurizers.py).

Blocker #6: The feature pipeline the agent composes in train.py.
A bug here silently corrupts every experiment.
"""

from __future__ import annotations

import pandas as pd
import pytest

from features.featurizers import (
    BaseFeaturizer,
    CategoricalFeaturizer,
    CompositeFeaturizer,
    NumericFeaturizer,
    get_default_featurizer,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 30, 35, 40],
        "income": [50000, 60000, 70000, 80000],
        "city": ["NYC", "LA", "NYC", "SF"],
        "status": ["active", "inactive", "active", "active"],
    })


# --- NumericFeaturizer ---


def test_numeric_auto_detect(sample_df):
    """Should auto-detect numeric columns."""
    f = NumericFeaturizer()
    result = f.fit_transform(sample_df)
    assert "age" in result.columns
    assert "income" in result.columns
    assert "city" not in result.columns


def test_numeric_explicit_columns(sample_df):
    """Should use only specified columns."""
    f = NumericFeaturizer(columns=["age"])
    result = f.fit_transform(sample_df)
    assert list(result.columns) == ["age"]
    assert "income" not in result.columns


def test_numeric_missing_column(sample_df):
    """Specified columns not in DataFrame should be skipped."""
    f = NumericFeaturizer(columns=["age", "nonexistent"])
    result = f.fit_transform(sample_df)
    assert "age" in result.columns
    assert "nonexistent" not in result.columns


def test_numeric_empty_dataframe():
    """Empty DataFrame should not crash."""
    f = NumericFeaturizer()
    df = pd.DataFrame()
    result = f.fit_transform(df)
    assert len(result) == 0


# --- CategoricalFeaturizer ---


def test_categorical_one_hot(sample_df):
    """Should one-hot encode categorical columns."""
    f = CategoricalFeaturizer(columns=["city"])
    result = f.fit_transform(sample_df)
    assert "city_LA" in result.columns
    assert "city_NYC" in result.columns
    assert "city_SF" in result.columns
    # NYC appears in rows 0 and 2
    assert result.iloc[0]["city_NYC"] == 1
    assert result.iloc[1]["city_NYC"] == 0


def test_categorical_auto_detect(sample_df):
    """Should auto-detect object/category columns."""
    f = CategoricalFeaturizer()
    result = f.fit_transform(sample_df)
    assert any("city" in col for col in result.columns)
    assert any("status" in col for col in result.columns)


def test_categorical_unseen_at_transform():
    """Unseen categories at transform time should produce zeros, not crash."""
    train = pd.DataFrame({"color": ["red", "blue", "red"]})
    test = pd.DataFrame({"color": ["green", "red", "purple"]})

    f = CategoricalFeaturizer(columns=["color"])
    f.fit(train)
    result = f.transform(test)

    # "green" and "purple" were not in training — their columns should be 0
    assert result.iloc[0]["color_red"] == 0  # green row
    assert result.iloc[1]["color_red"] == 1  # red row
    assert result.iloc[0]["color_blue"] == 0


def test_categorical_all_values_int():
    """One-hot encoded values should all be 0 or 1."""
    df = pd.DataFrame({"x": ["a", "b", "c", "a", "b"]})
    f = CategoricalFeaturizer(columns=["x"])
    result = f.fit_transform(df)
    for col in result.columns:
        assert set(result[col].unique()).issubset({0, 1})


# --- CompositeFeaturizer ---


def test_composite_concatenates(sample_df):
    """Should produce columns from all sub-featurizers."""
    f = CompositeFeaturizer([
        NumericFeaturizer(columns=["age"]),
        CategoricalFeaturizer(columns=["city"]),
    ])
    result = f.fit_transform(sample_df)
    assert "age" in result.columns
    assert "city_NYC" in result.columns


def test_composite_preserves_row_count(sample_df):
    """Output should have same number of rows as input."""
    f = get_default_featurizer()
    result = f.fit_transform(sample_df)
    assert len(result) == len(sample_df)


def test_composite_repr():
    """__repr__ should list sub-featurizer names."""
    f = CompositeFeaturizer([NumericFeaturizer(), CategoricalFeaturizer()])
    r = repr(f)
    assert "NumericFeaturizer" in r
    assert "CategoricalFeaturizer" in r


# --- get_default_featurizer ---


def test_default_featurizer_returns_composite():
    """Should return a CompositeFeaturizer."""
    f = get_default_featurizer()
    assert isinstance(f, CompositeFeaturizer)


# --- fit/transform consistency ---


def test_fit_transform_equals_fit_then_transform(sample_df):
    """fit_transform(X) must equal fit(X).transform(X)."""
    f1 = get_default_featurizer()
    f2 = get_default_featurizer()

    result1 = f1.fit_transform(sample_df)
    f2.fit(sample_df)
    result2 = f2.transform(sample_df)

    pd.testing.assert_frame_equal(result1, result2)
