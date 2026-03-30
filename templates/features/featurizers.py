"""Feature engineering strategies for the {{PROJECT_NAME}} ML pipeline.

READ-ONLY — INFRASTRUCTURE.

The autoresearch agent does not modify this file directly. Instead, it
modifies how train.py *uses* the featurizers — composing them differently,
selecting different column subsets, or adding preprocessing in train.py.

Provides pluggable featurizers following a scikit-learn-like fit/transform
interface. The CompositeFeaturizer chains multiple featurizers, concatenating
their output columns.

Exports:
  - BaseFeaturizer: Abstract base class.
  - NumericFeaturizer: Passes through numeric columns.
  - CategoricalFeaturizer: One-hot encodes categorical columns.
  - CompositeFeaturizer: Chains multiple featurizers.
  - get_default_featurizer: Returns the standard composite featurizer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseFeaturizer(ABC):
    """Abstract base class for feature extraction."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseFeaturizer":
        """Fit the featurizer to training data.

        Args:
            df: Training DataFrame.

        Returns:
            self
        """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data into feature DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with extracted features (numeric columns only).
        """

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


class NumericFeaturizer(BaseFeaturizer):
    """Passes through numeric columns as features.

    Customize the column list for your dataset.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns
        self._fitted_columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> "NumericFeaturizer":
        if self.columns:
            self._fitted_columns = [c for c in self.columns if c in df.columns]
        else:
            self._fitted_columns = df.select_dtypes(include=["number"]).columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self._fitted_columns].copy()


class CategoricalFeaturizer(BaseFeaturizer):
    """One-hot encodes categorical columns.

    Customize the column list for your dataset.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns
        self._categories: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame) -> "CategoricalFeaturizer":
        cols = self.columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cols:
            if col in df.columns:
                self._categories[col] = sorted(df[col].unique().tolist())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        for col, categories in self._categories.items():
            if col in df.columns:
                for cat in categories:
                    features[f"{col}_{cat}"] = (df[col] == cat).astype(int)
        return features


class CompositeFeaturizer(BaseFeaturizer):
    """Chains multiple featurizers, concatenating their output columns."""

    def __init__(self, featurizers: list[BaseFeaturizer]) -> None:
        self.featurizers = featurizers

    def fit(self, df: pd.DataFrame) -> "CompositeFeaturizer":
        for f in self.featurizers:
            f.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = [f.transform(df) for f in self.featurizers]
        return pd.concat(parts, axis=1)

    def __repr__(self) -> str:
        names = [type(f).__name__ for f in self.featurizers]
        return f"CompositeFeaturizer({names})"


def get_default_featurizer() -> CompositeFeaturizer:
    """Return the standard composite featurizer.

    Customize this function for your dataset. Add or remove featurizers
    to match your feature engineering needs.

    The autoresearch agent calls this from train.py. To experiment with
    different feature sets, the agent modifies how train.py calls this
    function or composes featurizers differently.
    """
    return CompositeFeaturizer([
        NumericFeaturizer(),
        CategoricalFeaturizer(),
    ])
