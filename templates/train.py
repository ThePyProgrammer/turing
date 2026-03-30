# AGENT-EDITABLE: The autoresearch agent modifies this file. See program.md.
# Training code for {{PROJECT_NAME}}.
# Default implementation: XGBoost classifier.
#
# Customize this file for your ML task. The autoresearch agent will
# iteratively modify this file to improve model performance.

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from evaluate import evaluate_model, format_metrics
from features.featurizers import get_default_featurizer
from prepare import load_config, load_splits


def train_model(
    config_path: str = "config.yaml",
    output_dir: str | None = None,
) -> None:
    """Train a model and print metrics.

    Flow:
      1. Load config
      2. Load train/val splits
      3. Apply featurizer to get feature matrices
      4. Train model with hyperparams from config
      5. Predict on val set
      6. Evaluate using evaluate_model()
      7. Print formatted metrics
      8. Save model to models/ directory

    Customize this function for your specific ML task.
    """
    start_time = time.time()

    # 1. Load config
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "{{TARGET_METRIC}}")

    # 2. Load splits
    splits_dir = config["data"]["splits_dir"]
    train_df, val_df, _test_df = load_splits(splits_dir)

    # 3. Apply featurizer
    featurizer = get_default_featurizer()
    target_column = config["data"].get("target_column", "label")

    X_train = featurizer.fit_transform(train_df)
    X_val = featurizer.transform(val_df)

    # Fill NaN values that may arise from feature engineering
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)

    y_train = train_df[target_column].values
    y_val = val_df[target_column].values

    # 4. Train model (hyperparams from config.yaml)
    model_config = config.get("model", {})
    hyperparams = model_config.get("hyperparams", {})
    model = XGBClassifier(
        **hyperparams,
        random_state=config["data"].get("random_state", 42),
    )
    model.fit(X_train, y_train)

    # 5. Predict on val set
    y_pred = model.predict(X_val)

    # 6. Evaluate
    metrics = evaluate_model(y_pred, y_val, config)

    train_seconds = time.time() - start_time

    # Add metadata for parseable output
    metrics["model_type"] = model_config.get("type", "xgboost")
    metrics["train_seconds"] = round(train_seconds, 1)

    # 7. Print formatted metrics
    print(format_metrics(metrics))

    # 8. Save model
    models_dir = output_dir or config["output"]["models_dir"]
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    model_file = models_path / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "featurizer": featurizer,
            "config": config,
        },
        model_file,
    )
    print(f"\nModel saved to {model_file}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train {{PROJECT_NAME}} model")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config override (default: config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Model output directory (default: from config)",
    )
    args = parser.parse_args()
    train_model(config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
