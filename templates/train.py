# AGENT-EDITABLE — HYPOTHESIS SPACE.
#
# This is the ONLY code file the autoresearch agent should modify.
# Training code for {{PROJECT_NAME}}.
# Default implementation: XGBoost classifier.
#
# The agent iteratively modifies this file to test hypotheses about
# model architecture, hyperparameters, and feature usage. The evaluation
# harness (evaluate.py) remains immutable — ensuring all experiments
# are measured by the same yardstick.
#
# See program.md for the full experiment loop protocol.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
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


def capture_environment(seed: int) -> dict:
    """Capture the full runtime environment for reproducibility.

    Records everything needed to reproduce this exact experiment:
    python version, package versions, hardware, OS, and all random seeds.
    """
    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "os": platform.system(),
        "seeds": {
            "random_state": seed,
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "not set"),
        },
        "packages": {},
    }

    # Key package versions
    for pkg_name in ["xgboost", "lightgbm", "sklearn", "numpy", "pandas", "joblib", "scipy"]:
        try:
            mod = __import__(pkg_name)
            env["packages"][pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    # GPU info (if torch is available)
    try:
        import torch
        env["packages"]["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
            }
            env["seeds"]["cuda_deterministic"] = True
    except ImportError:
        pass

    # Config file hash for drift detection
    config_path = Path("config.yaml")
    if config_path.exists():
        env["config_hash"] = hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]

    return env


def pin_all_seeds(seed: int) -> None:
    """Pin all random seeds for full reproducibility.

    Covers: stdlib random, numpy, PYTHONHASHSEED.
    Torch/CUDA seeds are pinned only if torch is available.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def train_model(
    config_path: str = "config.yaml",
    output_dir: str | None = None,
    seed_override: int | None = None,
) -> None:
    """Train a model and print metrics.

    Flow:
      1. Load config and pin seeds
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

    # 1. Load config and pin all seeds
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "{{TARGET_METRIC}}")
    random_state = seed_override if seed_override is not None else config["data"].get("random_state", 42)
    pin_all_seeds(random_state)

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
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # 5. Predict on train and val sets
    y_pred_val = model.predict(X_val)
    y_pred_train = model.predict(X_train)

    # 6. Evaluate on val (primary) and train (gap monitoring)
    metrics = evaluate_model(y_pred_val, y_val, config)
    train_metrics = evaluate_model(y_pred_train, y_train, config)

    # Compute train/val gap for overfitting detection
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "{{TARGET_METRIC}}")
    val_score = metrics.get(primary_metric)
    train_score = train_metrics.get(primary_metric)
    if val_score is not None and train_score is not None:
        metrics["train_" + primary_metric] = train_score
        metrics["overfit_gap"] = round(train_score - val_score, 4)

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

    # 9. Write training metadata for behavioral validation + reproducibility
    metadata = {
        "train_time_sec": round(train_seconds, 1),
        "model_size_bytes": model_file.stat().st_size if model_file.exists() else 0,
        "n_train_samples": len(X_train),
        "n_features": X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train.columns),
        "predictions_unique": int(len(set(y_pred_val.tolist()))),
        "environment": capture_environment(random_state),
    }
    metadata_path = Path("train_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random_state for multi-run statistical comparison",
    )
    args = parser.parse_args()
    train_model(config_path=args.config, output_dir=args.output_dir, seed_override=args.seed)


if __name__ == "__main__":
    main()
