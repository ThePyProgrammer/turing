#!/usr/bin/env python3
"""Format-specific model export handlers.

Each handler knows how to export a specific model type to a production
format. Returns the export path and metadata.

Supported formats:
  - joblib: scikit-learn, XGBoost, LightGBM (default)
  - xgboost_json: XGBoost native JSON format
  - lightgbm_text: LightGBM native text format
  - onnx: ONNX via framework-specific converters
  - torchscript: PyTorch JIT trace
  - tflite: TensorFlow Lite
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


# Registry of model types -> supported export formats
FORMAT_REGISTRY = {
    "xgboost": ["joblib", "xgboost_json", "onnx"],
    "lightgbm": ["joblib", "lightgbm_text", "onnx"],
    "random_forest": ["joblib", "onnx"],
    "gradient_boosting": ["joblib", "onnx"],
    "logistic_regression": ["joblib", "onnx"],
    "svm": ["joblib", "onnx"],
    "mlp": ["joblib", "onnx"],
    "pytorch": ["torchscript", "onnx"],
    "tensorflow": ["tflite", "onnx"],
    "keras": ["tflite", "onnx"],
    "catboost": ["joblib", "onnx"],
}

# Default format for each model type
DEFAULT_FORMAT = {
    "xgboost": "joblib",
    "lightgbm": "joblib",
    "random_forest": "joblib",
    "gradient_boosting": "joblib",
    "logistic_regression": "joblib",
    "svm": "joblib",
    "mlp": "joblib",
    "pytorch": "torchscript",
    "tensorflow": "tflite",
    "keras": "tflite",
    "catboost": "joblib",
}

# File extensions for each format
FORMAT_EXTENSIONS = {
    "joblib": ".joblib",
    "xgboost_json": ".json",
    "lightgbm_text": ".txt",
    "onnx": ".onnx",
    "torchscript": ".pt",
    "tflite": ".tflite",
}

# Dependencies required for each format
FORMAT_DEPENDENCIES = {
    "joblib": ["joblib"],
    "xgboost_json": ["xgboost>=1.7"],
    "lightgbm_text": ["lightgbm>=3.0"],
    "onnx": ["onnx", "onnxruntime"],
    "torchscript": ["torch>=1.9"],
    "tflite": ["tensorflow>=2.0"],
}


def get_supported_formats(model_type: str) -> list[str]:
    """Get supported export formats for a model type."""
    return FORMAT_REGISTRY.get(model_type, ["joblib"])


def get_default_format(model_type: str) -> str:
    """Get the default export format for a model type."""
    return DEFAULT_FORMAT.get(model_type, "joblib")


def detect_model_type(config: dict) -> str:
    """Detect model type from experiment config."""
    model_type = config.get("model", {}).get("type", "")
    if not model_type:
        model_type = config.get("model_type", "unknown")
    return model_type.lower().replace("-", "_").replace(" ", "_")


def export_joblib(
    model_path: str,
    output_dir: str,
    model_name: str,
) -> dict:
    """Export model as joblib bundle (copy if already joblib, else convert).

    Returns dict with path, size_bytes, format, and dependencies.
    """
    src = Path(model_path)
    if not src.exists():
        return {"error": f"Model file not found: {model_path}"}

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dst = out_path / f"{model_name}.joblib"
    shutil.copy2(str(src), str(dst))

    return {
        "path": str(dst),
        "format": "joblib",
        "size_bytes": dst.stat().st_size,
        "size_mb": round(dst.stat().st_size / 1024**2, 2),
        "dependencies": FORMAT_DEPENDENCIES["joblib"],
    }


def export_xgboost_json(
    model_path: str,
    output_dir: str,
    model_name: str,
) -> dict:
    """Export XGBoost model to native JSON format."""
    try:
        import joblib
        import xgboost as xgb
    except ImportError as e:
        return {"error": f"Missing dependency: {e}"}

    src = Path(model_path)
    if not src.exists():
        return {"error": f"Model file not found: {model_path}"}

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        model = joblib.load(str(src))
        # Handle wrapped models (e.g., in a pipeline)
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
        elif isinstance(model, xgb.Booster):
            booster = model
        else:
            return {"error": "Model is not an XGBoost model or doesn't have get_booster()"}

        dst = out_path / f"{model_name}.json"
        booster.save_model(str(dst))

        return {
            "path": str(dst),
            "format": "xgboost_json",
            "size_bytes": dst.stat().st_size,
            "size_mb": round(dst.stat().st_size / 1024**2, 2),
            "dependencies": FORMAT_DEPENDENCIES["xgboost_json"],
        }
    except Exception as e:
        return {"error": f"XGBoost JSON export failed: {e}"}


def export_lightgbm_text(
    model_path: str,
    output_dir: str,
    model_name: str,
) -> dict:
    """Export LightGBM model to native text format."""
    try:
        import joblib
        import lightgbm as lgb
    except ImportError as e:
        return {"error": f"Missing dependency: {e}"}

    src = Path(model_path)
    if not src.exists():
        return {"error": f"Model file not found: {model_path}"}

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        model = joblib.load(str(src))
        if hasattr(model, "booster_"):
            booster = model.booster_
        elif isinstance(model, lgb.Booster):
            booster = model
        else:
            return {"error": "Model is not a LightGBM model"}

        dst = out_path / f"{model_name}.txt"
        booster.save_model(str(dst))

        return {
            "path": str(dst),
            "format": "lightgbm_text",
            "size_bytes": dst.stat().st_size,
            "size_mb": round(dst.stat().st_size / 1024**2, 2),
            "dependencies": FORMAT_DEPENDENCIES["lightgbm_text"],
        }
    except Exception as e:
        return {"error": f"LightGBM text export failed: {e}"}


def export_onnx(
    model_path: str,
    output_dir: str,
    model_name: str,
    model_type: str,
) -> dict:
    """Export model to ONNX format."""
    try:
        import joblib
    except ImportError as e:
        return {"error": f"Missing dependency: {e}"}

    src = Path(model_path)
    if not src.exists():
        return {"error": f"Model file not found: {model_path}"}

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dst = out_path / f"{model_name}.onnx"

    try:
        model = joblib.load(str(src))

        # Try sklearn-onnx for scikit-learn compatible models
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            import numpy as np

            # Infer input shape from model if possible
            n_features = getattr(model, "n_features_in_", 10)
            initial_type = [("float_input", FloatTensorType([None, n_features]))]
            onx = convert_sklearn(model, initial_types=initial_type)

            with open(dst, "wb") as f:
                f.write(onx.SerializeToString())

            return {
                "path": str(dst),
                "format": "onnx",
                "size_bytes": dst.stat().st_size,
                "size_mb": round(dst.stat().st_size / 1024**2, 2),
                "dependencies": FORMAT_DEPENDENCIES["onnx"] + ["skl2onnx"],
            }
        except ImportError:
            return {"error": "ONNX export requires skl2onnx: pip install skl2onnx"}
        except Exception as e:
            return {"error": f"ONNX conversion failed: {e}"}

    except Exception as e:
        return {"error": f"ONNX export failed: {e}"}


def export_torchscript(
    model_path: str,
    output_dir: str,
    model_name: str,
) -> dict:
    """Export PyTorch model to TorchScript."""
    try:
        import torch
    except ImportError:
        return {"error": "TorchScript export requires PyTorch: pip install torch"}

    src = Path(model_path)
    if not src.exists():
        return {"error": f"Model file not found: {model_path}"}

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dst = out_path / f"{model_name}.pt"

    try:
        model = torch.load(str(src), map_location="cpu")
        if hasattr(model, "eval"):
            model.eval()

        # Try tracing with dummy input
        if hasattr(model, "example_input"):
            dummy = model.example_input
        else:
            # Create a dummy input — user may need to customize
            dummy = torch.randn(1, 10)

        scripted = torch.jit.trace(model, dummy)
        scripted.save(str(dst))

        return {
            "path": str(dst),
            "format": "torchscript",
            "size_bytes": dst.stat().st_size,
            "size_mb": round(dst.stat().st_size / 1024**2, 2),
            "dependencies": FORMAT_DEPENDENCIES["torchscript"],
        }
    except Exception as e:
        return {"error": f"TorchScript export failed: {e}"}


def export_tflite(
    model_path: str,
    output_dir: str,
    model_name: str,
) -> dict:
    """Export TensorFlow/Keras model to TFLite."""
    try:
        import tensorflow as tf
    except ImportError:
        return {"error": "TFLite export requires TensorFlow: pip install tensorflow"}

    src = Path(model_path)
    if not src.exists():
        return {"error": f"Model file not found: {model_path}"}

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dst = out_path / f"{model_name}.tflite"

    try:
        model = tf.keras.models.load_model(str(src))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        with open(dst, "wb") as f:
            f.write(tflite_model)

        return {
            "path": str(dst),
            "format": "tflite",
            "size_bytes": dst.stat().st_size,
            "size_mb": round(dst.stat().st_size / 1024**2, 2),
            "dependencies": FORMAT_DEPENDENCIES["tflite"],
        }
    except Exception as e:
        return {"error": f"TFLite export failed: {e}"}


def export_model(
    model_path: str,
    output_dir: str,
    model_name: str,
    model_type: str,
    export_format: str | None = None,
) -> dict:
    """Export a model to the specified format.

    Auto-selects format if not specified. Dispatches to format-specific handler.

    Args:
        model_path: Path to the original model file.
        output_dir: Directory to write exported model.
        model_name: Base name for the exported file.
        model_type: Model type (e.g., "xgboost", "pytorch").
        export_format: Target format (e.g., "joblib", "onnx"). Auto-detected if None.

    Returns:
        Export result dict with path, format, size, dependencies.
    """
    if not export_format:
        export_format = get_default_format(model_type)

    supported = get_supported_formats(model_type)
    if export_format not in supported:
        return {
            "error": f"Format '{export_format}' not supported for model type '{model_type}'. "
                     f"Supported: {supported}",
        }

    handlers = {
        "joblib": lambda: export_joblib(model_path, output_dir, model_name),
        "xgboost_json": lambda: export_xgboost_json(model_path, output_dir, model_name),
        "lightgbm_text": lambda: export_lightgbm_text(model_path, output_dir, model_name),
        "onnx": lambda: export_onnx(model_path, output_dir, model_name, model_type),
        "torchscript": lambda: export_torchscript(model_path, output_dir, model_name),
        "tflite": lambda: export_tflite(model_path, output_dir, model_name),
    }

    handler = handlers.get(export_format)
    if not handler:
        return {"error": f"No handler for format '{export_format}'"}

    return handler()
