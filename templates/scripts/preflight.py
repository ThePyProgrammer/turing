"""Pre-flight resource estimator for ML experiments.

Estimates VRAM, RAM, and disk requirements before running a training script.
Compares against available system resources and issues warnings or blocks
if the experiment is likely to fail due to resource constraints.

Works with any ML project — not Turing-specific. Analyzes:
- Dataset size and shape (from CSV/parquet/splits)
- Model type and architecture (from config or CLI)
- Batch size, precision, and expected memory multipliers
- Available system resources (RAM, VRAM, disk)

Usage:
    python scripts/preflight.py                          # Auto-detect from config.yaml
    python scripts/preflight.py --config config.yaml     # Explicit config
    python scripts/preflight.py --model-type xgboost --dataset data.csv
    python scripts/preflight.py --model-type torch --params 10M --batch-size 32
    python scripts/preflight.py --json                   # Machine-readable output
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# System resource detection
# ---------------------------------------------------------------------------

def get_system_ram_gb() -> float:
    """Get total system RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass
    # Fallback: read /proc/meminfo on Linux
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return round(kb / (1024 ** 2), 1)
    # macOS fallback
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return round(int(result.stdout.strip()) / (1024 ** 3), 1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0.0


def get_available_ram_gb() -> float:
    """Get available (free + cached) RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().available / (1024 ** 3), 1)
    except ImportError:
        pass
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        available = 0
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1])
                break
        if available:
            return round(available / (1024 ** 2), 1)
    return 0.0


def get_gpu_info() -> list[dict]:
    """Detect GPUs and their VRAM."""
    gpus = []

    # Try torch first
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(props.total_mem / (1024 ** 3), 1),
                    "vram_free_gb": round(
                        (props.total_mem - torch.cuda.memory_reserved(i)) / (1024 ** 3), 1,
                    ),
                    "source": "torch",
                })
            return gpus
    except ImportError:
        pass

    # Fallback: nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "vram_gb": round(int(parts[2]) / 1024, 1),
                        "vram_free_gb": round(int(parts[3]) / 1024, 1),
                        "source": "nvidia-smi",
                    })
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return gpus


def get_disk_free_gb(path: str = ".") -> float:
    """Get free disk space at path in GB."""
    usage = shutil.disk_usage(path)
    return round(usage.free / (1024 ** 3), 1)


# ---------------------------------------------------------------------------
# Dataset analysis
# ---------------------------------------------------------------------------

def estimate_dataset_memory(path: str | None = None, splits_dir: str | None = None) -> dict:
    """Estimate memory needed to load a dataset.

    Returns dict with: file_size_mb, estimated_ram_gb, rows, columns, dtype_info.
    """
    result = {
        "file_size_mb": 0.0,
        "estimated_ram_gb": 0.0,
        "rows": 0,
        "columns": 0,
        "source": None,
    }

    # Try splits directory first (Turing convention)
    if splits_dir:
        splits_path = Path(splits_dir)
        if splits_path.exists():
            total_size = sum(f.stat().st_size for f in splits_path.glob("*") if f.is_file())
            result["file_size_mb"] = round(total_size / (1024 ** 2), 1)
            # CSV in RAM is typically 3-5x the file size
            result["estimated_ram_gb"] = round(total_size * 4 / (1024 ** 3), 2)
            result["source"] = str(splits_path)

            # Try to count rows/columns from first file
            for f in sorted(splits_path.glob("*.csv")):
                try:
                    with open(f) as fh:
                        header = fh.readline()
                        result["columns"] = len(header.split(","))
                        lines = sum(1 for _ in fh)
                        result["rows"] += lines
                except OSError:
                    pass
            return result

    # Try single file
    if path:
        p = Path(path)
        if p.exists():
            result["file_size_mb"] = round(p.stat().st_size / (1024 ** 2), 1)
            result["estimated_ram_gb"] = round(p.stat().st_size * 4 / (1024 ** 3), 2)
            result["source"] = str(p)

            if p.suffix == ".csv":
                try:
                    with open(p) as fh:
                        header = fh.readline()
                        result["columns"] = len(header.split(","))
                        result["rows"] = sum(1 for _ in fh)
                except OSError:
                    pass

    return result


# ---------------------------------------------------------------------------
# Model resource estimation
# ---------------------------------------------------------------------------

def parse_param_count(s: str) -> int:
    """Parse parameter count strings like '10M', '1.5B', '350K'."""
    s = s.strip().upper()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(float(s))


def estimate_model_resources(
    model_type: str,
    n_estimators: int = 100,
    max_depth: int = 6,
    n_features: int = 50,
    n_samples: int = 10000,
    param_count: int | None = None,
    batch_size: int = 32,
    precision: str = "fp32",
    sequence_length: int = 512,
) -> dict:
    """Estimate training resource requirements for a model.

    Returns dict with: ram_gb, vram_gb, disk_gb, notes.
    """
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
    bpp = bytes_per_param.get(precision, 4)

    result = {
        "model_type": model_type,
        "ram_gb": 0.0,
        "vram_gb": 0.0,
        "disk_gb": 0.0,
        "requires_gpu": False,
        "notes": [],
    }

    mt = model_type.lower()

    if mt in ("xgboost", "lightgbm", "catboost"):
        # Tree-based: RAM-bound, no GPU required (unless GPU training explicitly used)
        # Rule of thumb: ~1KB per tree node, depth d → 2^d nodes per tree
        nodes_per_tree = min(2 ** max_depth, 2 ** 10)  # Cap at 1024
        tree_memory_gb = (n_estimators * nodes_per_tree * 8 * n_features) / (1024 ** 3)
        data_memory_gb = (n_samples * n_features * 8) / (1024 ** 3)  # float64
        result["ram_gb"] = round(tree_memory_gb + data_memory_gb * 2, 2)  # 2x for train + working
        result["disk_gb"] = round(tree_memory_gb * 2, 2)  # Model artifact
        result["notes"].append(f"{n_estimators} trees, depth {max_depth}")
        if n_samples > 1_000_000:
            result["notes"].append("large dataset — consider subsample parameter")

    elif mt in ("randomforest", "random_forest", "sklearn_rf", "extra_trees"):
        nodes_per_tree = min(2 ** max_depth, 2 ** 16)
        tree_memory_gb = (n_estimators * nodes_per_tree * 8 * n_features) / (1024 ** 3)
        data_memory_gb = (n_samples * n_features * 8) / (1024 ** 3)
        result["ram_gb"] = round(tree_memory_gb + data_memory_gb * 2, 2)
        result["disk_gb"] = round(tree_memory_gb * 3, 2)  # sklearn RF models are large
        result["notes"].append(f"{n_estimators} trees, unlimited depth" if max_depth > 20 else f"{n_estimators} trees, depth {max_depth}")

    elif mt in ("mlp", "neural_network", "nn"):
        if param_count is None:
            # Estimate from features: simple 2-layer MLP
            param_count = n_features * 256 + 256 * 128 + 128 * 1  # ~50K for typical tabular
        model_gb = (param_count * bpp) / (1024 ** 3)
        # Training: model + gradients + optimizer state (Adam = 2x) + activations
        train_multiplier = 4  # model + grad + 2x optimizer state
        activation_gb = (batch_size * param_count * bpp * 0.5) / (1024 ** 3)
        result["vram_gb"] = round(model_gb * train_multiplier + activation_gb, 2)
        result["ram_gb"] = round(model_gb * 2 + (n_samples * n_features * 8) / (1024 ** 3), 2)
        result["disk_gb"] = round(model_gb * 2, 2)
        result["requires_gpu"] = result["vram_gb"] > 1.0
        result["notes"].append(f"{param_count:,} parameters ({precision})")
        if result["vram_gb"] > 0.5:
            result["notes"].append("GPU recommended for reasonable training speed")

    elif mt in ("transformer", "llm", "bert", "gpt"):
        if param_count is None:
            param_count = 100_000_000  # Default 100M
        model_gb = (param_count * bpp) / (1024 ** 3)
        # Transformers: activations scale with batch_size * seq_len * hidden_dim
        hidden_dim = int((param_count / 12) ** 0.5)  # Rough estimate
        activation_gb = (batch_size * sequence_length * hidden_dim * bpp * 12) / (1024 ** 3)
        optimizer_gb = model_gb * 2  # Adam
        result["vram_gb"] = round(model_gb + optimizer_gb + activation_gb, 2)
        result["ram_gb"] = round(model_gb + 2, 2)  # Model + data loading overhead
        result["disk_gb"] = round(model_gb * 3, 2)  # Checkpoints
        result["requires_gpu"] = True
        result["notes"].append(f"{param_count:,} parameters ({precision})")
        result["notes"].append(f"batch_size={batch_size}, seq_len={sequence_length}")
        if result["vram_gb"] > 24:
            result["notes"].append("likely needs multi-GPU or gradient checkpointing")
        elif result["vram_gb"] > 12:
            result["notes"].append("needs >=16GB VRAM GPU (A100/A6000/RTX 4090)")
        elif result["vram_gb"] > 6:
            result["notes"].append("needs >=8GB VRAM GPU")

    elif mt in ("linear", "logistic", "ridge", "lasso", "elastic_net"):
        # Linear models: very lightweight
        result["ram_gb"] = round((n_samples * n_features * 8 * 2) / (1024 ** 3), 2)
        result["disk_gb"] = 0.01
        result["notes"].append("lightweight — no resource concerns")

    else:
        result["notes"].append(f"unknown model type '{model_type}' — using conservative estimates")
        result["ram_gb"] = round((n_samples * n_features * 8 * 3) / (1024 ** 3), 2)
        result["vram_gb"] = 0.0
        result["disk_gb"] = 0.5

    return result


# ---------------------------------------------------------------------------
# Preflight check
# ---------------------------------------------------------------------------

def run_preflight(
    model_type: str | None = None,
    config_path: str = "config.yaml",
    dataset_path: str | None = None,
    param_count: str | None = None,
    batch_size: int = 32,
    precision: str = "fp32",
    sequence_length: int = 512,
) -> dict:
    """Run a complete preflight check.

    Returns dict with: system, dataset, model, verdict, warnings.
    """
    # Load config if available
    config = {}
    if HAS_YAML and Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    # Auto-detect from config
    if not model_type:
        model_type = config.get("model", {}).get("type", "xgboost")
    hyperparams = config.get("model", {}).get("hyperparams", {})
    n_estimators = hyperparams.get("n_estimators", 100)
    max_depth = hyperparams.get("max_depth", 6)

    # System resources
    system = {
        "ram_total_gb": get_system_ram_gb(),
        "ram_available_gb": get_available_ram_gb(),
        "disk_free_gb": get_disk_free_gb(),
        "gpus": get_gpu_info(),
    }

    # Dataset analysis
    splits_dir = config.get("data", {}).get("splits_dir")
    data_source = dataset_path or config.get("data", {}).get("source")
    dataset = estimate_dataset_memory(path=data_source, splits_dir=splits_dir)

    # Model estimation
    params = parse_param_count(param_count) if param_count else None
    model = estimate_model_resources(
        model_type=model_type,
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_features=max(dataset.get("columns", 50), 1),
        n_samples=max(dataset.get("rows", 10000), 1),
        param_count=params,
        batch_size=batch_size,
        precision=precision,
        sequence_length=sequence_length,
    )

    # Total requirements
    total_ram = model["ram_gb"] + dataset["estimated_ram_gb"]
    total_vram = model["vram_gb"]
    total_disk = model["disk_gb"] + dataset["file_size_mb"] / 1024

    # Verdict
    warnings = []
    verdict = "PASS"

    if system["ram_available_gb"] > 0 and total_ram > system["ram_available_gb"] * 0.9:
        warnings.append(
            f"RAM: need ~{total_ram:.1f}GB but only {system['ram_available_gb']:.1f}GB available"
        )
        verdict = "WARN"

    if total_ram > system["ram_total_gb"] * 0.8:
        warnings.append(
            f"RAM: need ~{total_ram:.1f}GB, system has {system['ram_total_gb']:.1f}GB total — may cause swapping"
        )
        verdict = "FAIL"

    if model["requires_gpu"]:
        if not system["gpus"]:
            warnings.append(f"VRAM: model needs ~{total_vram:.1f}GB VRAM but no GPU detected")
            verdict = "FAIL"
        else:
            max_vram = max(g["vram_gb"] for g in system["gpus"])
            if total_vram > max_vram * 0.95:
                warnings.append(
                    f"VRAM: need ~{total_vram:.1f}GB but largest GPU has {max_vram:.1f}GB"
                )
                verdict = "FAIL"
            elif total_vram > max_vram * 0.8:
                warnings.append(
                    f"VRAM: need ~{total_vram:.1f}GB, GPU has {max_vram:.1f}GB — tight fit"
                )
                if verdict != "FAIL":
                    verdict = "WARN"

    if total_disk > system["disk_free_gb"] * 0.5:
        warnings.append(
            f"Disk: model + data need ~{total_disk:.1f}GB, only {system['disk_free_gb']:.1f}GB free"
        )
        if verdict != "FAIL":
            verdict = "WARN"

    return {
        "verdict": verdict,
        "warnings": warnings,
        "requirements": {
            "ram_gb": round(total_ram, 2),
            "vram_gb": round(total_vram, 2),
            "disk_gb": round(total_disk, 2),
            "requires_gpu": model["requires_gpu"],
        },
        "system": system,
        "dataset": dataset,
        "model": model,
    }


def format_preflight(result: dict) -> str:
    """Format preflight results for display."""
    v = result["verdict"]
    icon = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}.get(v, "?")

    lines = [
        f"Preflight Check: {icon} {v}",
        "=" * 50,
    ]

    # Requirements
    req = result["requirements"]
    lines.extend([
        "",
        "Requirements",
        "-" * 50,
        f"  RAM:     ~{req['ram_gb']:.1f} GB",
        f"  VRAM:    ~{req['vram_gb']:.1f} GB" + (" (GPU required)" if req["requires_gpu"] else " (no GPU needed)"),
        f"  Disk:    ~{req['disk_gb']:.1f} GB",
    ])

    # System
    sys_info = result["system"]
    lines.extend([
        "",
        "System",
        "-" * 50,
        f"  RAM:     {sys_info['ram_total_gb']:.1f} GB total, {sys_info['ram_available_gb']:.1f} GB available",
        f"  Disk:    {sys_info['disk_free_gb']:.1f} GB free",
    ])
    if sys_info["gpus"]:
        for gpu in sys_info["gpus"]:
            lines.append(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['vram_gb']:.1f} GB VRAM)")
    else:
        lines.append("  GPU:     none detected")

    # Dataset
    ds = result["dataset"]
    if ds["source"]:
        lines.extend([
            "",
            "Dataset",
            "-" * 50,
            f"  Source:  {ds['source']}",
            f"  Size:    {ds['file_size_mb']:.1f} MB on disk",
            f"  In RAM:  ~{ds['estimated_ram_gb']:.2f} GB estimated",
        ])
        if ds["rows"]:
            lines.append(f"  Shape:   {ds['rows']:,} rows x {ds['columns']} columns")

    # Model
    model = result["model"]
    lines.extend([
        "",
        "Model",
        "-" * 50,
        f"  Type:    {model['model_type']}",
        f"  RAM:     ~{model['ram_gb']:.2f} GB",
        f"  VRAM:    ~{model['vram_gb']:.2f} GB",
    ])
    for note in model["notes"]:
        lines.append(f"  Note:    {note}")

    # Warnings
    if result["warnings"]:
        lines.extend(["", "Warnings", "-" * 50])
        for w in result["warnings"]:
            lines.append(f"  {w}")

    # Verdict
    lines.extend(["", "=" * 50])
    if v == "PASS":
        lines.append("  System has sufficient resources. Proceed with training.")
    elif v == "WARN":
        lines.append("  Training may succeed but resources are tight. Monitor memory usage.")
    else:
        lines.append("  Training will likely fail. Address warnings before proceeding.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-flight resource check for ML training",
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--model-type", default=None, help="Model type (xgboost, lightgbm, torch, transformer, ...)")
    parser.add_argument("--dataset", default=None, help="Path to dataset file")
    parser.add_argument("--params", default=None, help="Parameter count (e.g., 10M, 1.5B)")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16", "int8", "int4"])
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length (transformers)")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")

    args = parser.parse_args()

    result = run_preflight(
        model_type=args.model_type,
        config_path=args.config,
        dataset_path=args.dataset,
        param_count=args.params,
        batch_size=args.batch_size,
        precision=args.precision,
        sequence_length=args.seq_len,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_preflight(result))

    if result["verdict"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
