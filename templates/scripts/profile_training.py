#!/usr/bin/env python3
"""Computational profiling for ML training runs.

Measures timing breakdown, memory usage, throughput, and identifies
bottlenecks. Maps bottleneck patterns to known fixes.

Usage:
    python scripts/profile_training.py                     # Profile best config
    python scripts/profile_training.py --exp-id exp-042    # Specific experiment
    python scripts/profile_training.py --detailed          # Include per-phase breakdown
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments


# Known bottleneck patterns -> recommendation mappings
BOTTLENECK_RECOMMENDATIONS = {
    "data_loading": [
        "Cache preprocessed data to disk — data loading dominates training time",
        "Use memory-mapped files or HDF5 for large datasets",
        "Check for unnecessary file I/O in the data pipeline",
    ],
    "preprocessing": [
        "Move preprocessing to a one-time step before training",
        "Cache feature transformations (encoders, scalers) to avoid recomputation",
        "Consider using vectorized operations instead of row-by-row processing",
    ],
    "model_training": [
        "Reduce model complexity (fewer estimators, smaller depth)",
        "Try a faster model type (LightGBM is typically faster than XGBoost)",
        "Enable GPU acceleration if available and model supports it",
    ],
    "evaluation": [
        "Reduce validation set size for intermediate checks",
        "Evaluate less frequently (every N iterations instead of every iteration)",
    ],
    "memory": [
        "Process data in batches to reduce peak memory",
        "Use float32 instead of float64 for features",
        "Release intermediate dataframes after use",
    ],
}


def find_experiment(experiments: list[dict], exp_id: str | None, metric: str, lower_is_better: bool) -> dict | None:
    """Find experiment by ID or return best kept."""
    if exp_id:
        for exp in experiments:
            if exp.get("experiment_id") == exp_id:
                return exp
        return None
    best = None
    best_val = float("inf") if lower_is_better else float("-inf")
    for exp in experiments:
        if exp.get("status") != "kept":
            continue
        val = exp.get("metrics", {}).get(metric)
        if val is None:
            continue
        if (lower_is_better and val < best_val) or (not lower_is_better and val > best_val):
            best_val = val
            best = exp
    return best


def get_system_memory_mb() -> float | None:
    """Get current process RSS memory in MB from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return None


def get_peak_memory_mb() -> float | None:
    """Get peak RSS memory in MB from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmPeak:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return None


def check_gpu_available() -> dict | None:
    """Check for GPU availability and return info."""
    # Try torch
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "type": "cuda",
                "device": torch.cuda.get_device_name(0),
                "memory_total_mb": round(torch.cuda.get_device_properties(0).total_mem / 1024**2),
            }
    except ImportError:
        pass
    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            return {
                "type": "cuda",
                "device": parts[0].strip(),
                "memory_total_mb": int(parts[1].strip()) if len(parts) > 1 else None,
            }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def profile_training_run(seed: int = 42, timeout: int = 600) -> dict:
    """Run train.py with profiling instrumentation.

    Wraps the training run and captures timing, memory, and throughput.
    """
    # Start memory tracking
    tracemalloc.start()
    mem_before = get_system_memory_mb()

    start_time = time.perf_counter()

    # Run training
    cmd = ["python", "train.py", "--seed", str(seed)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"Training timed out after {timeout}s"}

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Memory snapshot
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_after = get_system_memory_mb()
    peak_rss = get_peak_memory_mb()

    # Parse metrics from output
    metrics = {}
    timing_data = {}
    in_block = False
    for line in proc.stdout.splitlines():
        line_stripped = line.strip()
        if line_stripped == "---":
            if in_block:
                break
            in_block = True
            continue
        if in_block and ":" in line_stripped:
            key, value = line_stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                metrics[key] = float(value)
            except ValueError:
                metrics[key] = value

    # Extract timing if train.py reports it
    train_seconds = metrics.get("train_seconds")
    if isinstance(train_seconds, str):
        try:
            train_seconds = float(train_seconds)
        except ValueError:
            train_seconds = None

    # Build profile
    profile = {
        "total_time_sec": round(total_time, 2),
        "train_time_sec": round(float(train_seconds), 2) if train_seconds else round(total_time, 2),
        "overhead_sec": round(total_time - float(train_seconds), 2) if train_seconds else 0,
        "memory": {
            "peak_rss_mb": round(peak_rss, 1) if peak_rss else None,
            "python_peak_mb": round(peak / 1024**2, 1),
            "rss_before_mb": round(mem_before, 1) if mem_before else None,
            "rss_after_mb": round(mem_after, 1) if mem_after else None,
        },
        "metrics": metrics,
        "returncode": proc.returncode,
    }

    # Check GPU
    gpu_info = check_gpu_available()
    if gpu_info:
        profile["gpu"] = gpu_info
        # Try to get GPU memory usage
        try:
            import torch
            if torch.cuda.is_available():
                profile["memory"]["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        except ImportError:
            pass

    return profile


def estimate_timing_breakdown(profile: dict) -> dict:
    """Estimate timing breakdown from available data.

    Since we can't instrument inside train.py without modifying it,
    we estimate based on total time and known patterns.
    """
    total = profile.get("train_time_sec", profile.get("total_time_sec", 0))
    if total <= 0:
        return {}

    train_secs = profile.get("metrics", {}).get("train_seconds")
    if isinstance(train_secs, str):
        try:
            train_secs = float(train_secs)
        except ValueError:
            train_secs = None

    overhead = profile.get("overhead_sec", 0)

    breakdown = {
        "total_sec": round(total, 2),
        "overhead_sec": round(overhead, 2),
        "overhead_pct": round(overhead / total * 100, 1) if total > 0 else 0,
    }

    if train_secs:
        breakdown["training_sec"] = round(train_secs, 2)
        breakdown["training_pct"] = round(train_secs / total * 100, 1)

    return breakdown


def detect_bottleneck(profile: dict, breakdown: dict) -> dict:
    """Identify the primary bottleneck from profiling data."""
    bottlenecks = []

    total = profile.get("total_time_sec", 0)
    overhead = profile.get("overhead_sec", 0)

    # Check if overhead (data loading + setup) is dominant
    if total > 0 and overhead > 0:
        overhead_pct = overhead / total * 100
        if overhead_pct > 50:
            bottlenecks.append({
                "type": "data_loading",
                "severity": "high",
                "description": f"Non-training overhead is {overhead_pct:.0f}% of total time ({overhead:.1f}s of {total:.1f}s)",
                "pct_of_total": round(overhead_pct, 1),
            })

    # Check memory pressure
    peak_rss = profile.get("memory", {}).get("peak_rss_mb")
    if peak_rss and peak_rss > 4096:
        bottlenecks.append({
            "type": "memory",
            "severity": "medium" if peak_rss < 8192 else "high",
            "description": f"Peak memory usage is {peak_rss:.0f} MB",
            "pct_of_total": 0,
        })

    # Check GPU utilization
    gpu = profile.get("gpu")
    peak_gpu = profile.get("memory", {}).get("peak_gpu_mb")
    if gpu and peak_gpu:
        gpu_total = gpu.get("memory_total_mb", 0)
        if gpu_total > 0:
            util_pct = peak_gpu / gpu_total * 100
            if util_pct < 30:
                bottlenecks.append({
                    "type": "gpu_underutilized",
                    "severity": "medium",
                    "description": f"GPU memory utilization is only {util_pct:.0f}% — consider larger batch size",
                    "pct_of_total": 0,
                })

    # Check training time relative to expectations
    train_secs = profile.get("metrics", {}).get("train_seconds")
    if isinstance(train_secs, (int, float)) and train_secs > 300:
        bottlenecks.append({
            "type": "model_training",
            "severity": "medium",
            "description": f"Training takes {train_secs:.0f}s — consider model simplification",
            "pct_of_total": 0,
        })

    if not bottlenecks:
        return {
            "type": "none_detected",
            "severity": "low",
            "description": "No significant bottleneck detected",
            "pct_of_total": 0,
        }

    # Return most severe
    severity_order = {"high": 3, "medium": 2, "low": 1}
    bottlenecks.sort(key=lambda b: -severity_order.get(b["severity"], 0))
    return bottlenecks[0]


def generate_recommendations(bottleneck: dict, profile: dict) -> list[str]:
    """Generate actionable recommendations based on detected bottleneck."""
    bt = bottleneck.get("type", "none_detected")
    recs = BOTTLENECK_RECOMMENDATIONS.get(bt, [])

    # Add context-specific recommendations
    extra = []
    peak_rss = profile.get("memory", {}).get("peak_rss_mb")
    if peak_rss and peak_rss > 2048 and bt != "memory":
        extra.append(f"Memory usage is {peak_rss:.0f} MB — monitor for OOM risk on smaller machines")

    gpu = profile.get("gpu")
    if not gpu:
        extra.append("No GPU detected — GPU acceleration could significantly speed up training")

    total = profile.get("total_time_sec", 0)
    if total < 10:
        extra.append("Training is very fast — profiling overhead may distort results")

    return list(recs) + extra


def compute_throughput(profile: dict) -> dict:
    """Compute throughput metrics from profile data."""
    total_time = profile.get("train_time_sec", profile.get("total_time_sec", 0))
    metrics = profile.get("metrics", {})

    n_samples = metrics.get("n_samples") or metrics.get("n_train_samples")
    if isinstance(n_samples, str):
        try:
            n_samples = int(n_samples)
        except ValueError:
            n_samples = None

    throughput = {}
    if n_samples and total_time > 0:
        throughput["samples_per_sec"] = round(n_samples / total_time, 1)

    return throughput


def save_profile(profile_data: dict, output_dir: str = "experiments/profiles") -> Path:
    """Save profile results to YAML file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_id = profile_data.get("experiment_id", "unknown")
    filepath = out_path / f"{exp_id}-profile.yaml"
    with open(filepath, "w") as f:
        yaml.dump(profile_data, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_profile_report(profile_data: dict) -> str:
    """Format profile results as human-readable markdown."""
    if "error" in profile_data:
        return f"ERROR: {profile_data['error']}"

    exp_id = profile_data.get("experiment_id", "?")
    profile = profile_data.get("profile", {})
    breakdown = profile_data.get("breakdown", {})
    bottleneck = profile_data.get("bottleneck", {})
    recommendations = profile_data.get("recommendations", [])
    throughput = profile_data.get("throughput", {})

    lines = [
        f"# Training Profile: {exp_id}",
        "",
    ]

    # Timing
    lines.extend([
        "## Timing",
        "",
        f"- **Total time:** {profile.get('total_time_sec', 0):.1f}s",
    ])
    if breakdown.get("training_sec"):
        lines.append(f"- **Training:** {breakdown['training_sec']:.1f}s ({breakdown.get('training_pct', 0):.0f}%)")
    if breakdown.get("overhead_sec", 0) > 0:
        lines.append(f"- **Overhead:** {breakdown['overhead_sec']:.1f}s ({breakdown.get('overhead_pct', 0):.0f}%)")

    # Memory
    mem = profile.get("memory", {})
    lines.extend(["", "## Memory", ""])
    if mem.get("peak_rss_mb"):
        lines.append(f"- **Peak RSS:** {mem['peak_rss_mb']:.0f} MB")
    lines.append(f"- **Python peak:** {mem.get('python_peak_mb', 0):.1f} MB")
    if mem.get("peak_gpu_mb"):
        lines.append(f"- **GPU peak:** {mem['peak_gpu_mb']:.0f} MB")

    # GPU
    gpu = profile.get("gpu")
    if gpu:
        lines.extend(["", "## GPU", ""])
        lines.append(f"- **Device:** {gpu.get('device', 'unknown')}")
        if gpu.get("memory_total_mb"):
            lines.append(f"- **Total VRAM:** {gpu['memory_total_mb']} MB")

    # Throughput
    if throughput:
        lines.extend(["", "## Throughput", ""])
        if throughput.get("samples_per_sec"):
            lines.append(f"- **Samples/sec:** {throughput['samples_per_sec']:.1f}")

    # Bottleneck
    lines.extend([
        "",
        "## Bottleneck",
        "",
        f"**{bottleneck.get('type', 'none')}** ({bottleneck.get('severity', 'low')})",
        "",
        bottleneck.get("description", "No bottleneck detected."),
    ])

    # Recommendations
    if recommendations:
        lines.extend(["", "## Recommendations", ""])
        for rec in recommendations:
            lines.append(f"- {rec}")

    return "\n".join(lines)


def run_profile(
    exp_id: str | None = None,
    config_path: str = "config.yaml",
    log_path: str = "experiments/log.jsonl",
    seed: int = 42,
    timeout: int = 600,
) -> dict:
    """Run a complete training profile.

    Args:
        exp_id: Experiment ID (defaults to best).
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        seed: Random seed for the profiling run.
        timeout: Training timeout in seconds.

    Returns:
        Complete profile result dict.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiments = load_experiments(log_path)
    target_exp = find_experiment(experiments, exp_id, primary_metric, lower_is_better)

    if not target_exp:
        return {"error": f"No experiment found{f' with ID {exp_id}' if exp_id else ''}"}

    target_id = target_exp.get("experiment_id", "unknown")

    print(f"Profiling {target_id}...", file=sys.stderr)

    # Run profiled training
    profile = profile_training_run(seed=seed, timeout=timeout)

    if "error" in profile:
        return {"error": profile["error"], "experiment_id": target_id}

    # Analyze results
    breakdown = estimate_timing_breakdown(profile)
    bottleneck = detect_bottleneck(profile, breakdown)
    recommendations = generate_recommendations(bottleneck, profile)
    throughput = compute_throughput(profile)

    result = {
        "experiment_id": target_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "breakdown": breakdown,
        "bottleneck": bottleneck,
        "recommendations": recommendations,
        "throughput": throughput,
    }

    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Computational profiling for ML training")
    parser.add_argument("--exp-id", default=None, help="Experiment ID (defaults to best)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default="experiments/log.jsonl", help="Path to experiment log")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timeout", type=int, default=600, help="Training timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    result = run_profile(
        exp_id=args.exp_id,
        config_path=args.config,
        log_path=args.log,
        seed=args.seed,
        timeout=args.timeout,
    )

    if "error" not in result:
        filepath = save_profile(result)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(format_profile_report(result))


if __name__ == "__main__":
    main()
