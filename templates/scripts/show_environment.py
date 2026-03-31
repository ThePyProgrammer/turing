"""Display the runtime environment from train_metadata.json.

Shows python version, package versions, seeds, hardware, and config hash.
Useful for debugging reproducibility issues or comparing environments
across experiments.

Usage:
    python scripts/show_environment.py                    # Current experiment
    python scripts/show_environment.py --file path/to/metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_METADATA_PATH = "train_metadata.json"


def format_environment(metadata: dict) -> str:
    """Format environment info for display."""
    env = metadata.get("environment", {})
    if not env:
        return "No environment data recorded."

    lines = ["Environment"]
    lines.append("=" * 50)

    # Platform
    lines.append(f"  Python:    {env.get('python_version', '?')}")
    lines.append(f"  Platform:  {env.get('platform', '?')}")
    lines.append(f"  Machine:   {env.get('machine', '?')}")
    lines.append(f"  OS:        {env.get('os', '?')}")

    # GPU
    gpu = env.get("gpu")
    if gpu:
        lines.append(f"  GPU:       {gpu.get('name', '?')} (CUDA {gpu.get('cuda_version', '?')})")
        lines.append(f"  GPUs:      {gpu.get('device_count', '?')}")
    else:
        lines.append("  GPU:       none")

    # Seeds
    seeds = env.get("seeds", {})
    lines.append("")
    lines.append("Seeds")
    lines.append("-" * 50)
    for key, value in sorted(seeds.items()):
        lines.append(f"  {key}: {value}")

    # Packages
    packages = env.get("packages", {})
    if packages:
        lines.append("")
        lines.append("Packages")
        lines.append("-" * 50)
        for pkg, version in sorted(packages.items()):
            lines.append(f"  {pkg}: {version}")

    # Config hash
    config_hash = env.get("config_hash")
    if config_hash:
        lines.append("")
        lines.append(f"Config hash: {config_hash}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show experiment environment")
    parser.add_argument(
        "--file",
        default=DEFAULT_METADATA_PATH,
        help=f"Path to metadata file (default: {DEFAULT_METADATA_PATH})",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"No metadata file at {path}. Run training first.", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        metadata = json.load(f)

    print(format_environment(metadata))


if __name__ == "__main__":
    main()
