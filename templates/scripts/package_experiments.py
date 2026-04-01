#!/usr/bin/env python3
"""Package experiments into portable archive for sharing.

Collects config, metrics, seed studies, annotations, decision packets
per experiment. Generates manifest.yaml and README.md inside the
package directory. Does NOT create tar.gz -- just organizes files.

Usage:
    python scripts/package_experiments.py
    python scripts/package_experiments.py --experiments exp-042,exp-043
    python scripts/package_experiments.py --include model,data-hash,figures,code
    python scripts/package_experiments.py --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG = "experiments/log.jsonl"
DEFAULT_OUTPUT = "exports/packages"
VALID_INCLUDES = ["model", "data-hash", "figures", "code"]


def _load_yaml_list(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with open(p) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def _load_yaml_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            d = yaml.safe_load(f)
        return d if isinstance(d, dict) else None
    except (yaml.YAMLError, OSError):
        return None


def _file_hash(filepath: str) -> str | None:
    p = Path(filepath)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_experiment_artifacts(exp: dict, includes: list[str]) -> dict:
    """Collect all artifacts for a single experiment."""
    eid = exp.get("experiment_id", "unknown")
    art: dict = {"experiment_id": eid, "status": exp.get("status", "unknown"),
                 "metrics": exp.get("metrics", {}), "config": exp.get("config", {}),
                 "description": exp.get("description", ""), "timestamp": exp.get("timestamp", ""),
                 "family": exp.get("family")}
    # Seed study
    seed = _load_yaml_file(Path(f"experiments/seed_studies/{eid}-seeds.yaml"))
    if seed:
        art["seed_study"] = {"mean": seed.get("mean"), "std": seed.get("std"),
                              "cv_percent": seed.get("cv_percent"),
                              "seed_sensitive": seed.get("seed_sensitive", False)}
    # Decision packet
    dec = _load_yaml_file(Path(f"experiments/decisions/{eid}-decision.yaml"))
    if dec:
        art["decision"] = {"action": dec.get("action"), "reason": dec.get("reason", "")}
    # Ablation
    abl = _load_yaml_file(Path(f"experiments/ablations/{eid}-ablation.yaml"))
    if abl:
        art["ablation"] = {"metric": abl.get("metric"),
                            "n_ablations": len(abl.get("results", []))}
    # Reproduction
    repro = _load_yaml_file(Path(f"experiments/reproductions/{eid}-repro.yaml"))
    if repro:
        art["reproduction"] = {"verdict": repro.get("verdict"), "reason": repro.get("reason", "")}
    # Optional includes
    if "model" in includes:
        for pat in [f"models/{eid}", f"models/{eid}.*", f"checkpoints/{eid}/*"]:
            matches = list(Path(".").glob(pat))
            if matches:
                art["model_path"] = str(matches[0])
                break
    if "data-hash" in includes:
        dp = exp.get("config", {}).get("data", {}).get("path")
        if dp:
            h = _file_hash(dp)
            if h:
                art["data_hash"] = h
    if "figures" in includes:
        fig_dir = Path(f"experiments/figures/{eid}")
        art["figures"] = [str(f) for f in fig_dir.glob("*") if f.is_file()] if fig_dir.exists() else []
    if "code" in includes:
        art["train_py_hash"] = _file_hash("train.py")
        snap = Path(f"experiments/code/{eid}")
        if snap.exists():
            art["code_snapshot_path"] = str(snap)
    return art


def build_manifest(name: str, config: dict, artifacts: list[dict], includes: list[str]) -> dict:
    eval_cfg = config.get("evaluation", {})
    return {
        "package": {"name": name, "created": datetime.now(timezone.utc).isoformat(),
                     "generator": "turing:share", "version": "1.0"},
        "project": {"task": config.get("task_description", ""),
                     "primary_metric": eval_cfg.get("primary_metric", "accuracy"),
                     "lower_is_better": eval_cfg.get("lower_is_better", False)},
        "contents": {"experiments": len(artifacts), "includes": includes,
                      "has_seed_studies": any(a.get("seed_study") for a in artifacts),
                      "has_decisions": any(a.get("decision") for a in artifacts)},
        "experiments": [{"id": a["experiment_id"], "status": a["status"],
                          "family": a.get("family"), "metrics": a["metrics"]} for a in artifacts],
    }


def build_package_readme(config: dict, artifacts: list[dict], manifest: dict) -> str:
    metric = manifest["project"]["primary_metric"]
    d = "lower" if manifest["project"]["lower_is_better"] else "higher"
    L = [f"# Experiment Package: {manifest['package']['name']}", "",
         f"*Packaged {manifest['package']['created'][:19]} UTC*", "",
         "## Project", "", f"- **Task:** {config.get('task_description', 'N/A')}",
         f"- **Primary metric:** `{metric}` ({d} is better)", "", "## Experiments", "",
         f"| ID | Status | Family | {metric} |",
         f"|----|--------|--------|{'---'*max(len(metric)//3,1)}--|"]
    for a in artifacts:
        v = a.get("metrics", {}).get(metric)
        vs = f"{v:.4f}" if isinstance(v, (int, float)) else "---"
        L.append(f"| {a['experiment_id']} | {a['status']} | {a.get('family','---')} | {vs} |")
    seeds = [a for a in artifacts if a.get("seed_study")]
    if seeds:
        L.extend(["", "## Seed Studies", ""])
        for a in seeds:
            s = a["seed_study"]
            tag = "SEED-SENSITIVE" if s["seed_sensitive"] else "stable"
            L.append(f"- `{a['experiment_id']}`: mean={s['mean']:.4f} +/- {s['std']:.4f} [{tag}]")
    decs = [a for a in artifacts if a.get("decision")]
    if decs:
        L.extend(["", "## Decisions", ""])
        for a in decs:
            L.append(f"- `{a['experiment_id']}`: **{a['decision']['action']}** — {a['decision']['reason']}")
    L.extend(["", "## Files", "", "- `manifest.yaml` — Machine-readable manifest",
              "- `README.md` — This file", "- `experiments/` — Per-experiment artifacts",
              "- `config.yaml` — Project config snapshot", "", "---",
              "*Generated by `/turing:share`*"])
    return "\n".join(L)


def write_package(pkg_dir: Path, config: dict, artifacts: list[dict],
                   manifest: dict, readme: str, includes: list[str]) -> None:
    """Write all package files to the directory."""
    pkg_dir.mkdir(parents=True, exist_ok=True)
    with open(pkg_dir / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    (pkg_dir / "README.md").write_text(readme)
    with open(pkg_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    exp_dir = pkg_dir / "experiments"
    exp_dir.mkdir(exist_ok=True)
    for art in artifacts:
        sub = exp_dir / art["experiment_id"]
        sub.mkdir(exist_ok=True)
        with open(sub / "artifact.yaml", "w") as f:
            yaml.dump(art, f, default_flow_style=False, sort_keys=False)
        if "figures" in includes and art.get("figures"):
            fd = sub / "figures"
            fd.mkdir(exist_ok=True)
            for fp in art["figures"]:
                src = Path(fp)
                if src.exists():
                    shutil.copy2(src, fd / src.name)
        if "code" in includes and art.get("code_snapshot_path"):
            cs = Path(art["code_snapshot_path"])
            if cs.exists() and cs.is_dir():
                shutil.copytree(cs, sub / "code", dirs_exist_ok=True)
        if "model" in includes and art.get("model_path"):
            ms = Path(art["model_path"])
            if ms.exists():
                md = sub / "model"
                md.mkdir(exist_ok=True)
                if ms.is_dir():
                    shutil.copytree(ms, md, dirs_exist_ok=True)
                else:
                    shutil.copy2(ms, md / ms.name)


def save_package_report(result: dict, pkg_dir: Path) -> Path:
    rp = pkg_dir / "package-report.yaml"
    with open(rp, "w") as f:
        yaml.dump({"timestamp": result["timestamp"], "package_name": result["package_name"],
                    "package_dir": str(result["package_dir"]),
                    "experiments_packaged": result["experiments_packaged"],
                    "includes": result["includes"]}, f, default_flow_style=False, sort_keys=False)
    return rp


def format_package_report(result: dict) -> str:
    L = ["# Package Summary", "",
         f"- **Package:** {result['package_name']}",
         f"- **Location:** `{result['package_dir']}`",
         f"- **Experiments:** {result['experiments_packaged']}",
         f"- **Includes:** {', '.join(result['includes']) or 'metrics only'}", "", "## Contents", ""]
    for a in result.get("artifacts", []):
        extras = [k for k in ("seed_study", "decision", "ablation", "reproduction") if a.get(k)]
        es = f" [{', '.join(extras)}]" if extras else ""
        L.append(f"- `{a['experiment_id']}` ({a['status']}){es}")
    return "\n".join(L)


def package_experiments(experiment_ids=None, includes=None, config_path="config.yaml",
                         log_path=DEFAULT_LOG, output_dir=DEFAULT_OUTPUT) -> dict:
    """Package experiments into a portable directory."""
    includes = includes or []
    config = load_config(config_path)
    experiments = load_experiments(log_path)
    annotations = _load_yaml_list("experiments/annotations.yaml")
    if experiment_ids:
        selected = [e for e in experiments if e.get("experiment_id") in experiment_ids]
        if not selected:
            return {"error": f"No matching experiments for: {experiment_ids}"}
    else:
        selected = [e for e in experiments if e.get("status") == "kept"]
        if not selected:
            return {"error": "No kept experiments to package."}
    artifacts = [collect_experiment_artifacts(e, includes) for e in selected]
    for art in artifacts:
        eid = art["experiment_id"]
        anns = [a for a in annotations if a.get("experiment_id") == eid]
        if anns:
            art["annotations"] = anns
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    pkg_name = f"package-{len(artifacts)}exp-{ts}"
    pkg_dir = Path(output_dir) / pkg_name
    manifest = build_manifest(pkg_name, config, artifacts, includes)
    readme = build_package_readme(config, artifacts, manifest)
    write_package(pkg_dir, config, artifacts, manifest, readme, includes)
    result = {"timestamp": datetime.now(timezone.utc).isoformat(), "package_name": pkg_name,
              "package_dir": str(pkg_dir), "experiments_packaged": len(artifacts),
              "includes": includes, "artifacts": artifacts}
    save_package_report(result, pkg_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Package experiments into portable archive")
    parser.add_argument("--experiments", default=None, help="Comma-separated experiment IDs")
    parser.add_argument("--include", default=None, help="Extras: model,data-hash,figures,code")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()
    exp_ids = [e.strip() for e in args.experiments.split(",")] if args.experiments else None
    includes = []
    if args.include:
        includes = [i.strip() for i in args.include.split(",")]
        bad = [i for i in includes if i not in VALID_INCLUDES]
        if bad:
            print(f"ERROR: Invalid include(s): {bad}. Valid: {VALID_INCLUDES}", file=sys.stderr)
            sys.exit(1)
    result = package_experiments(exp_ids, includes, args.config, args.log, args.output)
    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(format_package_report(result))
        print(f"\nPackage saved to: {result['package_dir']}", file=sys.stderr)


if __name__ == "__main__":
    main()
