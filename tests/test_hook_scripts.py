from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates"


def _copy_ml_project(tmp_path: Path) -> Path:
    ml_dir = tmp_path / "ml" / "foo"
    shutil.copytree(TEMPLATES_DIR, ml_dir)
    (ml_dir / "experiments").mkdir(parents=True, exist_ok=True)
    (ml_dir / "models").mkdir(parents=True, exist_ok=True)
    return ml_dir


def _write_fake_uv(bin_dir: Path) -> Path:
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$PWD|$*\" >> \"$UV_LOG\"\n"
        "if [[ \"${1:-}\" == \"run\" ]]; then\n"
        "  shift\n"
        "fi\n"
        "if [[ \"${1:-}\" == \"python\" ]]; then\n"
        "  shift\n"
        f"  exec {shlex_quote(sys.executable)} \"$@\"\n"
        "fi\n"
        "exec \"$@\"\n"
    )
    uv.chmod(0o755)
    return uv


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def _hook_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_log = tmp_path / "uv.log"
    _write_fake_uv(bin_dir)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["UV_LOG"] = str(uv_log)
    return env, uv_log


def test_stop_hook_uses_uv_run_and_preserves_convergence_exit_code(tmp_path: Path):
    ml_dir = _copy_ml_project(tmp_path)
    env, uv_log = _hook_env(tmp_path)
    (ml_dir / "experiments" / "log.jsonl").write_text(
        "\n".join([
            json.dumps({"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.9}}),
            json.dumps({"experiment_id": "exp-002", "status": "kept", "metrics": {"accuracy": 0.89}}),
            json.dumps({"experiment_id": "exp-003", "status": "kept", "metrics": {"accuracy": 0.89}}),
            json.dumps({"experiment_id": "exp-004", "status": "kept", "metrics": {"accuracy": 0.88}}),
        ]) + "\n"
    )

    result = subprocess.run(
        ["bash", str(ml_dir / "scripts" / "stop-hook.sh")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    calls = uv_log.read_text()
    assert f"{ml_dir}|run python scripts/check_convergence.py" in calls


def test_post_train_hook_uses_uv_run_with_absolute_fallback_log(tmp_path: Path):
    ml_dir = _copy_ml_project(tmp_path)
    env, uv_log = _hook_env(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "run.log").write_text(
        "training output\n---\naccuracy: 0.91\nmodel_type: xgboost\ntrain_seconds: 1.5\n---\n"
    )

    result = subprocess.run(
        ["bash", str(ml_dir / "scripts" / "post-train-hook.sh")],
        cwd=outside,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Logged exp-001" in result.stdout
    calls = uv_log.read_text()
    assert f"{ml_dir}|run python scripts/parse_metrics.py {outside / 'run.log'} --raw" in calls
    assert f"{ml_dir}|run python scripts/log_experiment.py" in calls
    assert (ml_dir / "experiments" / "log.jsonl").exists()


def test_core_instructions_are_uv_first():
    checked_paths = [
        TEMPLATES_DIR / "README.md",
        TEMPLATES_DIR / "program.md",
        TEMPLATES_DIR / "scripts" / "scaffold.py",
        REPO_ROOT / "commands" / "init.md",
        REPO_ROOT / "commands" / "rules" / "loop-protocol.md",
        REPO_ROOT / "agents" / "ml-researcher.md",
        REPO_ROOT / "agents" / "ml-evaluator.md",
        REPO_ROOT / "bin" / "turing-init.sh",
    ]
    forbidden = [
        "source .venv/bin/activate",
        "python3 -m venv",
        "pip install -r requirements.txt",
    ]

    offenders = []
    for path in checked_paths:
        text = path.read_text()
        for pattern in forbidden:
            if pattern in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains {pattern}")

    assert offenders == []
