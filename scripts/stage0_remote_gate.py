#!/usr/bin/env python3
"""Run and archive the Stage 0 remote reproducibility gate.

This script intentionally uses only the Python standard library so it can run
before ``uv sync`` has created the project environment.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_COMMANDS = (
    ("01_uv_sync", ("uv", "sync", "--extra", "dev", "--extra", "gwosc")),
    ("02_pytest", ("uv", "run", "pytest", "-q")),
    ("03_run_all_core", ("uv", "run", "python", "scripts/run_all_core.py")),
    ("04_make_tables", ("uv", "run", "python", "scripts/make_tables.py")),
    ("05_make_all_figures", ("uv", "run", "python", "scripts/make_all_figures.py")),
)


def _capture(command: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Capture a diagnostic command without making it gate-critical."""
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": shlex.join(command)}
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "available": True,
        "command": shlex.join(command),
        "returncode": completed.returncode,
        "output": completed.stdout.strip(),
    }


def _git_status() -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout.strip()


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _environment_snapshot() -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_bootstrap": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "paper1_data_root": os.environ.get("PAPER1_DATA_ROOT"),
        "os_release": _read_optional(Path("/etc/os-release")),
        "cpu": _capture(["lscpu"]),
        "memory": _read_optional(Path("/proc/meminfo")),
        "gpu": _capture(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "uv": _capture(["uv", "--version"]),
    }


def _run_logged(name: str, command: tuple[str, ...], output_dir: Path) -> dict[str, Any]:
    log_path = output_dir / f"{name}.log"
    started = datetime.now(timezone.utc)
    started_clock = time.monotonic()
    header = f"$ {shlex.join(command)}\n"
    print(header, end="", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(header)
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        returncode = process.wait()
    finished = datetime.now(timezone.utc)
    return {
        "name": name,
        "command": shlex.join(command),
        "returncode": returncode,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": time.monotonic() - started_clock,
        "log": str(log_path),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the five-command Stage 0 gate and archive environment evidence."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="artifact directory (default: results/stage0/<UTC timestamp>_<commit>)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run diagnostically from a dirty tree; the gate cannot be accepted",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="run later commands after one command fails",
    )
    args = parser.parse_args()

    if shutil.which("git") is None:
        raise SystemExit("git is required")
    if shutil.which("uv") is None:
        raise SystemExit("uv is required; install it before running the Stage 0 gate")

    commit = _git_value("rev-parse", "HEAD")
    branch = _git_value("rev-parse", "--abbrev-ref", "HEAD")
    status_before = _git_status()
    if status_before and not args.allow_dirty:
        details = "\n".join(status_before)
        raise SystemExit(
            "Stage 0 requires a clean checkout. Commit/stash changes, then retry.\n"
            f"{details}"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else REPO_ROOT / "results" / "stage0" / f"{timestamp}_{commit[:12]}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    environment = _environment_snapshot()
    environment["git"] = {
        "commit": commit,
        "branch": branch,
        "status_before": status_before,
    }
    _write_json(output_dir / "environment_before.json", environment)

    command_results: list[dict[str, Any]] = []
    for name, command in ACCEPTANCE_COMMANDS:
        result = _run_logged(name, command, output_dir)
        command_results.append(result)
        if result["returncode"] != 0 and not args.continue_on_failure:
            break

    dependency_snapshot = _capture(["uv", "pip", "freeze"])
    (output_dir / "dependencies.txt").write_text(
        dependency_snapshot.get("output", "") + "\n",
        encoding="utf-8",
    )
    runtime_snapshot = _capture(
        [
            "uv",
            "run",
            "python",
            "-c",
            (
                "import json, platform, sys; "
                "print(json.dumps({'version': sys.version, "
                "'executable': sys.executable, 'platform': platform.platform()}))"
            ),
        ]
    )
    status_after = _git_status()
    all_commands_ran = len(command_results) == len(ACCEPTANCE_COMMANDS)
    all_commands_passed = all_commands_ran and all(
        result["returncode"] == 0 for result in command_results
    )
    accepted = not status_before and not status_after and all_commands_passed
    summary = {
        "stage": "Stage 0 remote reproducibility",
        "accepted": accepted,
        "requirements": {
            "clean_before": not status_before,
            "clean_after": not status_after,
            "all_five_commands_ran": all_commands_ran,
            "all_five_commands_passed": all_commands_passed,
        },
        "git": {
            "commit": commit,
            "branch": branch,
            "status_before": status_before,
            "status_after": status_after,
        },
        "runtime_after_sync": runtime_snapshot,
        "commands": command_results,
        "artifact_directory": str(output_dir),
    }
    _write_json(output_dir / "summary.json", summary)

    print(f"Stage 0 artifacts: {output_dir}")
    print(f"Stage 0 accepted: {accepted}")
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
