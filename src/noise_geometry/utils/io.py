"""Run metadata and machine-readable output helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import subprocess
from typing import Any


def git_commit_hash(repo_root: str | Path | None = None) -> str | None:
    """Return the current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


@dataclass
class RunRecord:
    """Standard record written by experiment entry points."""

    experiment_id: str
    status: str
    metrics: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
    preprocessing_metadata: dict[str, Any] = field(default_factory=dict)
    model_metadata: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None


def write_run_record(record: RunRecord, output: str | Path) -> Path:
    """Write an experiment record as pretty JSON."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
