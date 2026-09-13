"""Result files: atomic writes, hashing, schema checks and reconciliation.

Per-cell outputs are JSON files ``cells/<cell_id>.json`` under the run root.
Each carries ``cell_id``, ``attempt_uuid``, ``role``, ``seed``, ``params``,
``fits_used``, ``solves_used``, the package-specific ``result`` block and a
``provenance`` block.  Reduction reads every cell file, reconciles the observed
set against the matrix, and refuses (exit 3) on missing cells, unattributed
files, or duplicates unless the caller declares how duplicates are resolved.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


class ReconciliationError(RuntimeError):
    exit_code = 3


class SchemaError(ValueError):
    exit_code = 4


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_atomic(path: Path | str, obj, *, indent: int | None = 1) -> str:
    """Write JSON via a temp file + rename so a reader never sees a partial
    file; returns the SHA-256 of the written bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=indent, sort_keys=True, default=_jsonable).encode("utf-8")
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return hashlib.sha256(data).hexdigest()


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def code_provenance(files: list[Path | str], repo_root: Path | str) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=str(repo_root), text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unavailable"
    return {"code_commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "tree_dirty": bool(git("status", "--porcelain")),
            "source_sha256": {str(Path(f).relative_to(repo_root)) if str(f).startswith(str(repo_root)) else str(f): sha256_file(f) for f in files},
            "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
            "thread_limits": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


# -- schema -----------------------------------------------------------------------
CELL_REQUIRED = {"cell_id": str, "attempt_uuid": str, "role": str, "params": dict, "fits_used": int,
                 "solves_used": int, "result": dict, "provenance": dict, "development_only": bool}


def check_cell_schema(js: dict, result_required: dict | None = None, *, where: str = "cell") -> None:
    for k, t in CELL_REQUIRED.items():
        if k not in js:
            raise SchemaError(f"{where}: missing key {k!r}")
        if not isinstance(js[k], t):
            raise SchemaError(f"{where}: key {k!r} has type {type(js[k]).__name__}, expected {t.__name__}")
    if result_required:
        for k, t in result_required.items():
            if k not in js["result"]:
                raise SchemaError(f"{where}: result lacks {k!r}")
            v = js["result"][k]
            if t is float:
                if not isinstance(v, (int, float)) or isinstance(v, bool) or (isinstance(v, float) and not np.isfinite(v)):
                    raise SchemaError(f"{where}: result[{k!r}] must be a finite number, got {v!r}")
            elif not isinstance(v, t):
                raise SchemaError(f"{where}: result[{k!r}] has type {type(v).__name__}, expected {t.__name__}")


# -- reconciliation ------------------------------------------------------------------
def read_cells(run_root: Path | str) -> dict[str, list[tuple[Path, dict]]]:
    """All cell files keyed by cell_id (a list, so duplicates are visible)."""
    root = Path(run_root) / "cells"
    out: dict[str, list[tuple[Path, dict]]] = {}
    if not root.exists():
        return out
    for p in sorted(root.glob("*.json")):
        js = json.loads(p.read_text())
        out.setdefault(js.get("cell_id", p.stem), []).append((p, js))
    return out


def reconcile(matrix_cells: list[dict], observed: dict[str, list[tuple[Path, dict]]], *, ledger_completed: dict | None = None,
              roles=None) -> dict:
    """Expected-versus-observed by cell id.  Returns a report; raises
    :class:`ReconciliationError` when the run is not reducible as-is."""
    expected = {c["cell_id"]: c for c in matrix_cells if roles is None or c["role"] in set(roles)}
    missing = sorted(k for k in expected if k not in observed)
    extra = sorted(k for k in observed if k not in expected)
    duplicates = sorted(k for k, v in observed.items() if len(v) > 1)
    unattributed = []
    if ledger_completed is not None:
        for k, files in observed.items():
            for p, js in files:
                atts = {a["attempt_uuid"] for a in ledger_completed.get(k, [])}
                if js.get("attempt_uuid") not in atts:
                    unattributed.append(str(p))
    # file-name / content agreement
    misnamed = [str(p) for k, files in observed.items() for p, js in files if p.stem != js.get("cell_id")]
    report = {"expected": len(expected), "observed": len(observed), "missing": missing, "extra": extra,
              "duplicates": duplicates, "unattributed_files": unattributed, "misnamed_files": misnamed,
              "complete": not missing and not extra and not duplicates and not unattributed and not misnamed}
    return report


def require_reconciled(report: dict, *, allow_partial: bool = False) -> None:
    problems = []
    if report["missing"] and not allow_partial:
        problems.append(f"{len(report['missing'])} missing cells (first: {report['missing'][:3]})")
    if report["extra"]:
        problems.append(f"{len(report['extra'])} cell files not in the matrix (first: {report['extra'][:3]})")
    if report["duplicates"]:
        problems.append(f"{len(report['duplicates'])} cells with duplicate outputs (first: {report['duplicates'][:3]})")
    if report["unattributed_files"]:
        problems.append(f"{len(report['unattributed_files'])} outputs without a completed ledger attempt")
    if report["misnamed_files"]:
        problems.append(f"{len(report['misnamed_files'])} outputs whose file name and cell_id disagree")
    if problems:
        raise ReconciliationError("; ".join(problems))


def stable_bootstrap_ci(values, *, seed: int, replicates: int = 2000, alpha: float = 0.05) -> dict:
    """Percentile bootstrap of the mean with a fixed seed (deterministic)."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {"n": 0, "mean": None, "ci": None}
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(x, size=x.size, replace=True)) for _ in range(replicates)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else None,
            "ci": [float(lo), float(hi)], "replicates": replicates, "seed": seed}
