"""Shared provenance capture for the MLST 2026-09-10 experiment programme.

Every scientific runner in this package records, next to its results, the
information needed to re-run it: the EXP commit, whether the tree was dirty,
the interpreter and library versions, the resolved input paths with their
SHA-256 digests, the exact argv, the seeds and the wall-clock/resource cost.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: str | os.PathLike[str], _chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed so that multi-GB inputs are safe."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(_chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, timeout=60, check=False,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - provenance must never abort a run
        return ""


def git_provenance() -> dict:
    dirty = _git("status", "--porcelain")
    return {
        "repo_root": str(REPO_ROOT),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--always", "--dirty"),
        "tree_dirty": bool(dirty),
        "dirty_paths": dirty.splitlines()[:200],
    }


def environment() -> dict:
    mods = {}
    for name in ("numpy", "scipy", "pandas", "h5py", "yaml", "matplotlib"):
        try:
            mod = __import__(name)
            mods[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            mods[name] = "absent"
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "modules": mods,
    }


class RunRecorder:
    """Collects provenance for one runner invocation."""

    def __init__(self, run_id: str, protocol_refs: dict):
        self.run_id = run_id
        self.protocol_refs = protocol_refs
        self.inputs: list[dict] = []
        self.seeds: dict = {}
        self.notes: list[str] = []
        self._t0 = time.time()
        self._c0 = time.process_time()
        self._r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def add_input(self, role: str, path: str | os.PathLike[str]) -> Path:
        p = Path(path)
        rec = {"role": role, "path": str(p), "exists": p.exists()}
        if p.exists() and p.is_file():
            rec["bytes"] = p.stat().st_size
            rec["sha256"] = sha256_file(p)
        self.inputs.append(rec)
        return p

    def add_seed(self, name: str, value) -> None:
        self.seeds[name] = value

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def finalize(self, extra: dict | None = None) -> dict:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        rec = {
            "run_id": self.run_id,
            "protocol_refs": self.protocol_refs,
            "argv": sys.argv,
            "git": git_provenance(),
            "environment": environment(),
            "inputs": self.inputs,
            "seeds": self.seeds,
            "notes": self.notes,
            "resources": {
                "wall_seconds": round(time.time() - self._t0, 3),
                "cpu_seconds": round(time.process_time() - self._c0, 3),
                "user_seconds": round(ru.ru_utime, 3),
                "system_seconds": round(ru.ru_stime, 3),
                "max_rss_mib": round(ru.ru_maxrss / 1024.0, 1),
            },
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if extra:
            rec.update(extra)
        return rec


def write_json(path: str | os.PathLike[str], payload: dict) -> str:
    """Write JSON deterministically and return the SHA-256 of the bytes.

    The write goes to a temporary file in the same directory and is moved into
    place with ``os.replace``, so a concurrent reader -- the fit-budget scan that
    reads every stage output under a run tree, for one -- sees either no file or
    the whole file, never a half-written one.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, default=str).encode()
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    return sha256_bytes(data)
