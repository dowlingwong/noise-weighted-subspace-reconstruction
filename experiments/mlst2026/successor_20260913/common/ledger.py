"""Execution ledger: one accountable record of every attempt.

Design rules (from the 13 September amendment):

* an attempt is charged when it *starts* (a ``running`` record is appended
  before any compute), so a crash or SIGKILL leaves its charge behind;
* ``failed`` and ``interrupted`` attempts are never refunded; a retry adds a
  new attempt with its own uuid;
* the charged total is the sum over *all* attempts of the fits each planned
  (running/failed/interrupted) or actually used (completed);
* byte-identical outputs from two attempts are two executions; the
  ``attempt_uuid`` is the identity, not the digest;
* the budget check refuses to start an attempt when
  ``charged + planned > cap`` and exits with status 2.

The ledger is an append-only JSONL file.  Appends are serialised with an
advisory ``flock`` on a sidecar lock file, which is adequate for local disks and
*not* assumed adequate for network filesystems (the remote round must re-check
that; the serial fallback is to run one worker per ledger).
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

LEDGER_SCHEMA = "mlst-local-package-ledger/1"
STATES = ("running", "completed", "failed", "interrupted")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class BudgetExceeded(RuntimeError):
    exit_code = 2


class DuplicateCompletion(RuntimeError):
    exit_code = 5


@contextmanager
def _locked(path: Path):
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class ExecutionLedger:
    def __init__(self, path: Path | str, *, cap: int, package: str, protocol_sha256: str):
        self.path = Path(path)
        self.cap = int(cap)
        self.package = package
        self.protocol_sha256 = protocol_sha256

    # -- reading ------------------------------------------------------------------
    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("schema") != LEDGER_SCHEMA:
                raise ValueError(f"{self.path}: unexpected ledger schema {r.get('schema')!r}")
            if r.get("package") != self.package:
                raise ValueError(f"{self.path}: ledger belongs to package {r.get('package')!r}, not {self.package!r}")
            if r.get("protocol_sha256") != self.protocol_sha256:
                raise ValueError(f"{self.path}: ledger protocol hash differs from frozen protocol")
            out.append(r)
        return out

    def attempts(self) -> dict:
        """Latest state per attempt uuid (a running record followed by a
        completed/failed record for the same uuid is one attempt)."""
        by_uuid: dict[str, dict] = {}
        for r in self.records():
            u = r["attempt_uuid"]
            prev = by_uuid.get(u)
            if prev is None:
                by_uuid[u] = dict(r)
            else:
                prev.update({k: v for k, v in r.items() if k != "started_utc"})
        return by_uuid

    def charged(self) -> dict:
        tot = {"fits": 0, "solves": 0, "attempts": 0, "by_state": {s: 0 for s in STATES}}
        for a in self.attempts().values():
            st = a["state"]
            tot["by_state"][st] = tot["by_state"].get(st, 0) + 1
            tot["attempts"] += 1
            if st == "completed":
                tot["fits"] += int(a.get("fits_used", a["fits_planned"]))
                tot["solves"] += int(a.get("solves_used", a.get("solves_planned", 0)))
            else:
                tot["fits"] += max(int(a["fits_planned"]), int(a.get("fits_observed", 0)))
                tot["solves"] += int(a.get("solves_planned", 0))
        return tot

    def completed_by_cell(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for a in self.attempts().values():
            if a["state"] == "completed":
                out.setdefault(a["cell_id"], []).append(a)
        return out

    # -- writing ------------------------------------------------------------------
    def _append(self, rec: dict) -> None:
        rec = {"schema": LEDGER_SCHEMA, "package": self.package, "protocol_sha256": self.protocol_sha256, **rec}
        with _locked(self.path):
            with open(self.path, "a") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def start(self, cell: dict, *, allow_repeat: bool = False) -> str:
        """Reserve and charge an attempt for ``cell``.  Refuses (exit 2) when the
        cap would be exceeded; refuses (exit 5) a repeat of an already
        completed cell unless ``allow_repeat`` (which still charges again)."""
        with _locked(self.path):
            charged = self.charged()
            done = self.completed_by_cell()
            if cell["cell_id"] in done and not allow_repeat:
                raise DuplicateCompletion(f"cell {cell['cell_id']} already has a completed attempt; refusing to re-run "
                                          "without --allow-repeat (a repeat is charged again and flagged)")
            if charged["fits"] + int(cell["n_fits"]) > self.cap:
                raise BudgetExceeded(f"charged {charged['fits']} + planned {cell['n_fits']} exceeds cap {self.cap}")
            u = str(uuid.uuid4())
            rec = {"attempt_uuid": u, "cell_id": cell["cell_id"], "role": cell["role"], "state": "running",
                   "fits_planned": int(cell["n_fits"]), "solves_planned": int(cell.get("n_solves", 0)),
                   "started_utc": utc_now(), "hostname": socket.gethostname(), "pid": os.getpid(),
                   "repeat": bool(cell["cell_id"] in done)}
            rec = {"schema": LEDGER_SCHEMA, "package": self.package, "protocol_sha256": self.protocol_sha256, **rec}
            with open(self.path, "a") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return u

    def complete(self, attempt_uuid: str, cell: dict, *, fits_used: int, solves_used: int, output_path: Path, output_sha256: str, wall_s: float) -> None:
        self._append({"attempt_uuid": attempt_uuid, "cell_id": cell["cell_id"], "role": cell["role"], "state": "completed",
                      "fits_planned": int(cell["n_fits"]), "fits_used": int(fits_used), "solves_used": int(solves_used),
                      "finished_utc": utc_now(), "output": str(output_path), "output_sha256": output_sha256, "wall_s": float(wall_s)})

    def fail(self, attempt_uuid: str, cell: dict, *, reason: str, wall_s: float, fits_observed: int = 0) -> None:
        self._append({"attempt_uuid": attempt_uuid, "cell_id": cell["cell_id"], "role": cell["role"], "state": "failed",
                      "fits_planned": int(cell["n_fits"]), "fits_observed": int(fits_observed), "finished_utc": utc_now(), "error": reason[:2000], "wall_s": float(wall_s)})

    def sweep_interrupted(self, *, is_alive=None) -> int:
        """Mark ``running`` attempts whose process is gone as ``interrupted``
        (charged, never refunded).  Returns the number swept."""
        if is_alive is None:
            def is_alive(pid, host):
                if host != socket.gethostname():
                    return True  # cannot tell; leave it running
                try:
                    os.kill(pid, 0)
                    return True
                except OSError:
                    return False
        n = 0
        for u, a in self.attempts().items():
            if a["state"] == "running" and not is_alive(a.get("pid"), a.get("hostname")):
                self._append({"attempt_uuid": u, "cell_id": a["cell_id"], "role": a["role"], "state": "interrupted",
                              "fits_planned": int(a["fits_planned"]), "finished_utc": utc_now(),
                              "error": "process not alive at sweep"})
                n += 1
        return n

    def summary(self) -> dict:
        ch = self.charged()
        return {"ledger": str(self.path), "cap": self.cap, "charged": ch, "remaining": self.cap - ch["fits"],
                "distinct_completed_cells": len(self.completed_by_cell()),
                "cells_with_repeat_completion": sorted(k for k, v in self.completed_by_cell().items() if len(v) > 1)}
