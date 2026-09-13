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

Cost
----
The charged total and the set of completed cells are maintained **incrementally**
in memory.  The first use in a process reads the whole file once; afterwards each
``start``/``complete``/``fail`` reads only the bytes other processes have appended
since (``_sync``, which seeks to the byte offset this process last consumed and
re-reads the whole file only when the file has shrunk or been replaced).  The
earlier implementation re-parsed the entire ledger twice per ``start`` --- once
for :meth:`charged` and once for :meth:`completed_by_cell` --- which made a run
quadratic in the cell count: measured on this hardware with a 2.5 kB payload,
``start`` cost 1.2 ms at the first of 2 499 cells and 83 ms at the last, 69.6 s in
total.  The state kept here is exactly what a full re-parse would produce, and
``tests/test_successor_20260913_common.py`` asserts that against interleaved
appends from two instances on one file.
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
    """An append-only attempt ledger with incremental in-memory state.

    ``charged`` and the completed-cell set are folded record by record as the
    file is read, so no operation re-parses the whole ledger after the first
    time this process looks at it.  Every read path calls :meth:`_sync` first,
    which picks up whatever other processes have appended since; ``start`` does
    that inside the lock, so the budget and duplicate checks see every writer.
    """

    def __init__(self, path: Path | str, *, cap: int, package: str, protocol_sha256: str):
        self.path = Path(path)
        self.cap = int(cap)
        self.package = package
        self.protocol_sha256 = protocol_sha256
        self._reset_state()

    # -- incremental state --------------------------------------------------------
    def _reset_state(self) -> None:
        self._pos = 0                       # bytes of the file already folded in
        self._mtime_ns = -1
        self._by_uuid: dict[str, dict] = {}
        self._completed: dict[str, list[dict]] = {}
        self._charged = {"fits": 0, "solves": 0, "attempts": 0, "by_state": {s: 0 for s in STATES}}

    @staticmethod
    def _cost(a: dict) -> tuple[int, int]:
        """What one attempt charges, in its current state.  A completed attempt is
        charged what it used; anything else (running/failed/interrupted) is charged
        what it planned, and is never refunded."""
        if a["state"] == "completed":
            return (int(a.get("fits_used", a["fits_planned"])),
                    int(a.get("solves_used", a.get("solves_planned", 0))))
        return int(a["fits_planned"]), int(a.get("solves_planned", 0))

    def _charge(self, a: dict, sign: int) -> None:
        fits, solves = self._cost(a)
        st = a["state"]
        self._charged["fits"] += sign * fits
        self._charged["solves"] += sign * solves
        self._charged["attempts"] += sign
        self._charged["by_state"][st] = self._charged["by_state"].get(st, 0) + sign
        if st == "completed":
            lst = self._completed.setdefault(a["cell_id"], [])
            if sign > 0:
                lst.append(a)
            else:
                for i, x in enumerate(lst):
                    if x is a:
                        lst.pop(i)
                        break
                if not lst:
                    self._completed.pop(a["cell_id"], None)

    def _check(self, r: dict) -> dict:
        if r.get("schema") != LEDGER_SCHEMA:
            raise ValueError(f"{self.path}: unexpected ledger schema {r.get('schema')!r}")
        if r.get("package") != self.package:
            raise ValueError(f"{self.path}: ledger belongs to package {r.get('package')!r}, not {self.package!r}")
        return r

    def _apply(self, r: dict) -> None:
        """Fold one record into the state, exactly as :meth:`attempts` would: the
        first record for an attempt uuid creates it, later ones update it in place
        (``started_utc`` excepted) and the charge moves with the state."""
        self._check(r)
        u = r["attempt_uuid"]
        prev = self._by_uuid.get(u)
        if prev is None:
            a = dict(r)
            self._by_uuid[u] = a
            self._charge(a, +1)
        else:
            self._charge(prev, -1)
            prev.update({k: v for k, v in r.items() if k != "started_utc"})
            self._charge(prev, +1)

    def _sync(self) -> None:
        """Fold in whatever has been appended since this process last looked.

        Only the tail is read.  The whole file is re-read only when it has shrunk
        or been replaced (which an append-only ledger never does, but a fresh run
        root reusing this object would), and a trailing partial line is left for
        the next call rather than parsed."""
        try:
            st = self.path.stat()
        except FileNotFoundError:
            if self._pos or self._by_uuid:
                self._reset_state()
            return
        if st.st_size < self._pos:
            self._reset_state()
        elif st.st_size == self._pos and st.st_mtime_ns == self._mtime_ns:
            return
        with open(self.path, "rb") as fh:
            fh.seek(self._pos)
            buf = fh.read()
        cut = buf.rfind(b"\n")
        if cut < 0:
            return
        for line in buf[:cut + 1].decode("utf-8").splitlines():
            if line.strip():
                self._apply(json.loads(line))
        self._pos += cut + 1
        self._mtime_ns = st.st_mtime_ns

    # -- reading ------------------------------------------------------------------
    def records(self) -> list[dict]:
        """Every record, re-parsed from the file.  The state above is what the rest
        of the class uses; this stays for callers (and tests) that want the raw
        append order, and for :meth:`state_matches_full_reparse`."""
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            out.append(self._check(json.loads(line)))
        return out

    def attempts(self) -> dict:
        """Latest state per attempt uuid (a running record followed by a
        completed/failed record for the same uuid is one attempt).  The returned
        records are copies, so a caller may iterate while the ledger is appended to."""
        self._sync()
        return {u: dict(a) for u, a in self._by_uuid.items()}

    def attempt(self, attempt_uuid: str) -> dict | None:
        """One attempt by uuid, without copying the whole map (``reduce`` asks this
        once per cell and a copy per cell would be quadratic again)."""
        self._sync()
        a = self._by_uuid.get(attempt_uuid)
        return dict(a) if a is not None else None

    def charged(self) -> dict:
        self._sync()
        c = self._charged
        return {"fits": c["fits"], "solves": c["solves"], "attempts": c["attempts"],
                "by_state": dict(c["by_state"])}

    def completed_by_cell(self) -> dict[str, list[dict]]:
        self._sync()
        return {k: [dict(a) for a in v] for k, v in self._completed.items()}

    def state_matches_full_reparse(self) -> bool:
        """True when the incrementally-maintained state equals what a full re-parse
        of the file produces.  Used by the tests; not used in the run path."""
        by_uuid: dict[str, dict] = {}
        for r in self.records():
            u = r["attempt_uuid"]
            if u in by_uuid:
                by_uuid[u].update({k: v for k, v in r.items() if k != "started_utc"})
            else:
                by_uuid[u] = dict(r)
        tot = {"fits": 0, "solves": 0, "attempts": 0, "by_state": {s: 0 for s in STATES}}
        done: dict[str, int] = {}
        for a in by_uuid.values():
            fits, solves = self._cost(a)
            tot["fits"] += fits
            tot["solves"] += solves
            tot["attempts"] += 1
            tot["by_state"][a["state"]] = tot["by_state"].get(a["state"], 0) + 1
            if a["state"] == "completed":
                done[a["cell_id"]] = done.get(a["cell_id"], 0) + 1
        mine = self.completed_by_cell()
        return (self.charged() == tot and self.attempts() == by_uuid
                and {k: len(v) for k, v in mine.items()} == done)

    # -- writing ------------------------------------------------------------------
    def _write_locked(self, rec: dict) -> None:
        """Append one record and fold it into the state.  The caller holds the lock
        and has already called :meth:`_sync`, so ``self._pos`` is the end of the
        file and this record is the only thing after it."""
        line = (json.dumps(rec, sort_keys=True) + "\n").encode("utf-8")
        with open(self.path, "ab") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        self._apply(rec)
        self._pos += len(line)
        try:
            self._mtime_ns = self.path.stat().st_mtime_ns
        except OSError:                      # pragma: no cover - the file was just written
            self._mtime_ns = -1

    def _append(self, rec: dict) -> None:
        rec = {"schema": LEDGER_SCHEMA, "package": self.package, "protocol_sha256": self.protocol_sha256, **rec}
        with _locked(self.path):
            self._sync()
            self._write_locked(rec)

    def start(self, cell: dict, *, allow_repeat: bool = False) -> str:
        """Reserve and charge an attempt for ``cell``.  Refuses (exit 2) when the
        cap would be exceeded; refuses (exit 5) a repeat of an already
        completed cell unless ``allow_repeat`` (which still charges again).

        Both checks read the state maintained incrementally by :meth:`_sync`, which
        runs inside the lock and picks up every record another process has appended.
        Neither re-parses the ledger."""
        with _locked(self.path):
            self._sync()
            repeat = cell["cell_id"] in self._completed
            if repeat and not allow_repeat:
                raise DuplicateCompletion(f"cell {cell['cell_id']} already has a completed attempt; refusing to re-run "
                                          "without --allow-repeat (a repeat is charged again and flagged)")
            if self._charged["fits"] + int(cell["n_fits"]) > self.cap:
                raise BudgetExceeded(f"charged {self._charged['fits']} + planned {cell['n_fits']} exceeds cap {self.cap}")
            u = str(uuid.uuid4())
            rec = {"schema": LEDGER_SCHEMA, "package": self.package, "protocol_sha256": self.protocol_sha256,
                   "attempt_uuid": u, "cell_id": cell["cell_id"], "role": cell["role"], "state": "running",
                   "fits_planned": int(cell["n_fits"]), "solves_planned": int(cell.get("n_solves", 0)),
                   "started_utc": utc_now(), "hostname": socket.gethostname(), "pid": os.getpid(),
                   "repeat": bool(repeat)}
            self._write_locked(rec)
        return u

    def complete(self, attempt_uuid: str, cell: dict, *, fits_used: int, solves_used: int, output_path: Path, output_sha256: str, wall_s: float) -> None:
        self._append({"attempt_uuid": attempt_uuid, "cell_id": cell["cell_id"], "role": cell["role"], "state": "completed",
                      "fits_planned": int(cell["n_fits"]), "fits_used": int(fits_used), "solves_used": int(solves_used),
                      "finished_utc": utc_now(), "output": str(output_path), "output_sha256": output_sha256, "wall_s": float(wall_s)})

    def fail(self, attempt_uuid: str, cell: dict, *, reason: str, wall_s: float) -> None:
        self._append({"attempt_uuid": attempt_uuid, "cell_id": cell["cell_id"], "role": cell["role"], "state": "failed",
                      "fits_planned": int(cell["n_fits"]), "finished_utc": utc_now(), "error": reason[:2000], "wall_s": float(wall_s)})

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
        for u, a in sorted(self.attempts().items()):
            if a["state"] == "running" and not is_alive(a.get("pid"), a.get("hostname")):
                self._append({"attempt_uuid": u, "cell_id": a["cell_id"], "role": a["role"], "state": "interrupted",
                              "fits_planned": int(a["fits_planned"]), "finished_utc": utc_now(),
                              "error": "process not alive at sweep"})
                n += 1
        return n

    def summary(self) -> dict:
        ch = self.charged()
        done = self.completed_by_cell()
        return {"ledger": str(self.path), "cap": self.cap, "charged": ch, "remaining": self.cap - ch["fits"],
                "distinct_completed_cells": len(done),
                "cells_with_repeat_completion": sorted(k for k, v in done.items() if len(v) > 1)}
