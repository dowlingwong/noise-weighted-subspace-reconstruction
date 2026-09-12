"""Execution accounting for the T22 fit budget: identity, union and reservations.

This module is *additive*.  It does not change how ``FIT_LEDGER.json`` is built
or what it means: the archived ledger stays the record of what was executed, and
``t22_tidmad_narrowband.audit_budget`` stays the only thing that writes one.
What is added here is the accounting the ledger alone cannot do.

Why
---
The runner used to spend ``max(ledger_total, archive_total)``.  That is right
only when the two record the *same* executions.  It is wrong for a true union of
*disjoint* records: an execution recorded only in the ledger plus a different
execution present only on disk should sum, not max, and ``max`` silently makes
room under the cap for the smaller of the two.  :func:`union_executed`
deduplicates by execution identity instead, so overlapping records count once
and disjoint records count twice.

Stable execution identity
-------------------------
An *execution* is one invocation of a fit-executing stage that consumed compute.
Two records denote the same execution when they share an identity key.  There
are three kinds, in decreasing strength:

``reservation``
    ``r:<uuid>``.  A UUID4 minted by :func:`reserve` *before* any compute, stored
    in the reservation store and carried into the stage output's provenance under
    ``provenance["fit_reservation"]["uuid"]`` (see :func:`reservation_provenance`).
    It is stable under everything: renaming or moving the output, rewriting its
    bytes, losing the ledger, or the process dying before the output is written.
    It is the only key that exists *before* the output does, so it is the only
    one that can identify an execution that crashed.

``digest``
    ``d:<sha256>|<stage>|<output_file>|<smoke>``.  Content addressing: the
    SHA-256 of the whole stage-output file (its provenance block included),
    qualified by the three execution coordinates that the ledger, a disk scan and
    a reservation all record identically for the same execution.  It is stable
    under renaming or moving the *directory* -- the directory name is
    deliberately not part of the key -- which is what makes a relocated output
    still one execution.  Qualifying by stage / file name / smoke flag can only
    ever *split* a group, never merge more of them, so it is conservative for a
    cap.

``location``
    ``l:<stage>|<output_dir>|<output_file>``.  The fallback for a record that
    carries neither a reservation UUID nor a digest.  It merges with nothing:
    such a record is always counted on its own.

Reservations
------------
Fits are reserved *before* they are executed, under an exclusive process lock, so
that compute spent by a process that then crashed stays visible and a retry adds
to the expenditure instead of resetting it.  States are ``pending`` ->
``running`` -> ``completed`` / ``failed``, with ``interrupted`` reached from
``running`` by :func:`sweep_interrupted` when the owning process is gone.
``failed`` and ``interrupted`` are **never refunded**: they consumed compute.
The charged total is

    charged = union(completed executions) + pending + running + failed + interrupted

so a crash-and-retry cycle accumulates rather than resets.

Limits -- do not overstate these
--------------------------------
* Identity by digest merges two records of byte-identical output files.  Where
  both carry reservation uuids the uuids win and the executions stay apart (see
  :func:`_components`); where *neither* does, two genuinely distinct executions
  that produced identical bytes are still merged into one.  Reserving before
  executing is what removes that case, so an execution that ran without a
  reservation keeps it.
* Conversely, a ledger entry whose recorded digest is stale or a placeholder does
  *not* merge with the file on disk; it is then counted twice.  That is the
  conservative direction for a cap, but it is not correct deduplication.
* A stage output that is written, ledgered and *then deleted* is invisible to a
  disk scan; only the ledger or a reservation record keeps it charged.
* The lock serialises reservations against other processes that use this module
  on the same filesystem.  It serialises nothing against a process that edits the
  store by hand, and ``flock`` is advisory and unreliable on some network
  filesystems.
* The lock makes two *cooperating* runners serialise.  It does nothing about a
  process that writes the ledger or the store without taking it -- a hand-edit,
  another tool -- and it cannot make a stage that never reserved visible before
  its output exists.
* Nothing here is crash-*safe* in the sense of losing no work: a stage that dies
  mid-fit loses its fits.  What is guaranteed is that the compute it spent stays
  charged, so a retry adds to the expenditure instead of resetting it.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import socket
import time
import uuid as _uuid
from pathlib import Path

#: Reservation store, written beside ``FIT_LEDGER.json``.
RESERVATION_NAME = "FIT_RESERVATIONS.json"
RESERVATION_SCHEMA = "mlst-t22-fit-reservations/1"
#: The fit ledger this module reads (and never writes).
LEDGER_SCHEMA = "mlst-t22-fit-ledger/1"
#: Where a reservation UUID is carried inside a stage output's provenance.
PROVENANCE_KEY = "fit_reservation"

STATES = ("pending", "running", "completed", "failed", "interrupted")
#: Reservations holding a charge without a completed output behind them.
OPEN_STATES = ("pending", "running")
CHARGED_STATES = OPEN_STATES + ("failed", "interrupted")
_IDENTITY_ORDER = ("reservation", "digest", "location")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --- stable execution identity -------------------------------------------------
def execution_id(*, reservation_uuid=None, output_sha256=None, stage=None,
                 output_file=None, output_dir=None, smoke=False):
    """``(identifier, kind)`` for one execution record.

    ``kind`` is ``"reservation"``, ``"digest"`` or ``"location"``; the module
    docstring says exactly what each one is stable under.
    """
    if reservation_uuid:
        return f"r:{reservation_uuid}", "reservation"
    if output_sha256:
        return f"d:{output_sha256}|{stage}|{output_file}|{int(bool(smoke))}", "digest"
    return f"l:{stage}|{output_dir}|{output_file}", "location"


def _merge_keys(reservation_uuid, output_sha256, stage, output_file, smoke) -> list:
    """The keys on which two records *merge*.

    A location is an identifier but never a merge key: two records in different
    run trees can share a directory name without being the same execution.
    """
    keys = []
    if reservation_uuid:
        keys.append(f"r:{reservation_uuid}")
    if output_sha256:
        keys.append(f"d:{output_sha256}|{stage}|{output_file}|{int(bool(smoke))}")
    return keys


def uuid_in_output(js):
    """The reservation UUID a stage output carries in its provenance, if any."""
    if not isinstance(js, dict):
        return None
    block = (js.get("provenance") or {}).get(PROVENANCE_KEY)
    if isinstance(block, dict) and block.get("uuid"):
        return str(block["uuid"])
    return None


def reservation_provenance(reservation: dict) -> dict:
    """The block a stage writes into ``provenance[PROVENANCE_KEY]``, tying its
    output back to the reservation that paid for it."""
    return {"schema": RESERVATION_SCHEMA, "uuid": reservation["uuid"],
            "stage": reservation.get("stage"), "planned": reservation.get("planned"),
            "created_utc": reservation.get("created_utc")}


# --- the fit ledger, read-only --------------------------------------------------
def _ledger_sums(ledger: dict) -> dict:
    real = [e for e in ledger.get("entries", []) if not e.get("smoke")]
    return {"executed_total": int(sum(int(e.get("fits_executed") or 0) for e in real)),
            "unique_scientific_total": int(sum(int(e.get("fits_executed") or 0) for e in real
                                               if not e.get("deterministic_rescore_of")))}


def validate_fit_ledger(ledger, path=None) -> dict:
    """Refuse a ledger that cannot be trusted: wrong schema, malformed entries, or
    **stale** stored totals that disagree with its own entries."""
    where = f"fit ledger {path}" if path else "fit ledger"
    if not isinstance(ledger, dict) or ledger.get("schema") != LEDGER_SCHEMA:
        raise SystemExit(f"{where}: schema must be {LEDGER_SCHEMA!r}")
    entries = ledger.get("entries")
    if not isinstance(entries, list) or not all(
            isinstance(e, dict) and isinstance(e.get("fits_executed"), int)
            and not isinstance(e["fits_executed"], bool) and e["fits_executed"] >= 0
            for e in entries):
        raise SystemExit(f"{where}: every entry needs a non-negative integer fits_executed")
    sums = _ledger_sums(ledger)
    stored = {k: ledger.get(k) for k in sums if k in ledger}
    bad = {k: (stored[k], sums[k]) for k in stored if stored[k] != sums[k]}
    if bad:
        raise SystemExit(f"{where}: stored totals disagree with its entries ("
                         + ", ".join(f"{k}: stored {s}, entries {c}"
                                     for k, (s, c) in sorted(bad.items()))
                         + "); refusing to trust it")
    return sums


def require_fit_ledger(path) -> dict:
    """Load the ledger at ``path``; refuse a missing or untrustworthy one."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"fit ledger {p} does not exist; cumulative expenditure cannot be "
                         "established (reconstruct it with --stage audit-budget)")
    try:
        ledger = json.loads(p.read_text())
    except ValueError as exc:
        raise SystemExit(f"fit ledger {p} is not valid JSON: {exc}")
    validate_fit_ledger(ledger, p)
    return ledger


# --- candidate records ----------------------------------------------------------
def _candidate(*, source, stage, output_file, output_dir, ref, fits, smoke,
               digest=None, reservation_uuid=None, inference_only=None,
               config_id=None, refits_models=True, finished=None, state="completed"):
    ident, kind = execution_id(reservation_uuid=reservation_uuid, output_sha256=digest,
                               stage=stage, output_file=output_file,
                               output_dir=output_dir, smoke=smoke)
    return {"source": source, "stage": stage, "output_file": output_file,
            "output_dir": output_dir, "ref": ref, "fits": int(fits or 0),
            "smoke": bool(smoke), "digest": digest, "reservation_uuid": reservation_uuid,
            "inference_only": inference_only, "config_id": config_id or f"stage:{stage}",
            "refits_models": bool(refits_models), "finished": finished or "",
            "state": state, "execution_id": ident, "identity": kind,
            "keys": _merge_keys(reservation_uuid, digest, stage, output_file, smoke)}


def _from_ledger_entry(entry: dict, base) -> dict:
    out_dir, out_file = entry.get("output_dir"), entry.get("output_file")
    ref = f"{out_dir}/{out_file}"
    return _candidate(source="ledger", stage=entry.get("stage"), output_file=out_file,
                      output_dir=out_dir, ref=str(Path(base) / ref) if base else ref,
                      fits=entry.get("fits_executed"), smoke=entry.get("smoke"),
                      digest=entry.get("output_sha256"),
                      reservation_uuid=entry.get("reservation_uuid"),
                      inference_only=bool(entry.get("deterministic_rescore_of")),
                      config_id=entry.get("config_id"),
                      refits_models=entry.get("refits_models", True),
                      finished=entry.get("output_finished_utc"))


def _from_scan_row(row: dict) -> dict:
    p = Path(row["output"])
    stage = p.stem[len("t22_"):] if p.stem.startswith("t22_") else p.stem
    digest = res_uuid = finished = None
    try:
        data = p.read_bytes()
    except OSError:
        data = None
    if data is not None:
        digest = hashlib.sha256(data).hexdigest()
        try:
            js = json.loads(data)
        except ValueError:
            js = None
        if isinstance(js, dict):
            res_uuid = uuid_in_output(js)
            finished = (js.get("provenance") or {}).get("finished_utc")
    # A bare output on disk carries no re-score annotation: it counts as a new
    # scientific evaluation unless a ledger entry it merges with says otherwise.
    return _candidate(source="scan", stage=stage, output_file=p.name,
                      output_dir=p.parent.name, ref=str(p), fits=row.get("fits"),
                      smoke=row.get("smoke"), digest=digest, reservation_uuid=res_uuid,
                      inference_only=None, finished=finished)


def _from_reservation(r: dict) -> dict:
    fits = r.get("actual_fits")
    fits = r.get("planned") if fits is None else fits
    out = Path(str(r.get("output_ref") or ""))
    return _candidate(source="reservation", stage=r.get("stage"),
                      output_file=out.name or None, output_dir=out.parent.name or None,
                      ref=r.get("output_ref") or f"reservation:{r['uuid']}", fits=fits,
                      smoke=r.get("smoke", False), digest=r.get("output_sha256"),
                      reservation_uuid=r.get("uuid"),
                      inference_only=r.get("inference_only"), config_id=r.get("config_id"),
                      refits_models=r.get("refits_models", True),
                      finished=r.get("completed_utc"), state="completed")


def _merge(group: list) -> dict:
    """Collapse candidates sharing an identity key into one execution record."""
    fits = [c["fits"] for c in group]
    opinions = [c["inference_only"] for c in group if c["inference_only"] is not None]
    kind = sorted({c["identity"] for c in group}, key=_IDENTITY_ORDER.index)[0]
    return {
        # the id of the strongest identity present, so a group holding a
        # reservation is named by its uuid rather than by its bytes
        "execution_id": sorted(c["execution_id"] for c in group
                               if c["identity"] == kind)[0],
        "identity": kind,
        "stage": next((c["stage"] for c in group if c["stage"]), None),
        "output_file": next((c["output_file"] for c in group if c["output_file"]), None),
        "fits": max(fits),
        "fits_disagree": len(set(fits)) > 1,
        # smoke only if every source says so; scientific unless every source with
        # an opinion says it was a deterministic re-score
        "smoke": all(c["smoke"] for c in group),
        "inference_only": bool(opinions) and all(opinions),
        "refits_models": any(c["refits_models"] for c in group),
        "config_id": next((c["config_id"] for c in group if c["config_id"]), None),
        "finished": min((c["finished"] for c in group if c["finished"]), default=""),
        "sources": sorted({c["source"] for c in group}),
        "refs": sorted({c["ref"] for c in group if c["ref"]}),
        "records_merged": len(group),
    }


def _digest_key(c: dict):
    return next((k for k in c["keys"] if k.startswith("d:")), None)


def _contested_digests(cands: list) -> set:
    """Digest keys claimed by more than one reservation uuid.

    Two *distinct* reservations whose outputs happen to be byte-identical share a
    digest key.  The digest cannot then decide which execution a record belongs
    to, and merging on it would collapse two executions that really did both
    spend compute -- exactly the way a cap is silently widened.
    """
    claims: dict = {}
    for c in cands:
        dk = _digest_key(c)
        if dk and c["reservation_uuid"]:
            claims.setdefault(dk, set()).add(c["reservation_uuid"])
    return {k for k, uuids in claims.items() if len(uuids) > 1}


def _components(cands: list) -> list:
    """Group candidate records into executions.

    Union-find over the merge keys, with one exclusion: **a reservation uuid is
    an identity, not a hint**.  Where a digest key is claimed by two different
    uuids, the records carrying a uuid merge on their uuid alone and ignore the
    digest, so the two executions stay apart.  A record of those same bytes that
    carries no uuid cannot be attributed to either; it merges only with other
    unclaimed records of the same bytes and is counted as its own execution,
    which over-counts rather than under-counts a cap.
    """
    contested = _contested_digests(cands)
    parent: dict = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, c in enumerate(cands):
        node = f"c{i}"
        parent.setdefault(node, node)
        for k in c["keys"]:
            if k in contested and c["reservation_uuid"]:
                continue                 # identity is the uuid, not the bytes
            parent.setdefault(k, k)
            join(node, k)
    groups: dict = {}
    for i, c in enumerate(cands):
        groups.setdefault(find(f"c{i}"), []).append(c)
    return list(groups.values())


def union_executed(ledger=None, scan=None, *, ledger_path=None, reservations=None) -> dict:
    """The deduplicated union of what the ledger, the disk scan and the
    reservation store each saw.

    ``ledger``        a loaded ``mlst-t22-fit-ledger/1`` dict, or ``None``.
    ``scan``          the dict ``t22_tidmad_narrowband.scan_executed_fits``
                      returns, or ``None``.
    ``reservations``  a loaded reservation store, or ``None``.

    Every record carries the source(s) that saw it.  Totals:

    ``executed``                  every non-smoke fit executed, re-scores and
                                  restarts included.  The cap quantity, and the
                                  archived ledger's ``executed_total``.
    ``unique_scientific``         ``executed`` minus inference-only recomputation:
                                  the archived ``unique_scientific_total``.
    ``inference_only_fits``       fits executed by deterministic re-scores -- real
                                  compute, no new scientific evaluation.
    ``fitted_models``             executed fits that instantiated and fitted a
                                  model.  Equal to ``executed`` for every T22
                                  record; a record marked ``refits_models: false``
                                  (a re-score of stored fits) separates them.
    ``restart_fits``              fits of scientific records repeating a
                                  configuration an earlier scientific record had
                                  already evaluated.
    ``scientific_configurations`` distinct configuration ids among scientific
                                  records with at least one fit.
    ``reserved_open_fits``        pending + running + failed + interrupted.
    ``charged``                   ``executed + reserved_open_fits``.
    """
    cands, counts = [], {"ledger": 0, "scan": 0, "reservations": 0}
    if ledger is not None:
        validate_fit_ledger(ledger, ledger_path)
        base = Path(ledger_path).parent if ledger_path else None
        for e in ledger.get("entries", []):
            cands.append(_from_ledger_entry(e, base))
            counts["ledger"] += 1
    if scan is not None:
        for row in scan.get("outputs", []):
            cands.append(_from_scan_row(row))
            counts["scan"] += 1
    open_reservations = []
    if reservations is not None:
        validate_reservations(reservations)
        for r in reservations.get("reservations", []):
            if r.get("state") == "completed":
                cands.append(_from_reservation(r))
                counts["reservations"] += 1
            elif r.get("state") in CHARGED_STATES:
                open_reservations.append(r)

    records = [_merge(g) for g in _components(cands)]
    records.sort(key=lambda r: (r["finished"], r["execution_id"]))

    real = [r for r in records if not r["smoke"]]
    executed = sum(r["fits"] for r in real)
    inference_only_fits = sum(r["fits"] for r in real if r["inference_only"])
    fitted_models = sum(r["fits"] for r in real if r["refits_models"])
    smoke_executed = sum(r["fits"] for r in records if r["smoke"])

    seen, restart_fits, configs = set(), 0, []
    for r in real:
        if r["inference_only"] or r["fits"] <= 0:
            continue
        if r["config_id"] in seen:
            restart_fits += r["fits"]
            r["restart_of_configuration"] = True
        else:
            seen.add(r["config_id"])
            configs.append(r["config_id"])

    # An open reservation whose output nevertheless landed is already counted by
    # the record that carries its uuid; it holds only the remainder.  Without
    # this, one execution that died between writing its output and completing its
    # reservation would be charged twice.
    counted = {r["execution_id"]: r["fits"] for r in real}

    def _open_charge(r):
        return max(0, _reservation_charge(r) - counted.get(f"r:{r.get('uuid')}", 0))

    open_fits = sum(_open_charge(r) for r in open_reservations)
    return {
        "records": records,
        "executed": int(executed),
        "unique_scientific": int(executed - inference_only_fits),
        "inference_only_fits": int(inference_only_fits),
        "fitted_models": int(fitted_models),
        "restart_fits": int(restart_fits),
        "scientific_configurations": len(configs),
        "configuration_ids": configs,
        "smoke_executed": int(smoke_executed),
        "reserved_open_fits": int(open_fits),
        "open_reservations": [{"uuid": r.get("uuid"), "state": r.get("state"),
                               "stage": r.get("stage"), "charge": _open_charge(r),
                               "reserved": _reservation_charge(r)}
                              for r in open_reservations],
        "charged": int(executed + open_fits),
        "source_records": counts,
        "duplicates_merged": len(cands) - len(records),
        "ledger_only": sum(1 for r in records if r["sources"] == ["ledger"]),
        "scan_only": sum(1 for r in records if r["sources"] == ["scan"]),
        "seen_by_both": sum(1 for r in records if {"ledger", "scan"} <= set(r["sources"])),
    }


def charged_total(ledger=None, scan=None, reservations=None, *, ledger_path=None) -> int:
    """``union(completed) + pending + running + failed + interrupted``."""
    return union_executed(ledger, scan, ledger_path=ledger_path,
                          reservations=reservations)["charged"]


# --- the reservation store --------------------------------------------------------
def _reservation_charge(r: dict) -> int:
    """What a reservation costs *against the cap*.

    A failed or interrupted attempt consumed compute and is never refunded; an
    over-run charges what actually ran.  A reservation marked ``smoke`` charges
    nothing: the protocol exempts smoke runs from the cap and totals them apart,
    exactly as the ledger does for a smoke entry.
    """
    if r.get("smoke"):
        return 0
    return max(int(r.get("planned") or 0), int(r.get("actual_fits") or 0))


def validate_reservations(store, path=None) -> None:
    where = f"reservation store {path}" if path else "reservation store"
    if not isinstance(store, dict) or store.get("schema") != RESERVATION_SCHEMA:
        raise SystemExit(f"{where}: schema must be {RESERVATION_SCHEMA!r}")
    rows = store.get("reservations")
    if not isinstance(rows, list):
        raise SystemExit(f"{where}: 'reservations' must be a list")
    for r in rows:
        if not isinstance(r, dict) or not r.get("uuid"):
            raise SystemExit(f"{where}: every reservation needs a uuid")
        if r.get("state") not in STATES:
            raise SystemExit(f"{where}: reservation {r.get('uuid')} has state "
                             f"{r.get('state')!r}, not one of {STATES}")
        if not isinstance(r.get("planned"), int) or isinstance(r["planned"], bool) \
                or r["planned"] < 0:
            raise SystemExit(f"{where}: reservation {r['uuid']} needs a non-negative "
                             "integer planned")
    uuids = [r["uuid"] for r in rows]
    if len(set(uuids)) != len(uuids):
        raise SystemExit(f"{where}: duplicate reservation uuids")
    total = sum(_reservation_charge(r) for r in rows if r.get("state") in CHARGED_STATES)
    if "charged_open_total" in store and int(store["charged_open_total"]) != total:
        raise SystemExit(f"{where}: stored charged_open_total {store['charged_open_total']} "
                         f"disagrees with its reservations ({total}); refusing to trust it")


@contextlib.contextmanager
def locked(store_path):
    """Exclusive lock around a read-modify-write of the file at ``store_path``.

    The lock file is created with ``O_CREAT | O_EXCL`` so the first process to
    need it creates it without clobbering a concurrent creator; mutual exclusion
    itself is ``fcntl.flock(LOCK_EX)``, which serialises the whole
    read-modify-write.  The lock file is never unlinked -- unlinking races with
    processes that already hold a descriptor on it.

    Used for the reservation store and, by the runner, for the fit ledger's own
    append: both are read-modify-write cycles that a second process must not
    interleave with.  ``flock`` is advisory -- it binds only processes that take
    the same lock, and on some network filesystems it does not bind even those.
    :func:`lock_is_effective` probes an actual directory rather than assuming.
    """
    lock = Path(str(store_path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
    except FileExistsError:
        fd = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def lock_is_effective(directory, *, workers=6, rounds=25, timeout_s=120):
    """Probe whether :func:`locked` really excludes on *this* filesystem.

    ``flock`` is advisory, and on some network filesystems (NFS without a lock
    manager, some FUSE mounts) it silently does not exclude at all.  Rather than
    assume, this forks ``workers`` children that each do ``rounds`` locked
    read-modify-write increments of a counter file in ``directory``.  A working
    lock ends at ``workers * rounds``; anything less is a lost update, which is
    exactly the failure the lock exists to prevent.

    Returns ``{"effective": bool, "observed": int, "expected": int, ...}``.  It
    is a probe, not a proof: a pass means no lost update was observed in this
    many contended cycles on this mount, nothing stronger.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    counter = directory / "lock_probe.json"
    counter.write_text(json.dumps({"n": 0}))
    errors = directory / "lock_probe_errors"
    errors.mkdir(exist_ok=True)
    kids = []
    for w in range(int(workers)):
        pid = os.fork()
        if pid == 0:                                   # pragma: no cover - child
            try:
                for _ in range(int(rounds)):
                    with locked(counter):
                        n = json.loads(counter.read_text())["n"]
                        time.sleep(0.0005)             # widen the race window
                        counter.write_text(json.dumps({"n": n + 1}))
                os._exit(0)
            except BaseException as exc:
                try:
                    (errors / f"{w}.txt").write_text(f"{type(exc).__name__}: {exc}")
                except BaseException:
                    pass
                os._exit(1)
        kids.append(pid)
    status_by_pid, bad = {}, 0
    deadline = time.time() + float(timeout_s)
    for pid in kids:
        while True:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                ok = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
                status_by_pid[pid] = "ok" if ok else f"raw status {status}"
                bad += 0 if ok else 1
                break
            if time.time() > deadline:                 # pragma: no cover - hung child
                os.kill(pid, 9)
                os.waitpid(pid, 0)
                status_by_pid[pid] = "timed out"
                bad += 1
                break
            time.sleep(0.01)
    observed = int(json.loads(counter.read_text())["n"])
    expected = int(workers) * int(rounds)
    reported = sorted(f.read_text() for f in errors.glob("*.txt"))
    counter.unlink(missing_ok=True)
    Path(str(counter) + ".lock").unlink(missing_ok=True)
    for f in errors.glob("*.txt"):
        f.unlink()
    errors.rmdir()
    return {"effective": bool(observed == expected and not bad), "observed": observed,
            "expected": expected, "lost_updates": expected - observed,
            "failed_workers": bad, "worker_status": status_by_pid,
            "worker_errors": reported, "directory": str(directory)}


def load_reservations(path, *, required=False, cap=None):
    """The reservation store at ``path`` (``None`` if absent and not required)."""
    p = Path(path)
    if not p.is_file():
        if required:
            raise SystemExit(f"reservation store {p} does not exist")
        return None
    try:
        store = json.loads(p.read_text())
    except ValueError as exc:
        raise SystemExit(f"reservation store {p} is not valid JSON: {exc}")
    validate_reservations(store, p)
    if cap is not None and int(store.get("hard_cap", cap)) != int(cap):
        raise SystemExit(f"reservation store {p} records hard cap {store.get('hard_cap')}, "
                         f"the protocol declares {cap}")
    return store


def _load_unlocked(path, cap=None):
    """Load inside the lock.  ``cap`` given creates an empty store when absent;
    ``cap`` omitted requires the store to exist already."""
    p = Path(path)
    if not p.is_file():
        if cap is None:
            raise SystemExit(f"reservation store {p} does not exist")
        return {"schema": RESERVATION_SCHEMA, "hard_cap": int(cap),
                "reservations": [], "charged_open_total": 0}
    return load_reservations(p, required=True, cap=cap)


def _save_unlocked(path, store) -> None:
    p = Path(path)
    store["charged_open_total"] = int(sum(_reservation_charge(r) for r in store["reservations"]
                                          if r.get("state") in CHARGED_STATES))
    store["updated_utc"] = _utc_now()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, p)


def _proc_start_ticks(pid):
    """Field 22 of ``/proc/<pid>/stat``, the process start time.  Recorded with a
    reservation so a recycled pid is not mistaken for the original owner."""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
    except (OSError, ValueError):
        return None
    try:
        return int(stat[stat.rindex(")") + 2:].split()[19])
    except (ValueError, IndexError):
        return None


def process_alive(pid, hostname=None, start_ticks=None):
    """``True`` / ``False`` / ``None`` (cannot tell: another host, or no pid)."""
    if pid is None:
        return None
    if hostname and hostname != socket.gethostname():
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError):
        return None
    if start_ticks is not None:
        now = _proc_start_ticks(pid)
        if now is not None and now != start_ticks:
            return False            # the pid was recycled: the owner is gone
    return True


def _find(store, uid):
    for r in store["reservations"]:
        if r["uuid"] == uid:
            return r
    raise SystemExit(f"no reservation {uid} in the store")


def reserve(path, cap, planned, meta=None, *, stage=None, spent=0, exposure=None,
            config_id=None, smoke=False, now=None):
    """Reserve ``planned`` fits *before* executing them.  Returns the reservation.

    ``spent`` is the deduplicated total of *completed* executions (from
    :func:`union_executed`); the open charge is recomputed here under the lock, so
    two concurrent reservations cannot both see the same room and both take it.
    The reservation is refused, and nothing is written, when

        spent + open charge + planned > cap.

    A ``smoke`` reservation is recorded but charges nothing: the protocol exempts
    smoke runs from the cap and totals them apart.  Neither it nor a zero-fit
    reservation is ever refused: a charge of nothing cannot cross a cap, even one
    that is already spent.
    """
    planned = int(planned)
    if planned < 0:
        raise ValueError("planned must be non-negative")
    stamp = now or _utc_now()
    with locked(path):
        store = _load_unlocked(path, cap)
        open_charge = int(sum(_reservation_charge(r) for r in store["reservations"]
                              if r.get("state") in CHARGED_STATES))
        before = int(spent) + open_charge
        # a smoke run is exempt from the cap and totalled apart, so it reserves
        # its fits (they are recorded) without charging for them
        charge = 0 if smoke else planned
        # a reservation that charges nothing cannot cross a cap, and must not be
        # refused by one that is already spent: a zero-fit reservation exists to
        # record and bound a stage's data access, not to buy compute
        if charge and before + charge > int(cap):
            raise SystemExit(
                f"reservation refused: charged {before} (completed {int(spent)}, open "
                f"reservations {open_charge}) + planned {planned} = {before + charge} "
                f"would exceed the hard cap {cap}")
        row = {"uuid": str(_uuid.uuid4()), "state": "pending", "stage": stage,
               "planned": planned, "actual_fits": None, "smoke": bool(smoke),
               "config_id": config_id, "created_utc": stamp, "heartbeat_utc": stamp,
               "pid": os.getpid(), "hostname": socket.gethostname(),
               "pid_start_ticks": _proc_start_ticks(os.getpid()),
               "output_ref": None, "output_sha256": None, "reason": None,
               "charged_before": before, "meta": dict(meta or {}),
               "exposure": list(exposure or []),
               "history": [{"state": "pending", "utc": stamp}]}
        store["reservations"].append(row)
        _save_unlocked(path, store)
    return dict(row)


def _transition(path, uid, new_state, *, allowed, now=None, **fields):
    stamp = now or _utc_now()
    with locked(path):
        store = _load_unlocked(path)
        row = _find(store, uid)
        if allowed is not None and row["state"] not in allowed:
            raise SystemExit(f"reservation {uid} is {row['state']}, not one of {allowed}; "
                             f"refusing to move it to {new_state}")
        row.update(fields)
        row["state"] = new_state
        row["heartbeat_utc"] = stamp
        row.setdefault("history", []).append({"state": new_state, "utc": stamp})
        _save_unlocked(path, store)
        return dict(row)


def mark_running(path, uid, *, pid=None, now=None):
    """The reservation's compute has started; record who owns it."""
    pid = os.getpid() if pid is None else int(pid)
    return _transition(path, uid, "running", allowed=("pending", "running"), now=now,
                       pid=pid, hostname=socket.gethostname(),
                       pid_start_ticks=_proc_start_ticks(pid))


@contextlib.contextmanager
def running(path, uid, *, now=None):
    """Hold a reservation ``running`` for the duration of its compute.

    Enter: ``pending`` -> ``running``, recording this process as the owner.  Any
    exception -- an ordinary error, a refusal, a signal that unwinds -- leaves
    it ``failed``, which is charged and never refunded, and then propagates
    unchanged.  Leaving without an exception leaves it ``running``: the caller
    marks it ``completed`` once the output it paid for is on disk.

    A death that unwinds nothing (SIGKILL, a node going away) runs no handler at
    all.  The reservation stays ``running``, which is still charged, until
    :func:`sweep_interrupted` finds the owning process gone.
    """
    mark_running(path, uid, now=now)
    try:
        yield uid
    except BaseException as exc:                    # noqa: BLE001 - re-raised below
        try:
            mark_failed(path, uid, f"{type(exc).__name__}: {exc}"[:500])
        except BaseException:                       # pragma: no cover - never mask
            pass
        raise


def heartbeat(path, uid, now=None):
    """Refresh the liveness timestamp of a running reservation."""
    stamp = now or _utc_now()
    with locked(path):
        store = _load_unlocked(path)
        row = _find(store, uid)
        row["heartbeat_utc"] = stamp
        _save_unlocked(path, store)
        return dict(row)


def mark_completed(path, uid, actual_fits, output_ref, *, output_sha256=None,
                   inference_only=None, now=None):
    """The output landed.  The charge becomes what actually ran: an under-run
    releases the unspent remainder, an over-run keeps the larger number."""
    return _transition(path, uid, "completed", allowed=("pending", "running"), now=now,
                       actual_fits=int(actual_fits),
                       output_ref=str(output_ref) if output_ref else None,
                       output_sha256=output_sha256, inference_only=inference_only,
                       completed_utc=now or _utc_now())


def mark_failed(path, uid, reason, *, actual_fits=None, now=None):
    """The attempt failed.  It consumed compute; the charge is **not** refunded."""
    return _transition(path, uid, "failed", allowed=("pending", "running"), now=now,
                       reason=str(reason),
                       actual_fits=None if actual_fits is None else int(actual_fits))


def _age_seconds(then, now) -> float:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    try:
        return time.mktime(time.strptime(now, fmt)) - time.mktime(time.strptime(then, fmt))
    except (TypeError, ValueError):
        return 0.0


def sweep_interrupted(path, now=None, *, stale_after_s=None, is_alive=None):
    """Reclassify ``running`` reservations whose owning process is gone.

    A reservation is swept to ``interrupted`` when its recorded pid is on this
    host and no longer alive (the recorded ``/proc`` start time guarding against
    pid reuse), or when ``stale_after_s`` is given and its heartbeat is older than
    that.  **Nothing is refunded**: the compute it spent before dying is still
    charged.  A reservation owned by another host cannot be probed from here; it
    is left ``running`` -- still charged -- and reported as unverifiable.
    """
    stamp = now or _utc_now()
    alive_fn = is_alive or process_alive
    swept, unverifiable = [], []
    with locked(path):
        store = _load_unlocked(path)
        for row in store["reservations"]:
            if row.get("state") != "running":
                continue
            alive = alive_fn(row.get("pid"), row.get("hostname"), row.get("pid_start_ticks"))
            stale = (alive is not False and stale_after_s is not None
                     and _age_seconds(row.get("heartbeat_utc"), stamp) > float(stale_after_s))
            if alive is False or stale:
                row["state"] = "interrupted"
                row["reason"] = ("owning process gone" if alive is False
                                 else f"heartbeat older than {stale_after_s}s")
                row.setdefault("history", []).append({"state": "interrupted", "utc": stamp})
                swept.append(row["uuid"])
            elif alive is None:
                unverifiable.append(row["uuid"])
        if swept:
            _save_unlocked(path, store)
        charged = int(sum(_reservation_charge(r) for r in store["reservations"]
                          if r.get("state") in CHARGED_STATES))
    return {"swept": swept, "unverifiable": unverifiable, "charged_open_total": charged}


def attach_exposure(path, uid, *, dataset_id, region=None, note=None, now=None):
    """Record which dataset file and time region a reservation intends to touch.

    This is a *record*, and it changes no charge.  Nothing in *this* module reads
    it to grant, widen or check access to anything.  The enforcement lives where
    the files are opened: ``t22_tidmad_narrowband.exposure_allowance`` makes a
    stage's declared exposure binding for the duration of that stage, and
    ``open_instrument_file`` refuses a file or a sample region outside it.  That
    docstring states how far the guarantee reaches.
    """
    stamp = now or _utc_now()
    with locked(path):
        store = _load_unlocked(path)
        row = _find(store, uid)
        row.setdefault("exposure", []).append(
            {"dataset_id": str(dataset_id), "region": region, "note": note,
             "recorded_utc": stamp})
        _save_unlocked(path, store)
        return dict(row)
