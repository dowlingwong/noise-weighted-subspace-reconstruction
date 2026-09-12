"""B0: the fit accounting as the runner actually executes it, end to end.

``tests/test_fit_accounting.py`` tests the accounting module on its own.  These
tests drive ``t22_tidmad_narrowband.main`` -- the real stage path, the real
pre-flight, the real ledger append -- and check the properties that only the
*integrated* path can have:

* the reservation exists, and is ``running``, before the first data access;
* the uuid reaches the stage output's provenance and the ledger entry;
* concurrent launches and concurrent completions lose nothing and cannot
  oversubscribe the cap;
* a process death mid-run stays charged and a retry accumulates;
* byte-identical outputs from distinct reservations stay two executions.

Everything is synthetic.  ``--data-root`` points at a directory that does not
exist, every route to instrument data raises, and the estimator is replaced, so
a test that reached real data would fail rather than read it.  No T22 stage is
run, no HDF5 file is opened, and nothing under ``results/`` is touched.

The concurrency tests run on the **execution filesystem** (a fixture directory
under ``/ceph/dwong``, removed afterwards), not on ``/tmp``: ``flock`` is
advisory and can be a no-op on a network mount, and that has to be measured
where the runs actually happen rather than assumed.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.mlst2026 import fit_accounting as fa  # noqa: E402
from experiments.mlst2026 import t22_tidmad_narrowband as t22  # noqa: E402
from test_mlst_t22_narrowband import (  # noqa: E402,F401
    P, _patch_h5py, _sha, _write_chars, _write_freeze)

#: Where a contended-lock test may build its fixture tree.  It must be the
#: filesystem the runs execute on, not the local temporary directory.
CEPH_BASE = Path("/ceph/dwong")

needs_fork = pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="needs fork to spawn real concurrent processes")


# --- fixtures -------------------------------------------------------------------
def _tree(base: Path, P) -> types.SimpleNamespace:
    """A bound run tree: a matching characterisation and freeze, an empty run
    directory, and a ``--data-root`` that does not exist."""
    chars = _write_chars(base / "inputs" / "chars.json", P)
    freeze = _write_freeze(base / "inputs" / "freeze.json", P, chars)
    runs = base / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    return types.SimpleNamespace(
        base=base, chars=chars, freeze=freeze, runs=runs, data=base / "no-data",
        ledger=runs / t22.LEDGER_NAME, store=runs / fa.RESERVATION_NAME)


def _argv(tree, stage: str, out_name: str, *extra) -> list:
    return ["--stage", stage, "--protocol", str(tree.protocol), "--out",
            str(tree.runs / out_name), "--freeze-file", str(tree.freeze),
            "--characterisation", str(tree.chars), "--data-root", str(tree.data), *extra]


def _seed_ledger(tree, fits: int) -> None:
    """One prior entry, so the run tree has a ledger before anything starts.

    Without it the first completion and the next launch race on
    ``require_ledger_beside_outputs``, which is a refusal, not an accounting
    error -- but it is not what these tests are measuring.
    """
    entries = [{"stage": "development", "output_dir": "T22_development_v0",
                "output_file": "t22_development.json", "output_sha256": "0" * 64,
                "fits_executed": int(fits), "smoke": False, "git_commit": "test",
                "recorded_utc": "2026-09-11T00:00:00Z"}]
    ledger = t22._ledger_totals({"schema": t22.LEDGER_SCHEMA, "hard_cap": 250,
                                 "counting_convention": "test", "entries": entries})
    tree.ledger.write_text(json.dumps(ledger, indent=2))


def _evaluator(fits: int, hook=None):
    """A stage_evaluate that opens nothing and reports ``fits`` fits."""
    def fake(args, P_, rec, frozen, files, label, chars):
        if hook is not None:
            hook()
        return {label: {"per_file": {}, "files_evaluated": list(files),
                        "fit_count": int(fits)}}, {}
    return fake


@pytest.fixture
def no_instrument_data(monkeypatch):
    """Every route to instrument data raises."""
    def boom(*a, **k):
        raise AssertionError("data read")
    for name in ("noise_bin_power", "load_stretch", "score_cell"):
        monkeypatch.setattr(t22, name, boom)
    _patch_h5py(monkeypatch, boom)
    return boom


@pytest.fixture
def local(tmp_path, P):
    t = _tree(tmp_path, P)
    t.protocol = P.path
    return t


@pytest.fixture
def on_ceph(P):
    """An isolated fixture tree on the execution filesystem, removed afterwards."""
    if not os.access(CEPH_BASE, os.W_OK):
        pytest.skip(f"{CEPH_BASE} is not writable here")
    base = CEPH_BASE / f".b0_lock_fixture_{os.getpid()}"
    if base.exists():                                 # pragma: no cover - stale fixture
        shutil.rmtree(base)
    base.mkdir(parents=True)
    try:
        t = _tree(base, P)
        t.protocol = P.path
        yield t
    finally:
        shutil.rmtree(base, ignore_errors=True)


# --- 1. the reservation comes before the data ------------------------------------
def test_the_reservation_is_running_before_the_first_data_access(local, no_instrument_data,
                                                                 monkeypatch):
    """The order the reviewer asked for: reserve, then open data.

    ``_science_paths`` is the runner's first touch of the data root (the science
    files are hashed there, before noise_bin_power, load_stretch, stage_evaluate,
    score_cell or any h5py call).  Reading the store from inside it shows what
    was reserved at that instant.
    """
    seen = {}

    def spy(data_root):
        seen["store"] = json.loads(local.store.read_text())
        raise AssertionError("data read")

    monkeypatch.setattr(t22, "_science_paths", spy)
    monkeypatch.setattr(t22, "stage_evaluate", _evaluator(24))
    with pytest.raises(AssertionError, match="data read"):
        t22.main(_argv(local, "validation", "T22_validation_v1"))

    (row,) = seen["store"]["reservations"]
    assert row["state"] == "running" and row["planned"] == 24
    assert row["stage"] == "validation" and row["pid"] == os.getpid()
    assert [h["state"] for h in row["history"]] == ["pending", "running"]
    # and the declared exposure was recorded before the read, not after it
    assert {e["dataset_id"] for e in row["exposure"]} == set(
        json.loads(local.freeze.read_text())["frozen"]["validation_files"])

    # the attempt is now failed, still charged, and nothing was written
    after = fa.load_reservations(local.store)["reservations"][0]
    assert after["state"] == "failed" and "data read" in after["reason"]
    assert fa.charged_total(reservations=fa.load_reservations(local.store)) == 24
    assert not (local.runs / "T22_validation_v1").exists()
    assert not local.ledger.exists()


def test_the_estimator_runs_inside_a_running_reservation(local, no_instrument_data,
                                                         monkeypatch):
    """The state at estimator time is ``running``, not ``pending``."""
    seen = {}
    monkeypatch.setattr(t22, "stage_evaluate", _evaluator(
        24, hook=lambda: seen.update(store=json.loads(local.store.read_text()))))
    assert t22.main(_argv(local, "validation", "T22_validation_v1")) == 0
    (row,) = seen["store"]["reservations"]
    assert row["state"] == "running" and row["actual_fits"] is None


# --- 2. the uuid is the execution's identity everywhere --------------------------
def test_a_completed_stage_carries_its_uuid_into_the_output_and_the_ledger(
        local, no_instrument_data, monkeypatch):
    monkeypatch.setattr(t22, "stage_evaluate", _evaluator(24))
    assert t22.main(_argv(local, "validation", "T22_validation_v1")) == 0

    out = local.runs / "T22_validation_v1" / "t22_validation.json"
    js = json.loads(out.read_text())
    (row,) = fa.load_reservations(local.store)["reservations"]
    assert row["state"] == "completed" and row["actual_fits"] == 24
    assert [h["state"] for h in row["history"]] == ["pending", "running", "completed"]
    assert row["output_sha256"] == _sha(out) and row["output_ref"] == str(out)
    assert fa.uuid_in_output(js) == row["uuid"]

    ledger = json.loads(local.ledger.read_text())
    (entry,) = ledger["entries"]
    assert entry["reservation_uuid"] == row["uuid"] and entry["fits_executed"] == 24

    # one execution, seen by all three sources, counted once
    u = fa.union_executed(ledger, t22.scan_executed_fits([local.runs]),
                          ledger_path=local.ledger,
                          reservations=fa.load_reservations(local.store))
    assert u["executed"] == 24 and u["charged"] == 24
    assert len(u["records"]) == 1 and u["records"][0]["records_merged"] == 3
    assert u["records"][0]["identity"] == "reservation"


# --- 3. the union as the runner computes it --------------------------------------
def _budget(tree, planned=24, stage="validation"):
    return t22.check_cumulative_budget(t22.Protocol(tree.protocol), tree.ledger, planned,
                                       stage, scan_dirs=[tree.runs])


def test_overlapping_disjoint_duplicate_and_moved_records_through_the_runner(
        local, no_instrument_data, monkeypatch):
    """The four record shapes, counted by ``check_cumulative_budget`` itself."""
    monkeypatch.setattr(t22, "stage_evaluate", _evaluator(24))
    assert t22.main(_argv(local, "validation", "T22_validation_v1")) == 0

    # (a) overlapping: the ledger entry and the output on disk are one execution
    assert _budget(local)["cumulative_executed_before"] == 24

    # (b) moved: the same execution under a renamed directory is still one
    (local.runs / "T22_validation_v1").rename(local.runs / "T22_validation_moved")
    assert _budget(local)["cumulative_executed_before"] == 24

    # (c) duplicate: the same entry recorded twice is still one
    ledger = json.loads(local.ledger.read_text())
    ledger["entries"].append(dict(ledger["entries"][0]))
    t22._ledger_totals(ledger)
    local.ledger.write_text(json.dumps(ledger, indent=2))
    assert json.loads(local.ledger.read_text())["executed_total"] == 48   # the ledger sums
    assert _budget(local)["union"]["executed"] == 24                     # the union does not
    assert _budget(local)["union"]["duplicates_merged"] >= 1

    # (d) disjoint: one execution the ledger alone still knows about (its output
    # has been moved out of the run tree) and a different one present only on
    # disk.  max(ledger, scan) takes the larger of 24 and 24 and leaves room the
    # cap does not have; the union sums them to 48.
    ledger["entries"] = ledger["entries"][:1]
    t22._ledger_totals(ledger)
    local.ledger.write_text(json.dumps(ledger, indent=2))
    assert t22.main(_argv(local, "validation", "T22_validation_v2")) == 0
    full = json.loads(local.ledger.read_text())
    full["entries"] = [e for e in full["entries"] if e["output_dir"] != "T22_validation_v2"]
    local.ledger.write_text(json.dumps(t22._ledger_totals(full), indent=2))
    shutil.rmtree(local.runs / "T22_validation_moved")

    b = _budget(local)
    assert b["ledger_executed_total"] == 24 and b["archive_scan_executed_total"] == 24
    assert b["legacy_max_floor"] == 24                   # what max() would have spent
    assert b["union"]["executed"] == 48 == b["cumulative_executed_before"]
    u = fa.union_executed(t22.load_ledger(local.ledger, t22.Protocol(local.protocol)),
                          t22.scan_executed_fits([local.runs]), ledger_path=local.ledger,
                          reservations=fa.load_reservations(local.store))
    assert sorted(tuple(r["sources"]) for r in u["records"]) == [
        ("ledger", "reservation"), ("reservation", "scan")]


def test_byte_identical_outputs_from_distinct_reservations_stay_two_executions(
        local, no_instrument_data, monkeypatch):
    """Two executions that produced the same bytes must not collapse into one."""
    # (a) in the integrated path the uuid is *in* the output, so two runs of the
    # same payload are distinguishable on disk as well as in the store
    monkeypatch.setattr(t22, "stage_evaluate", _evaluator(24))
    assert t22.main(_argv(local, "validation", "T22_validation_v1")) == 0
    assert t22.main(_argv(local, "validation", "T22_validation_v2")) == 0
    outs = sorted(local.runs.glob("T22_validation_v*/t22_validation.json"))
    assert _sha(outs[0]) != _sha(outs[1])
    assert {json.loads(o.read_text())["validation"]["fit_count"] for o in outs} == {24}
    assert _budget(local)["union"]["executed"] == 48

    # (b) the pathological case the identity rule exists for: two reservations,
    # one byte pattern, nothing in the bytes to tell them apart.  Merging on the
    # digest would have charged one execution for two.
    empty = local.base / "twins"
    empty.mkdir()
    store, ledger_path = empty / fa.RESERVATION_NAME, empty / t22.LEDGER_NAME
    a = fa.reserve(store, 250, 24, stage="validation")
    b = fa.reserve(store, 250, 24, stage="validation")
    payload = {"stage": "validation", "smoke": False, "validation": {"fit_count": 24}}
    twins = []
    for name in ("T22_validation_t1", "T22_validation_t2"):
        d = empty / name
        d.mkdir()
        q = d / "t22_validation.json"
        q.write_text(json.dumps(payload, indent=2, sort_keys=True))
        twins.append(q)
    digest = _sha(twins[0])
    assert digest == _sha(twins[1])
    for res, q in zip((a, b), twins):
        fa.mark_completed(store, res["uuid"], 24, q, output_sha256=digest)
    entries = [{"stage": "validation", "output_dir": q.parent.name,
                "output_file": q.name, "output_sha256": digest, "fits_executed": 24,
                "smoke": False, "reservation_uuid": res["uuid"],
                "recorded_utc": "2026-09-12T00:00:00Z"}
               for res, q in zip((a, b), twins)]
    ledger_path.write_text(json.dumps(t22._ledger_totals(
        {"schema": t22.LEDGER_SCHEMA, "hard_cap": 250, "counting_convention": "test",
         "entries": entries}), indent=2))

    Pp = t22.Protocol(local.protocol)
    u = fa.union_executed(t22.load_ledger(ledger_path, Pp), None, ledger_path=ledger_path,
                          reservations=fa.load_reservations(store))
    assert u["executed"] == 48 and len(u["records"]) == 2
    assert {r["identity"] for r in u["records"]} == {"reservation"}
    assert sorted(r["execution_id"] for r in u["records"]) == sorted(
        [f"r:{a['uuid']}", f"r:{b['uuid']}"])

    # (c) and a bare record of those same bytes that names no reservation cannot
    # be attributed to either.  It is counted on its own -- 72 for what is really
    # 48 -- which over-charges the cap rather than hiding an execution from it.
    full = t22.check_cumulative_budget(Pp, ledger_path, 0, "validation", scan_dirs=[empty])
    assert full["archive_scan_executed_total"] == 48
    assert full["union"]["executed"] == 72
    assert full["cumulative_executed_before"] == 72


# --- 4. concurrency, on the filesystem the runs execute on ------------------------
def _child(argv, fits, out_path, barrier):
    """A forked runner: contend, run one stage, record what happened."""
    try:
        t22.stage_evaluate = _evaluator(fits)
        barrier.wait(timeout=120)
        code = t22.main(argv)
        result = {"ok": code == 0}
    except SystemExit as exc:
        result = {"ok": False, "error": str(exc)}
    except BaseException as exc:                        # noqa: BLE001 - reported
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    Path(out_path).write_text(json.dumps(result))
    os._exit(0)


def _launch(tree, n, fits=24):
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(n)
    reports = tree.base / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    procs = [ctx.Process(target=_child,
                         args=(_argv(tree, "validation", f"T22_validation_c{i}"), fits,
                               str(reports / f"{i}.json"), barrier))
             for i in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
        assert p.exitcode == 0, f"worker exited {p.exitcode}"
    return [json.loads((reports / f"{i}.json").read_text()) for i in range(n)]


@needs_fork
def test_flock_actually_excludes_on_the_execution_filesystem(on_ceph):
    """Does ``flock`` hold on CephFS?  Measured, not assumed.

    Contending processes each do locked read-modify-write increments of a
    counter; a lost update means the lock did not exclude and every guarantee
    built on it is void for this mount.
    """
    probe = fa.lock_is_effective(on_ceph.base / "lockprobe", workers=6, rounds=25)
    assert probe["effective"], (
        f"flock did NOT exclude on {on_ceph.base}: {probe['observed']} of "
        f"{probe['expected']} increments survived.  The reservation store and the "
        f"fit ledger are not safe against concurrent writers on this filesystem; "
        f"serialise the runs by hand.")


@needs_fork
def test_concurrent_completions_lose_no_ledger_entry(on_ceph, no_instrument_data):
    """Eight stages completing at once: eight reservations, eight ledger entries.

    The ledger append is a read-modify-write.  Unlocked, the completions
    interleave and the last writer wins, so entries -- and the fits they record
    -- silently disappear from the cumulative total.
    """
    _seed_ledger(on_ceph, 8)
    results = _launch(on_ceph, 8, fits=24)
    assert all(r["ok"] for r in results), [r for r in results if not r["ok"]]

    ledger = json.loads(on_ceph.ledger.read_text())
    assert len(ledger["entries"]) == 9                 # the seed plus eight stages
    assert ledger["executed_total"] == 8 + 8 * 24 == 200
    rows = fa.load_reservations(on_ceph.store)["reservations"]
    assert len(rows) == 8 and {r["state"] for r in rows} == {"completed"}
    assert len({r["uuid"] for r in rows}) == 8         # nothing lost to a lost update
    assert len({e["reservation_uuid"] for e in ledger["entries"]
                if "reservation_uuid" in e}) == 8

    u = fa.union_executed(ledger, t22.scan_executed_fits([on_ceph.runs]),
                          ledger_path=on_ceph.ledger,
                          reservations=fa.load_reservations(on_ceph.store))
    assert u["executed"] == 200 and u["charged"] == 200 <= 250


def _append_worker(ledger_path, protocol, worker, rounds, barrier, out):  # pragma: no cover
    """Hammer the ledger append from a real process."""
    try:
        P = t22.Protocol(Path(protocol))
        barrier.wait(timeout=120)
        for seq in range(rounds):
            t22.record_in_ledger(Path(ledger_path), P, {
                "stage": "development", "output_dir": f"T22_w{worker}_s{seq}",
                "output_file": "t22_development.json", "output_sha256": f"{worker:02d}{seq:02d}"
                + "0" * 60, "fits_executed": 1, "smoke": False,
                "worker": worker, "seq": seq, "recorded_utc": "2026-09-12T00:00:00Z"})
        Path(out).write_text(json.dumps({"ok": True}))
    except BaseException as exc:                        # noqa: BLE001 - reported
        Path(out).write_text(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
    os._exit(0)


@needs_fork
def test_concurrent_ledger_appends_lose_nothing(on_ceph):
    """The append is a read-modify-write; contended, it must still lose nothing.

    Eight processes append twenty entries each to one ledger on the execution
    filesystem.  Every entry must survive: an entry dropped here is fits that
    were executed and are no longer charged against the cap.  With the lock
    removed this test fails -- entries go missing -- which is what makes it a
    check of the lock rather than of the weather.
    """
    workers, rounds = 8, 20
    _seed_ledger(on_ceph, 0)
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(workers)
    reports = on_ceph.base / "append-reports"
    reports.mkdir(parents=True, exist_ok=True)
    procs = [ctx.Process(target=_append_worker,
                         args=(str(on_ceph.ledger), str(on_ceph.protocol), w, rounds,
                               barrier, str(reports / f"{w}.json")))
             for w in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
        assert p.exitcode == 0, f"worker exited {p.exitcode}"
    reported = [json.loads((reports / f"{w}.json").read_text()) for w in range(workers)]
    assert all(r["ok"] for r in reported), [r for r in reported if not r["ok"]]

    ledger = json.loads(on_ceph.ledger.read_text())
    appended = {(e["worker"], e["seq"]) for e in ledger["entries"] if "worker" in e}
    missing = {(w, s) for w in range(workers) for s in range(rounds)} - appended
    assert not missing, f"{len(missing)} ledger entries lost to the race: {sorted(missing)[:8]}"
    assert len(ledger["entries"]) == workers * rounds + 1
    assert ledger["executed_total"] == workers * rounds


@needs_fork
def test_concurrent_launches_cannot_oversubscribe_the_cap(on_ceph, no_instrument_data):
    """Room for two, six launches: exactly two get it, and the cap is not crossed.

    The unlocked pre-flight can pass for all six at once; the reservation, taken
    under the store's lock against the union plus every open charge, is what
    actually decides.
    """
    _seed_ledger(on_ceph, 202)                         # 202 + 24 + 24 == 250
    results = _launch(on_ceph, 6, fits=24)
    ok = [r for r in results if r["ok"]]
    refused = [r for r in results if not r["ok"]]
    assert len(ok) == 2, results
    assert len(refused) == 4
    assert all("hard cap 250" in r["error"] for r in refused), refused

    ledger = json.loads(on_ceph.ledger.read_text())
    assert ledger["executed_total"] == 250 and not ledger["executed_total_exceeds_cap"]
    rows = fa.load_reservations(on_ceph.store)["reservations"]
    assert len(rows) == 2 and {r["state"] for r in rows} == {"completed"}
    u = fa.union_executed(ledger, t22.scan_executed_fits([on_ceph.runs]),
                          ledger_path=on_ceph.ledger,
                          reservations=fa.load_reservations(on_ceph.store))
    assert u["charged"] == 250 <= 250
    # and a seventh launch is refused on the spent total
    with pytest.raises(SystemExit, match="hard cap 250"):
        t22.main(_argv(on_ceph, "validation", "T22_validation_after"))


# --- 5. a process that dies mid-run ------------------------------------------------
def _suicide_child(argv):                               # pragma: no cover - child
    """Start a stage and die inside it without unwinding anything."""
    t22.stage_evaluate = _evaluator(24, hook=lambda: os.kill(os.getpid(), 9))
    t22.main(argv)


@needs_fork
def test_a_process_death_mid_run_stays_charged_and_a_retry_accumulates(
        local, no_instrument_data, monkeypatch):
    _seed_ledger(local, 0)
    ctx = multiprocessing.get_context("fork")
    p = ctx.Process(target=_suicide_child,
                    args=(_argv(local, "validation", "T22_validation_dead"),))
    p.start()
    p.join(timeout=120)
    assert p.exitcode == -9, f"expected SIGKILL, got {p.exitcode}"

    # it died with no handler run: the reservation is still 'running' and charged
    (row,) = fa.load_reservations(local.store)["reservations"]
    assert row["state"] == "running" and row["planned"] == 24
    assert fa.charged_total(reservations=fa.load_reservations(local.store)) == 24
    assert not (local.runs / "T22_validation_dead").exists()
    assert json.loads(local.ledger.read_text())["entries"] == json.loads(
        local.ledger.read_text())["entries"][:1]

    # the budget charges it before the sweep even runs
    assert _budget(local)["union"]["reserved_open_fits"] == 24

    # the retry sweeps it to interrupted, keeps the charge, and adds its own
    monkeypatch.setattr(t22, "stage_evaluate", _evaluator(24))
    assert t22.main(_argv(local, "validation", "T22_validation_retry")) == 0
    rows = {r["state"]: r for r in fa.load_reservations(local.store)["reservations"]}
    assert set(rows) == {"interrupted", "completed"}
    assert rows["interrupted"]["uuid"] == row["uuid"]
    assert rows["interrupted"]["reason"] == "owning process gone"
    assert _budget(local)["cumulative_executed_before"] == 48     # 24 lost + 24 executed
    assert _budget(local)["union"]["executed"] == 24
    assert _budget(local)["union"]["reserved_open_fits"] == 24


@needs_fork
def test_a_retry_never_resets_the_expenditure(local, no_instrument_data, monkeypatch):
    """Three failed attempts charge three times, however they failed."""
    _seed_ledger(local, 0)
    monkeypatch.setattr(t22, "stage_evaluate", _evaluator(24, hook=lambda: (_ for _ in ()).throw(
        RuntimeError("estimator blew up"))))
    for i in range(3):
        with pytest.raises(RuntimeError, match="estimator blew up"):
            t22.main(_argv(local, "validation", f"T22_validation_try{i}"))
    rows = fa.load_reservations(local.store)["reservations"]
    assert len(rows) == 3 and {r["state"] for r in rows} == {"failed"}
    assert fa.charged_total(reservations=fa.load_reservations(local.store)) == 72
    assert _budget(local)["cumulative_executed_before"] == 72
