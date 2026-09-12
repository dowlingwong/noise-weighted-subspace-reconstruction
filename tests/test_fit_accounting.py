"""Execution accounting: stable identity, the union, and the reservation ledger.

Every fixture here is synthetic and lives under ``tmp_path``, except the one test
that reads the archived T22 run tree -- read-only -- to pin the union against the
disclosed counts (416 executed, 240 distinct, hard cap 250).  No T22 stage is run
and no instrument file is opened.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.mlst2026 import fit_accounting as fa  # noqa: E402
from experiments.mlst2026 import t22_tidmad_narrowband as t22  # noqa: E402

#: The archived run whose counts are cited by the manuscript and by the
#: enforcement verifier.  Read only; never written by these tests.
ARCHIVE = ROOT / "results" / "t22_2026-09-11"

PROTOCOL_CANDIDATES = [Path(p) for p in (os.environ.get("T22_PROTOCOL", ""),) if p] + [
    ROOT.parent / "arXiv-v1-polished-extended-preprint"
    / "plan" / "mlst_protocol_2026-09-11-t22-stage0.json"]


@pytest.fixture(scope="module")
def P():
    for p in PROTOCOL_CANDIDATES:
        if p.is_file():
            return t22.Protocol(p)
    pytest.skip("T22 stage-0 protocol JSON not found (set $T22_PROTOCOL)")


# --- fixtures -------------------------------------------------------------------
def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _payload(stage="development", *, fits=176, finished="2026-09-11T16:15:13Z", tag="a",
             reservation=None):
    """A stage output with the shape ``fits_in_output`` reads."""
    js = {"stage": stage, "smoke": False, "tag": tag,
          "provenance": {"finished_utc": finished}}
    if stage == "development":
        js["fit_count"] = fits
    else:
        js[stage] = {"fit_count": fits}
    if reservation is not None:
        js["provenance"][fa.PROVENANCE_KEY] = fa.reservation_provenance(reservation)
    return js


def _write_output(directory: Path, name: str, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / name
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return p


def _entry(stage, out_dir, out_file, fits, digest, **kw):
    entry = {"stage": stage, "output_dir": out_dir, "output_file": out_file,
             "fits_executed": int(fits), "output_sha256": digest, "smoke": False,
             "output_finished_utc": kw.pop("finished", "2026-09-11T16:00:00Z"),
             "recorded_utc": "2026-09-11T20:59:13Z"}
    entry.update(kw)
    return entry


def _ledger(entries, cap=250):
    """A ledger whose stored totals are the honest sum of its entries."""
    real = [e for e in entries if not e.get("smoke")]
    return {"schema": t22.LEDGER_SCHEMA, "hard_cap": cap,
            "counting_convention": "one estimator on one (stretch, band) cell",
            "entries": list(entries),
            "executed_total": sum(e["fits_executed"] for e in real),
            "unique_scientific_total": sum(e["fits_executed"] for e in real
                                           if not e.get("deterministic_rescore_of"))}


def _naive_max(ledger, scan) -> int:
    """What ``check_cumulative_budget`` used to spend."""
    return max(ledger["executed_total"] if ledger else 0, scan["executed_total"])


# --- the union ------------------------------------------------------------------
def test_an_overlapping_record_is_counted_once(tmp_path):
    p = _write_output(tmp_path / "T22_development_v1", "t22_development.json", _payload())
    scan = t22.scan_executed_fits([tmp_path])
    ledger = _ledger([_entry("development", "T22_development_v1", "t22_development.json",
                             176, _sha(p))])
    u = fa.union_executed(ledger, scan, ledger_path=tmp_path / t22.LEDGER_NAME)
    assert u["executed"] == 176 and len(u["records"]) == 1
    assert u["records"][0]["sources"] == ["ledger", "scan"]
    assert u["records"][0]["identity"] == "digest"
    assert u["duplicates_merged"] == 1 and u["seen_by_both"] == 1
    assert u["charged"] == 176 == _naive_max(ledger, scan)


def test_disjoint_records_sum_where_max_understated_them(tmp_path):
    """The case ``max()`` got wrong: one execution only in the ledger, a different
    one only on disk."""
    _write_output(tmp_path / "T22_validation_v1", "t22_validation.json",
                  _payload("validation", fits=60, finished="2026-09-11T16:24:15Z"))
    scan = t22.scan_executed_fits([tmp_path])
    # A development run whose output is no longer on disk, recorded by digest.
    gone = hashlib.sha256(b"executed then archived elsewhere").hexdigest()
    ledger = _ledger([_entry("development", "T22_development_v1", "t22_development.json",
                             176, gone)])
    u = fa.union_executed(ledger, scan, ledger_path=tmp_path / t22.LEDGER_NAME)
    assert u["executed"] == 236
    assert _naive_max(ledger, scan) == 176 < u["executed"]
    assert u["ledger_only"] == 1 and u["scan_only"] == 1 and u["duplicates_merged"] == 0


def test_the_same_output_recorded_twice_in_the_ledger_is_one_execution():
    digest = hashlib.sha256(b"one output").hexdigest()
    ledger = _ledger([
        _entry("development", "T22_development_v1", "t22_development.json", 176, digest),
        _entry("development", "T22_development_v1_copy", "t22_development.json", 176, digest)])
    assert ledger["executed_total"] == 352            # the ledger's own naive sum
    u = fa.union_executed(ledger, None)
    assert u["executed"] == 176 and len(u["records"]) == 1
    assert u["records"][0]["records_merged"] == 2 and u["duplicates_merged"] == 1


def test_a_moved_output_is_still_one_execution(tmp_path):
    payload = _payload()
    p = _write_output(tmp_path / "T22_development_moved", "t22_development.json", payload)
    scan = t22.scan_executed_fits([tmp_path])
    ledger = _ledger([_entry("development", "T22_development_v1", "t22_development.json",
                             176, _sha(p))])      # the ledger still names the old directory
    u = fa.union_executed(ledger, scan, ledger_path=tmp_path / t22.LEDGER_NAME)
    assert u["executed"] == 176 and len(u["records"]) == 1
    assert u["records"][0]["sources"] == ["ledger", "scan"]


def test_records_sharing_only_a_placeholder_digest_are_not_merged():
    """A digest identifies an execution only together with its stage, output file
    and smoke flag -- so two entries carrying the same placeholder stay apart."""
    zero = "0" * 64
    ledger = _ledger([
        _entry("development", "T22_development_v0", "t22_development.json", 176, zero),
        _entry("validation", "T22_validation_v1", "t22_validation.json", 54, zero),
        _entry("development", "T22_development_smoke", "t22_development.json", 500, zero,
               smoke=True)])
    u = fa.union_executed(ledger, None)
    assert u["executed"] == 230 and u["smoke_executed"] == 500
    assert len(u["records"]) == 3 and u["duplicates_merged"] == 0


def test_inference_only_recomputation_is_charged_but_is_not_new_science():
    ledger = _ledger([
        _entry("development", "T22_development_v1", "t22_development.json", 176,
               hashlib.sha256(b"v1").hexdigest(), finished="2026-09-11T16:15:13Z"),
        _entry("development", "T22_development_v2", "t22_development.json", 176,
               hashlib.sha256(b"v2").hexdigest(), finished="2026-09-11T16:20:22Z",
               deterministic_rescore_of="T22_development_v1/t22_development.json")])
    u = fa.union_executed(ledger, None)
    assert u["executed"] == 352                       # both executions cost compute
    assert u["inference_only_fits"] == 176
    assert u["unique_scientific"] == 176              # the re-score adds no evaluation
    assert u["fitted_models"] == 352                  # but it did instantiate models
    assert u["scientific_configurations"] == 1 and u["restart_fits"] == 0


def test_a_restart_is_a_fitted_model_but_not_a_new_scientific_configuration():
    ledger = _ledger([
        _entry("development", "T22_development_v1", "t22_development.json", 176,
               hashlib.sha256(b"r1").hexdigest(), finished="2026-09-11T16:15:13Z",
               config_id="dev/band-grid"),
        _entry("development", "T22_development_v2", "t22_development.json", 176,
               hashlib.sha256(b"r2").hexdigest(), finished="2026-09-11T16:40:00Z",
               config_id="dev/band-grid"),        # same configuration, re-fitted
        _entry("validation", "T22_validation_v1", "t22_validation.json", 24,
               hashlib.sha256(b"r3").hexdigest(), finished="2026-09-11T17:00:00Z",
               config_id="holdout/validation")])
    u = fa.union_executed(ledger, None)
    assert u["executed"] == 376
    assert u["unique_scientific"] == 376              # no re-score: all of it is science
    assert u["restart_fits"] == 176                   # of which a restart of one config
    assert u["fitted_models"] == 376
    assert u["scientific_configurations"] == 2
    assert u["configuration_ids"] == ["dev/band-grid", "holdout/validation"]


def test_a_record_that_refits_nothing_separates_fitted_models_from_executed():
    ledger = _ledger([
        _entry("development", "T22_development_v1", "t22_development.json", 176,
               hashlib.sha256(b"f1").hexdigest()),
        _entry("development", "T22_development_v2", "t22_development.json", 24,
               hashlib.sha256(b"f2").hexdigest(), finished="2026-09-11T17:00:00Z",
               refits_models=False)])
    u = fa.union_executed(ledger, None)
    assert u["executed"] == 200 and u["fitted_models"] == 176


def test_byte_identical_outputs_from_distinct_reservations_stay_distinct(tmp_path):
    """Two executions that produced the same bytes are still two executions."""
    store = tmp_path / fa.RESERVATION_NAME
    a = fa.reserve(store, 250, 24, stage="validation")
    b = fa.reserve(store, 250, 24, stage="validation")
    assert a["uuid"] != b["uuid"]
    pa = _write_output(tmp_path / "T22_validation_v1", "t22_validation.json",
                       _payload("validation", fits=24))
    pb = _write_output(tmp_path / "T22_validation_v2", "t22_validation.json",
                       _payload("validation", fits=24))
    assert _sha(pa) == _sha(pb)                   # identical bytes, different runs
    fa.mark_completed(store, a["uuid"], 24, pa, output_sha256=_sha(pa))
    fa.mark_completed(store, b["uuid"], 24, pb, output_sha256=_sha(pb))

    u = fa.union_executed(reservations=fa.load_reservations(store))
    assert u["executed"] == 48 and len(u["records"]) == 2
    assert sorted(r["execution_id"] for r in u["records"]) == sorted(
        [f"r:{a['uuid']}", f"r:{b['uuid']}"])
    assert all(r["identity"] == "reservation" for r in u["records"])

    # the same through the ledger: two entries, same digest, different uuids
    ledger = _ledger([
        _entry("validation", "T22_validation_v1", "t22_validation.json", 24, _sha(pa),
               reservation_uuid=a["uuid"]),
        _entry("validation", "T22_validation_v2", "t22_validation.json", 24, _sha(pb),
               reservation_uuid=b["uuid"])])
    assert fa.union_executed(ledger, None)["executed"] == 48


def test_an_unclaimed_record_of_contested_bytes_is_counted_on_its_own(tmp_path):
    """A digest two reservations both claim cannot attribute a third record.

    The conservative direction for a cap is to count it separately rather than
    merge it into whichever execution happens to be listed first.
    """
    store = tmp_path / fa.RESERVATION_NAME
    a = fa.reserve(store, 250, 24, stage="validation")
    b = fa.reserve(store, 250, 24, stage="validation")
    pa = _write_output(tmp_path / "T22_validation_v1", "t22_validation.json",
                       _payload("validation", fits=24))
    pb = _write_output(tmp_path / "T22_validation_v2", "t22_validation.json",
                       _payload("validation", fits=24))
    fa.mark_completed(store, a["uuid"], 24, pa, output_sha256=_sha(pa))
    fa.mark_completed(store, b["uuid"], 24, pb, output_sha256=_sha(pb))
    # a ledger entry of those same bytes that names no reservation
    ledger = _ledger([_entry("validation", "T22_validation_v3", "t22_validation.json",
                             24, _sha(pa))])
    u = fa.union_executed(ledger, None, reservations=fa.load_reservations(store))
    assert len(u["records"]) == 3 and u["executed"] == 72
    assert u["duplicates_merged"] == 0


def test_a_smoke_reservation_is_recorded_and_charges_nothing(tmp_path):
    store = tmp_path / fa.RESERVATION_NAME
    r = fa.reserve(store, 250, 500, stage="development", smoke=True, spent=200)
    assert r["planned"] == 500 and r["smoke"] is True
    # exempt from the cap, like a smoke ledger entry: recorded, totalled apart
    assert fa.charged_total(reservations=fa.load_reservations(store)) == 0
    with pytest.raises(SystemExit):                # a real run still cannot
        fa.reserve(store, 250, 500, stage="development", spent=200)


def test_execution_identity_kinds_and_what_they_key_on():
    res, kind = fa.execution_id(reservation_uuid="u-1", output_sha256="ab", stage="development",
                                output_file="t22_development.json", output_dir="d")
    assert kind == "reservation" and res == "r:u-1"
    a, ka = fa.execution_id(output_sha256="ab", stage="development",
                            output_file="t22_development.json", output_dir="v1")
    b, _ = fa.execution_id(output_sha256="ab", stage="development",
                           output_file="t22_development.json", output_dir="v2_renamed")
    assert ka == "digest" and a == b              # the directory is not part of the key
    c, _ = fa.execution_id(output_sha256="ab", stage="validation",
                           output_file="t22_validation.json", output_dir="v1")
    assert c != a                                 # the stage and file name are
    loc, kl = fa.execution_id(stage="development", output_file="t22_development.json",
                              output_dir="v1")
    assert kl == "location" and loc.startswith("l:")


# --- refusals -------------------------------------------------------------------
def test_a_missing_fit_ledger_is_refused(tmp_path):
    with pytest.raises(SystemExit) as exc:
        fa.require_fit_ledger(tmp_path / t22.LEDGER_NAME)
    assert "does not exist" in str(exc.value) and "audit-budget" in str(exc.value)


def test_a_stale_fit_ledger_is_refused(tmp_path):
    ledger = _ledger([_entry("development", "T22_development_v1", "t22_development.json",
                             176, hashlib.sha256(b"s").hexdigest())])
    ledger["executed_total"] = 24                     # hand-edited downwards
    path = tmp_path / t22.LEDGER_NAME
    path.write_text(json.dumps(ledger, indent=2))
    with pytest.raises(SystemExit) as exc:
        fa.require_fit_ledger(path)
    assert "disagree" in str(exc.value) and "refusing to trust it" in str(exc.value)
    with pytest.raises(SystemExit):
        fa.union_executed(ledger, None)


def test_a_stale_reservation_store_is_refused(tmp_path):
    store = tmp_path / fa.RESERVATION_NAME
    fa.reserve(store, 250, 24, {"why": "test"}, stage="validation")
    raw = json.loads(store.read_text())
    raw["charged_open_total"] = 0                     # hand-edited downwards
    store.write_text(json.dumps(raw, indent=2))
    with pytest.raises(SystemExit) as exc:
        fa.load_reservations(store)
    assert "charged_open_total" in str(exc.value) and "refusing to trust it" in str(exc.value)


# --- reservations ---------------------------------------------------------------
def test_a_reservation_is_charged_before_the_output_exists(tmp_path):
    store = tmp_path / fa.RESERVATION_NAME
    r = fa.reserve(store, 250, 24, {"stage": "validation"}, stage="validation")
    assert r["state"] == "pending" and r["pid"] == os.getpid()
    loaded = fa.load_reservations(store)
    u = fa.union_executed(None, None, reservations=loaded)
    assert u["executed"] == 0 and u["reserved_open_fits"] == 24 and u["charged"] == 24
    fa.mark_running(store, r["uuid"])
    assert fa.charged_total(reservations=fa.load_reservations(store)) == 24


def test_a_reservation_is_refused_rather_than_crossing_the_cap(tmp_path):
    store = tmp_path / fa.RESERVATION_NAME
    fa.reserve(store, 250, 200, stage="development")
    with pytest.raises(SystemExit) as exc:
        fa.reserve(store, 250, 60, stage="validation")
    assert "would exceed the hard cap 250" in str(exc.value)
    assert len(fa.load_reservations(store)["reservations"]) == 1
    # completed expenditure counted elsewhere is passed in and also refuses
    with pytest.raises(SystemExit):
        fa.reserve(store, 250, 24, stage="validation", spent=40)


def test_a_failed_attempt_is_charged_and_a_retry_adds_to_it(tmp_path):
    store = tmp_path / fa.RESERVATION_NAME
    first = fa.reserve(store, 250, 24, stage="validation")
    fa.mark_running(store, first["uuid"])
    fa.mark_failed(store, first["uuid"], "estimator raised")
    after = fa.load_reservations(store)
    assert after["reservations"][0]["state"] == "failed"
    assert fa.charged_total(reservations=after) == 24     # not refunded
    second = fa.reserve(store, 250, 24, stage="validation", spent=0)
    assert second["charged_before"] == 24                 # the retry starts from 24
    assert fa.charged_total(reservations=fa.load_reservations(store)) == 48


def test_a_completed_reservation_does_not_double_count_with_its_output(tmp_path):
    store = tmp_path / fa.RESERVATION_NAME
    r = fa.reserve(store, 250, 176, stage="development")
    fa.mark_running(store, r["uuid"])
    out = _write_output(tmp_path / "T22_development_v1", "t22_development.json",
                        _payload(reservation=r))
    fa.mark_completed(store, r["uuid"], 176, out, output_sha256=_sha(out))
    scan = t22.scan_executed_fits([tmp_path])
    ledger = _ledger([_entry("development", "T22_development_v1", "t22_development.json",
                             176, _sha(out), reservation_uuid=r["uuid"])])
    u = fa.union_executed(ledger, scan, ledger_path=tmp_path / t22.LEDGER_NAME,
                          reservations=fa.load_reservations(store))
    assert u["executed"] == 176 and u["charged"] == 176
    assert len(u["records"]) == 1 and u["records"][0]["records_merged"] == 3
    assert u["records"][0]["sources"] == ["ledger", "reservation", "scan"]
    assert u["records"][0]["identity"] == "reservation"


def test_exposure_records_are_kept_and_change_no_charge(tmp_path):
    store = tmp_path / fa.RESERVATION_NAME
    r = fa.reserve(store, 250, 24, stage="validation")
    fa.attach_exposure(store, r["uuid"], dataset_id="abra_validation_0010.h5",
                       region={"start_sample": 0, "n_samples": 1024}, note="planned")
    row = fa.load_reservations(store)["reservations"][0]
    assert row["exposure"][0]["dataset_id"] == "abra_validation_0010.h5"
    assert row["exposure"][0]["region"]["n_samples"] == 1024
    assert fa.charged_total(reservations=fa.load_reservations(store)) == 24


# --- concurrency ----------------------------------------------------------------
def _reserve_worker(store, cap, planned, barrier, out_dir, idx):
    """Child process: contend for the store, then record what happened."""
    try:
        barrier.wait(timeout=60)
        r = fa.reserve(store, cap, planned, {"worker": idx}, stage="development")
        result = {"ok": True, "uuid": r["uuid"], "before": r["charged_before"]}
    except SystemExit as exc:
        result = {"ok": False, "error": str(exc)}
    except BaseException as exc:                      # noqa: BLE001 - reported, not raised
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    Path(out_dir, f"{idx}.json").write_text(json.dumps(result))


def _run_workers(n, store, cap, planned, out_dir):
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(n)
    out_dir.mkdir(parents=True, exist_ok=True)
    procs = [ctx.Process(target=_reserve_worker,
                         args=(str(store), cap, planned, barrier, str(out_dir), i))
             for i in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"worker exited {p.exitcode}"
    return [json.loads(Path(out_dir, f"{i}.json").read_text()) for i in range(n)]


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(),
                    reason="needs fork to spawn real concurrent processes")
def test_concurrent_reservations_all_survive_when_there_is_room(tmp_path):
    store = tmp_path / fa.RESERVATION_NAME
    results = _run_workers(8, store, 1000, 10, tmp_path / "out")
    assert all(r["ok"] for r in results), [r for r in results if not r["ok"]]
    uuids = [r["uuid"] for r in results]
    assert len(set(uuids)) == 8                       # nothing lost to a lost update
    rows = fa.load_reservations(store)["reservations"]
    assert sorted(x["uuid"] for x in rows) == sorted(uuids)
    assert fa.charged_total(reservations=fa.load_reservations(store)) == 80
    # the reservations serialised: every charged_before is distinct and stepped
    assert sorted(r["before"] for r in results) == [0, 10, 20, 30, 40, 50, 60, 70]


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(),
                    reason="needs fork to spawn real concurrent processes")
def test_concurrent_reservations_never_oversubscribe_the_cap(tmp_path):
    store = tmp_path / fa.RESERVATION_NAME
    cap, planned, n = 50, 10, 8
    results = _run_workers(n, store, cap, planned, tmp_path / "out")
    ok = [r for r in results if r["ok"]]
    refused = [r for r in results if not r["ok"]]
    assert len(ok) == 5 and len(refused) == 3
    assert all("would exceed the hard cap 50" in r["error"] for r in refused)
    rows = fa.load_reservations(store)["reservations"]
    assert len(rows) == 5 and len({x["uuid"] for x in rows}) == 5
    charged = fa.charged_total(reservations=fa.load_reservations(store))
    assert charged == 50 <= cap                       # exactly right, never over


# --- crash and restart ------------------------------------------------------------
def _crash_worker(store, cap, planned, out):
    """Reserve, start running, then die without completing or failing."""
    r = fa.reserve(store, cap, planned, {"worker": "crash"}, stage="validation")
    fa.mark_running(store, r["uuid"])
    Path(out).write_text(json.dumps({"uuid": r["uuid"], "pid": os.getpid()}))
    os._exit(0)                                       # no unwinding, no cleanup


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(),
                    reason="needs fork to kill a real owning process")
def test_a_dead_running_reservation_is_swept_and_still_charged(tmp_path):
    store = tmp_path / fa.RESERVATION_NAME
    note = tmp_path / "crashed.json"
    ctx = multiprocessing.get_context("fork")
    p = ctx.Process(target=_crash_worker, args=(str(store), 250, 24, str(note)))
    p.start()
    p.join(timeout=60)                                # join reaps the pid
    assert p.exitcode == 0
    crashed = json.loads(note.read_text())

    before = fa.load_reservations(store)["reservations"][0]
    assert before["state"] == "running" and before["pid"] == crashed["pid"]
    assert fa.charged_total(reservations=fa.load_reservations(store)) == 24

    swept = fa.sweep_interrupted(store)
    assert swept["swept"] == [crashed["uuid"]] and swept["unverifiable"] == []
    row = fa.load_reservations(store)["reservations"][0]
    assert row["state"] == "interrupted" and row["planned"] == 24
    assert fa.charged_total(reservations=fa.load_reservations(store)) == 24   # no refund

    retry = fa.reserve(store, 250, 24, stage="validation")
    assert retry["charged_before"] == 24
    assert fa.charged_total(reservations=fa.load_reservations(store)) == 48   # accumulates


def test_a_reservation_owned_by_another_host_is_never_swept(tmp_path):
    store = tmp_path / fa.RESERVATION_NAME
    r = fa.reserve(store, 250, 24, stage="validation")
    fa.mark_running(store, r["uuid"])
    raw = json.loads(store.read_text())
    raw["reservations"][0]["hostname"] = "some-other-node"
    store.write_text(json.dumps(raw, indent=2))
    swept = fa.sweep_interrupted(store)
    assert swept["swept"] == [] and swept["unverifiable"] == [r["uuid"]]
    assert fa.load_reservations(store)["reservations"][0]["state"] == "running"
    assert swept["charged_open_total"] == 24           # charged either way


def test_a_stale_heartbeat_is_swept_only_when_a_timeout_is_asked_for(tmp_path):
    store = tmp_path / fa.RESERVATION_NAME
    r = fa.reserve(store, 250, 24, stage="validation", now="2026-09-11T10:00:00Z")
    fa.mark_running(store, r["uuid"], now="2026-09-11T10:00:00Z")

    def alive(*a, **k):                               # the owner is alive
        return True

    assert fa.sweep_interrupted(store, "2026-09-11T18:00:00Z", is_alive=alive)["swept"] == []
    out = fa.sweep_interrupted(store, "2026-09-11T18:00:00Z", stale_after_s=3600,
                               is_alive=alive)
    assert out["swept"] == [r["uuid"]] and out["charged_open_total"] == 24


# --- the archived run -------------------------------------------------------------
def test_the_union_reproduces_the_archived_counts_exactly(P):
    """416 executed and 240 unique scientific, against the real archive.

    If this disagrees the union logic is wrong: the archived ledger is evidence
    and is never edited to make a test pass.
    """
    ledger_path = ARCHIVE / t22.LEDGER_NAME
    assert ledger_path.is_file(), f"archived ledger missing: {ledger_path}"
    before = _sha(ledger_path)
    ledger = fa.require_fit_ledger(ledger_path)
    scan = t22.scan_executed_fits([ARCHIVE])
    u = fa.union_executed(ledger, scan, ledger_path=ledger_path)

    assert u["executed"] == 416
    assert u["unique_scientific"] == 240
    assert u["inference_only_fits"] == 176            # the one deterministic re-score
    assert u["fitted_models"] == 416
    assert u["restart_fits"] == 0
    assert u["charged"] == 416                        # no open reservations exist
    assert u["smoke_executed"] == 0

    # the same numbers the ledger itself stores, and the old max() agreed here
    assert ledger["executed_total"] == 416 and ledger["unique_scientific_total"] == 240
    assert _naive_max(ledger, scan) == 416

    # eight ledger entries and four scanned outputs collapse to eight executions:
    # the four fit-bearing ones were seen by both sources.
    assert len(u["records"]) == 8 and u["duplicates_merged"] == 4
    assert u["seen_by_both"] == 4
    assert sum(r["fits"] for r in u["records"]
               if {"ledger", "scan"} <= set(r["sources"])) == 416
    assert not any(r["fits_disagree"] for r in u["records"])
    assert [r["stage"] for r in u["records"] if r["fits"]] == [
        "development", "development", "validation", "final-test"]

    # the historical cap arithmetic is unchanged
    assert u["executed"] > P.cap == 250
    assert u["executed"] + 24 == 440 > P.cap
    assert _sha(ledger_path) == before                # the archive was not touched


def test_the_runner_budget_check_spends_the_union_not_the_maximum(tmp_path, P):
    """The wiring: a ledger-only execution plus a disk-only execution must sum.

    Under ``max()`` this stage would have been allowed (176 + 24 <= 250).
    """
    runs = tmp_path / "runs"
    _write_output(runs / "T22_validation_v1", "t22_validation.json",
                  _payload("validation", fits=60, finished="2026-09-11T16:24:15Z"))
    gone = hashlib.sha256(b"a development run archived elsewhere").hexdigest()
    ledger_path = runs / t22.LEDGER_NAME
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(_ledger(
        [_entry("development", "T22_development_v1", "t22_development.json", 176, gone)],
        cap=P.cap), indent=2))

    scan = t22.scan_executed_fits([runs])
    assert _naive_max(json.loads(ledger_path.read_text()), scan) == 176
    assert 176 + 24 <= P.cap                          # the old arithmetic let this through

    with pytest.raises(SystemExit) as exc:
        t22.check_cumulative_budget(P, ledger_path, 24, "validation", scan_dirs=[runs])
    msg = str(exc.value)
    assert "cumulative executed fits 236" in msg and "planned 24" in msg
    assert f"hard cap {P.cap}" in msg


def test_the_runner_budget_check_adds_open_reservations(tmp_path, P):
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    store = runs / fa.RESERVATION_NAME
    r = fa.reserve(store, P.cap, 240, stage="development")
    fa.mark_running(store, r["uuid"])
    fa.mark_failed(store, r["uuid"], "killed by the scheduler")
    with pytest.raises(SystemExit) as exc:
        t22.check_cumulative_budget(P, runs / t22.LEDGER_NAME, 24, "validation",
                                    scan_dirs=[runs])
    assert "cumulative executed fits 240" in str(exc.value)
