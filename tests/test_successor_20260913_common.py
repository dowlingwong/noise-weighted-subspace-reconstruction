"""Behavioural tests of the shared local-package infrastructure, using a toy
package that fits nothing.  Every rule the amendment asks for is exercised
here once so that the four scientific packages inherit it."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.mlst2026.successor_20260913.common import cli, fitcount as FC, ledger as L, matrix as M, protocol as PR, results as R  # noqa: E402

TOY_PROTOCOL = {
    "schema": PR.SCHEMA, "package": "toy", "version": "2026-09-13", "title": "toy", "question": "q",
    "hypotheses": ["h"], "independent_unit": "seed", "estimands": {"m": "mean"}, "controls": ["zero"],
    "seed_namespaces": {"development": [900000, 900099], "pilot": [1000, 1019], "confirmation": [2000, 2099]},
    "inference": {"rule": "fixed"}, "stopping_rule": "fixed allocation: all matrix cells, no early stop",
    "failure_interpretation": "report", "budget": {"estimator_fits_cap": 6, "fixture_to_production_scale": {"default": 10.0}},
    "explicitly_not_claimed": ["anything"],
}


def _toy_matrix(P, fixture):
    cells = []
    seeds = range(900000, 900003) if fixture else range(1000, 1003)
    role = "development" if fixture else "pilot"
    for s in seeds:
        cells.append(M.make_cell(role=role, seed=s, params={"x": 1}, n_fits=2, family="A"))
    cells.append(M.make_cell(role="algebraic", seed=None, params={"check": "identity"}, n_fits=0, n_solves=1, family="alg"))
    return M.RunMatrix("toy", P["_sha256"], cells)


def _toy_estimator(rng):
    """Stands in for an estimator invocation; counted where it happens."""
    FC.count_fit("toy_estimator")
    return float(rng.normal())


def _toy_run_cell(P, cell, fixture, ctx):
    if cell["role"] == "algebraic":
        return {"value": 0.0}, 0, 1
    rng = np.random.default_rng(cell["seed"])
    v = _toy_estimator(rng) + _toy_estimator(rng)      # two estimator invocations
    return {"value": v}, FC.observed(), 0


def _toy_reduce(P, matrix, cells, out, partial):
    vals = [c["result"]["value"] for c in cells.values() if c["role"] != "algebraic"]
    return {"mean": float(np.mean(vals)) if vals else None, "n": len(vals)}


def _main(argv, tmp_protocol, *, build_matrix=None, run_cell=None):
    return cli.main_template(package="toy", description="toy", build_matrix=build_matrix or _toy_matrix,
                             run_cell=run_cell or _toy_run_cell,
                             reduce_run=_toy_reduce, result_required={"value": float}, source_files=[Path(__file__)],
                             argv=[argv[0], "--protocol", str(tmp_protocol)] + argv[1:])


@pytest.fixture
def proto(tmp_path):
    p = tmp_path / "PROTOCOL_TOY.json"
    p.write_text(json.dumps(TOY_PROTOCOL))
    return p


def test_protocol_rejects_outcome_dependent_stopping(tmp_path):
    bad = dict(TOY_PROTOCOL, stopping_rule="budget spent, or acceptance decided, whichever first")
    with pytest.raises(PR.ProtocolError):
        PR.validate_protocol(bad)


def test_protocol_rejects_overlapping_namespaces():
    bad = dict(TOY_PROTOCOL, seed_namespaces={"development": [0, 10], "pilot": [5, 20], "confirmation": [30, 40]})
    with pytest.raises(PR.ProtocolError):
        PR.validate_protocol(bad)


def test_validate_and_enumerate_compute_nothing(proto, tmp_path, capsys):
    assert _main(["validate"], proto) == 0
    out = tmp_path / "enum"
    assert _main(["enumerate", "--out", str(out)], proto) == 0
    assert (out / "run_matrix.jsonl").exists() and not (out / "cells").exists()
    summ = json.loads((out / "run_matrix_summary.json").read_text())
    assert summ["fits"] == 6 and summ["solves"] == 1 and summ["by_role"]["pilot"]["cells"] == 3


def test_run_refuses_archive_roots(proto):
    rc = _main(["run", "--out", str(REPO / "results" / "toy_should_never_exist")], proto)
    assert rc == 6
    assert not (REPO / "results" / "toy_should_never_exist").exists()


def test_fixture_uses_development_namespace_only(proto, tmp_path):
    out = tmp_path / "fx"
    assert _main(["run", "--out", str(out), "--fixture"], proto) == 0
    cells = R.read_cells(out)
    seeds = {js["seed"] for files in cells.values() for _, js in files if js["seed"] is not None}
    assert seeds and all(900000 <= s <= 900099 for s in seeds)
    assert all(js["development_only"] for files in cells.values() for _, js in files)
    assert _main(["reduce", "--out", str(out)], proto) == 0
    red = json.loads((out / "REDUCED.json").read_text())
    assert red["development_only"] is True and red["reconciliation"]["complete"]


def test_full_run_reduce_is_deterministic_and_resumable(proto, tmp_path):
    out = tmp_path / "run"
    # partial run: stop after 2 cells, then resume
    assert _main(["run", "--out", str(out), "--max-cells", "2"], proto) == 0
    rc = _main(["reduce", "--out", str(out)], proto)
    assert rc == 3, "reduce must refuse a partial run without --allow-partial"
    assert _main(["reduce", "--out", str(out), "--allow-partial"], proto) == 0
    assert json.loads((out / "REDUCED.json").read_text())["partial"] is True
    assert _main(["run", "--out", str(out)], proto) == 0
    assert _main(["reduce", "--out", str(out)], proto) == 0
    a = json.loads((out / "REDUCED.json").read_text())
    assert _main(["reduce", "--out", str(out)], proto) == 0
    b = json.loads((out / "REDUCED.json").read_text())
    assert a["mean"] == b["mean"] and a["partial"] is False
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=6, package="toy", protocol_sha256=PR.load_protocol(proto)["_sha256"])
    assert led.charged()["fits"] == 6 and led.charged()["by_state"]["completed"] == 4


def test_failure_is_charged_and_retry_accumulates(proto, tmp_path):
    out = tmp_path / "fail"
    P = PR.load_protocol(proto)
    m = _toy_matrix(P, False)
    victim = [c for c in m.cells if c["role"] == "pilot"][0]["cell_id"]
    rc = _main(["run", "--out", str(out), "--fail-cell", victim], proto)
    assert rc == 1
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=6, package="toy", protocol_sha256=PR.load_protocol(proto)["_sha256"])
    ch = led.charged()
    assert ch["by_state"]["failed"] == 1 and ch["fits"] == 2  # the failed attempt stays charged
    # the retry needs 6 more fits but only 4 remain under the cap -> the third pilot cell is refused
    rc = _main(["run", "--out", str(out)], proto)
    assert rc == 2
    ch = led.charged()
    assert ch["fits"] == 6 and ch["by_state"]["completed"] == 2 and ch["by_state"]["failed"] == 1


def test_duplicate_completion_refused_and_repeat_flagged(proto, tmp_path):
    out = tmp_path / "dup"
    assert _main(["run", "--out", str(out)], proto) == 0
    # re-running is a no-op skip; forcing a repeat needs room under the cap -> refused by budget
    assert _main(["run", "--out", str(out)], proto) == 0
    P = PR.load_protocol(proto)
    m = _toy_matrix(P, False)
    alg = [c for c in m.cells if c["role"] == "algebraic"][0]["cell_id"]
    rc = _main(["run", "--out", str(out), "--cells", alg, "--allow-repeat"], proto)
    assert rc == 0
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=6, package="toy", protocol_sha256=PR.load_protocol(proto)["_sha256"])
    assert led.summary()["cells_with_repeat_completion"] == [alg]
    # a duplicated output file (copied by hand) is detected by reduce
    cells = list((out / "cells").glob("*.json"))
    src = cells[0]
    js = json.loads(src.read_text())
    js["cell_id"] = "deadbeefdeadbeef"
    (out / "cells" / "deadbeefdeadbeef.json").write_text(json.dumps(js))
    assert _main(["reduce", "--out", str(out)], proto) == 3


def test_tampered_output_is_rejected_at_reduce(proto, tmp_path):
    out = tmp_path / "tamper"
    assert _main(["run", "--out", str(out)], proto) == 0
    p = sorted((out / "cells").glob("*.json"))[0]
    js = json.loads(p.read_text())
    js["result"]["value"] = 42.0
    p.write_text(json.dumps(js))
    assert _main(["reduce", "--out", str(out)], proto) == 3


def test_interrupted_attempt_is_swept_and_charged(proto, tmp_path):
    out = tmp_path / "intr"
    P = PR.load_protocol(proto)
    m = _toy_matrix(P, False)
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=6, package="toy", protocol_sha256=P["_sha256"])
    cell = [c for c in m.cells if c["role"] == "pilot"][0]
    led.start(cell)
    n = led.sweep_interrupted(is_alive=lambda pid, host: False)
    assert n == 1
    ch = led.charged()
    assert ch["by_state"]["interrupted"] == 1 and ch["fits"] == 2


# --- the ledger keeps its state incrementally, and that state is the truth ------------------
def _led(path, cap=10 ** 9):
    return L.ExecutionLedger(path, cap=cap, package="toy", protocol_sha256="0" * 64)


def _cell(i, fits=2, solves=1, role="pilot"):
    return {"cell_id": f"c{i:05d}", "role": role, "n_fits": fits, "n_solves": solves}


def test_incremental_ledger_state_agrees_with_a_full_reparse_across_two_instances(tmp_path):
    """Two ``ExecutionLedger`` objects on one file, appending in an interleaved
    order, must each end with the state a full re-parse of the file produces.
    This is the correctness condition for keeping the charge and the completed-cell
    set in memory instead of re-reading the ledger on every ``start``."""
    path = tmp_path / "EXECUTION_LEDGER.jsonl"
    a, b = _led(path), _led(path)
    for i in range(6):
        writer, other = (a, b) if i % 2 == 0 else (b, a)
        cell = _cell(i)
        u = writer.start(cell)
        if i % 3 == 0:
            writer.complete(u, cell, fits_used=2, solves_used=1, output_path=tmp_path / f"{i}.json",
                            output_sha256="d" * 64, wall_s=0.1)
        elif i % 3 == 1:
            writer.fail(u, cell, reason="injected", wall_s=0.1)
        # i % 3 == 2 is left running, and stays charged
        # the other instance must see the append without being told about it
        assert other.charged()["attempts"] == i + 1
    # a repeat completion of an already completed cell, from the instance that did
    # not write the first one
    cell0 = _cell(0)
    u = b.start(cell0, allow_repeat=True)
    b.complete(u, cell0, fits_used=2, solves_used=1, output_path=tmp_path / "0b.json",
               output_sha256="e" * 64, wall_s=0.1)
    a.sweep_interrupted(is_alive=lambda pid, host: False)
    for led in (a, b):
        assert led.state_matches_full_reparse(), led.path
    # and the two instances agree with each other
    assert a.charged() == b.charged()
    assert {k: len(v) for k, v in a.completed_by_cell().items()} == \
           {k: len(v) for k, v in b.completed_by_cell().items()}
    assert len(a.completed_by_cell()["c00000"]) == 2          # the repeat is a second completion
    assert a.summary()["cells_with_repeat_completion"] == ["c00000"]
    # a fresh instance, which must rebuild the same state from the file alone
    assert _led(path).charged() == a.charged()


def test_ledger_start_cost_does_not_grow_with_the_cell_count(tmp_path):
    """The defect this guards: ``start`` used to call ``charged()`` and
    ``completed_by_cell()``, each of which re-read and re-parsed the whole ledger,
    so a run was quadratic in the cell count (measured: 1.2 ms at cell 0 and 83 ms
    at cell 2 498, 69.6 s for 2 499 cells).  The state is now folded incrementally,
    so the per-``start`` cost is flat.

    The threshold is deliberately loose: this runs on a 2-core container where a
    sub-millisecond operation is noisy, so the test asserts a shape (no growth by
    more than 4x over a 20x longer ledger) rather than a time."""
    import time
    path = tmp_path / "EXECUTION_LEDGER.jsonl"
    led = _led(path)

    def burn(n, first_id):
        for i in range(n):
            cell = _cell(first_id + i)
            u = led.start(cell)
            led.complete(u, cell, fits_used=2, solves_used=1, output_path=tmp_path / "x.json",
                         output_sha256="d" * 64, wall_s=0.0)

    def time_starts(n, first_id):
        best = []
        for i in range(n):
            cell = _cell(first_id + i)
            t0 = time.perf_counter()
            u = led.start(cell)
            best.append(time.perf_counter() - t0)
            led.complete(u, cell, fits_used=2, solves_used=1, output_path=tmp_path / "x.json",
                         output_sha256="d" * 64, wall_s=0.0)
        best.sort()
        return best[len(best) // 2]

    burn(100, 0)
    at_100 = time_starts(40, 1000)
    burn(1860, 2000)
    at_2000 = time_starts(40, 8000)
    assert led.charged()["attempts"] == 2040
    # 20x the ledger length must not cost anything like 20x the time
    assert at_2000 < max(4.0 * at_100, 4e-3), (at_100, at_2000)


def test_ledger_still_charges_at_start_and_never_refunds(tmp_path):
    """Every behaviour the incremental state has to preserve, in one place."""
    path = tmp_path / "EXECUTION_LEDGER.jsonl"
    led = L.ExecutionLedger(path, cap=5, package="toy", protocol_sha256="0" * 64)
    c0 = _cell(0, fits=2)
    u0 = led.start(c0)
    assert led.charged()["fits"] == 2                       # charged at start, before any compute
    led.fail(u0, c0, reason="boom", wall_s=0.0)
    assert led.charged()["fits"] == 2                       # a failure is not refunded
    c1 = _cell(1, fits=2)
    u1 = led.start(c1)
    led.complete(u1, c1, fits_used=1, solves_used=0, output_path=path, output_sha256="a" * 64, wall_s=0.0)
    assert led.charged()["fits"] == 3                       # a completion is charged what it used
    with pytest.raises(L.DuplicateCompletion) as dup:
        led.start(_cell(1, fits=2))
    assert L.DuplicateCompletion.exit_code == 5 and "already has a completed attempt" in str(dup.value)
    with pytest.raises(L.BudgetExceeded) as be:
        led.start(_cell(2, fits=3))
    assert L.BudgetExceeded.exit_code == 2 and "exceeds cap 5" in str(be.value)
    u2 = led.start(_cell(2, fits=2))
    assert led.charged()["fits"] == 5
    assert led.sweep_interrupted(is_alive=lambda pid, host: False) == 1
    assert led.charged()["by_state"]["interrupted"] == 1 and led.charged()["fits"] == 5
    assert led.state_matches_full_reparse()
    assert u0 != u1 != u2


def test_help_never_computes(proto, tmp_path):
    with pytest.raises(SystemExit) as e:
        _main(["--help"], proto)
    assert e.value.code == 0


# --- the fit-accounting guard is a real check, not a constant compared with itself ---------
def _ledger(out):
    return [json.loads(l) for l in (out / "EXECUTION_LEDGER.jsonl").read_text().splitlines() if l.strip()]


def test_fit_guard_rejects_a_cell_that_restates_the_declared_constant(proto, tmp_path):
    """A cell that reports ``cell['n_fits']`` instead of what it observed is the
    defect the guard exists to catch: the declaration would be compared with
    itself.  The instrumentation makes it a third, independent number."""
    def restating_run_cell(P, cell, fixture, ctx):
        if cell["role"] == "algebraic":
            return {"value": 0.0}, 0, 1
        rng = np.random.default_rng(cell["seed"])
        v = _toy_estimator(rng)                    # ONE estimator invocation ...
        return {"value": v}, int(cell["n_fits"]), 0   # ... but the declared two are restated

    out = tmp_path / "restated"
    assert _main(["run", "--out", str(out)], proto, run_cell=restating_run_cell) == 4
    assert not list((out / "cells").glob("*.json"))
    recs = _ledger(out)
    assert [r["state"] for r in recs] == ["running", "failed"]
    assert "instrumentation mismatch" in recs[-1]["error"]


def test_fit_guard_rejects_a_wrong_declaration_in_the_matrix(proto, tmp_path):
    """The other direction: the matrix declares three fits, the cell performs two."""
    def wrong_matrix(P, fixture):
        m = _toy_matrix(P, fixture)
        for c in m.cells:
            if c["role"] == "pilot":
                c["n_fits"] = 3
        return m

    out = tmp_path / "misdeclared"
    assert _main(["run", "--out", str(out)], proto, build_matrix=wrong_matrix) == 4
    recs = _ledger(out)
    assert recs[-1]["state"] == "failed" and "accounting mismatch" in recs[-1]["error"]
    # charged and not refunded: the failed attempt still counts its planned fits
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=6, package="toy",
                            protocol_sha256=recs[-1]["protocol_sha256"])
    assert led.charged()["fits"] == 3 and led.charged()["by_state"]["failed"] == 1


def test_observed_fit_breakdown_is_written_into_every_cell_file(proto, tmp_path):
    out = tmp_path / "obs"
    assert _main(["run", "--out", str(out)], proto) == 0
    for p in sorted((out / "cells").glob("*.json")):
        js = json.loads(p.read_text())
        assert js["fits_observed"]["total"] == js["fits_used"]
        if js["role"] == "pilot":
            assert js["fits_observed"]["by_label"] == {"toy_estimator": 2}


def test_count_fit_outside_a_counting_region_is_inert(proto):
    assert FC.current() is None
    assert FC.count_fit("nowhere") == 1
    with pytest.raises(RuntimeError):
        FC.observed()
    with FC.counting() as fc:
        FC.count_fit("a", n=2)
        FC.count_fit("b")
        assert FC.observed() == 3 and fc.as_dict()["by_label"] == {"a": 2, "b": 1}
    assert FC.current() is None


def test_ledger_rejects_changed_protocol_before_resume(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps({"schema": L.LEDGER_SCHEMA, "package": "toy",
                                "protocol_sha256": "original"}) + "\n")
    assert len(L.ExecutionLedger(path, cap=6, package="toy", protocol_sha256="original").records()) == 1
    with pytest.raises(ValueError, match="protocol hash"):
        L.ExecutionLedger(path, cap=6, package="toy", protocol_sha256="changed").records()


def test_fit_limit_refuses_before_extra_estimator_body():
    calls = []
    @FC.counted("tracked")
    def estimator():
        calls.append(1)
    with FC.counting(limit=1) as observed:
        estimator()
        with pytest.raises(R.SchemaError, match="before invocation"):
            estimator()
    assert len(calls) == observed.total == 1


def test_failed_observed_overrun_is_never_undercharged(tmp_path):
    led = L.ExecutionLedger(tmp_path / "ledger.jsonl", cap=10, package="toy", protocol_sha256="hash")
    cell = {"cell_id": "test", "role": "pilot", "n_fits": 2}
    attempt = led.start(cell)
    led.fail(attempt, cell, reason="legacy overrun", wall_s=0.0, fits_observed=3)
    assert led.charged()["fits"] == 3

# =============================================================================
# the measurement harness (common.benchmark) and the measured resource estimate
# =============================================================================
from experiments.mlst2026.successor_20260913.common import benchmark as BM  # noqa: E402

SUCC = REPO / "experiments" / "mlst2026" / "successor_20260913"
PACKAGES = ("t19s", "t20s", "t21s", "t22s", "cresst_s")
#: the enumerated production matrix, as `enumerate` prints it (cells, fits, solves)
PRODUCTION_COUNTS = {"t19s": (136, 3840, 3605), "t20s": (964, 4800, 140), "t21s": (2499, 2496, 2498),
                     "t22s": (727, 0, 0), "cresst_s": (395, 1056, 0)}
REQUIRED_ESTIMATE_FIELDS = ("schema", "estimate_basis", "hardware", "thread_settings", "families",
                            "unmeasured_families", "cpu_seconds", "total_cpu_seconds_point",
                            "wall_time_hours_by_workers", "wall_time_model", "parallelism_assumption",
                            "reduction_work", "harness_work", "work_accounting", "peak_memory", "storage",
                            "transfer", "setup", "unknowns", "cross_check_flat_fixture_scale",
                            "priced_counts")


def test_power_law_fit_recovers_a_quadratic():
    """A synthetic O(n^2) input must come back with exponent 2."""
    sizes = [8, 16, 32, 64, 128]
    times = [3e-6 * n ** 2 for n in sizes]
    fit = BM.fit_power_law(sizes, times)
    assert fit.b == pytest.approx(2.0, abs=1e-9)
    assert fit.a == pytest.approx(3e-6, rel=1e-9)
    assert fit.residual_log_rms == pytest.approx(0.0, abs=1e-12)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    assert fit.predict(256) == pytest.approx(3e-6 * 256 ** 2, rel=1e-9)


def test_power_law_fit_recovers_the_exponent_under_jitter_and_reports_a_residual():
    rng = np.random.default_rng(7)
    sizes = [16, 32, 64, 128, 256]
    times = [1e-9 * n ** 3 * float(np.exp(rng.normal(0, 0.05))) for n in sizes]
    fit = BM.fit_power_law(sizes, times)
    assert fit.b == pytest.approx(3.0, abs=0.15)
    assert fit.residual_log_rms > 0.0                    # the jitter is reported, not smoothed away
    assert fit.max_abs_log_residual >= fit.residual_log_rms


def test_power_law_fit_refuses_degenerate_input():
    with pytest.raises(ValueError):
        BM.fit_power_law([4], [1.0])
    with pytest.raises(ValueError):
        BM.fit_power_law([4, 4, 4], [1.0, 1.1, 1.2])
    with pytest.raises(ValueError):
        BM.fit_power_law([1, 2], [0.0, 1.0])


def test_exponent_verdict_names_a_disagreement_instead_of_hiding_it():
    ok = BM.exponent_verdict(2.95, 3.0)
    assert ok["agrees"] and "agrees" in ok["note"]
    low = BM.exponent_verdict(1.4, 3.0)
    assert not low["agrees"] and low["difference"] == pytest.approx(-1.6)
    assert "overhead" in low["note"] and "UNDERSTATES" in low["note"]
    high = BM.exponent_verdict(3.9, 3.0)
    assert not high["agrees"] and "ABOVE" in high["note"]


def test_extrapolation_labels_a_projection_of_more_than_a_decade():
    same = BM.extrapolation(256, 256)
    assert same["is_measurement"] and not same["spans_more_than_one_decade"]
    near = BM.extrapolation(128, 256)
    assert not near["is_measurement"] and not near["spans_more_than_one_decade"]
    far = BM.extrapolation(24, 4096)
    assert far["spans_more_than_one_decade"] and "PROJECTED" in far["label"]
    assert far["extrapolation_factor_in_size"] == pytest.approx(4096 / 24)


def test_timed_discards_the_warmup_and_reports_min_median_max():
    calls = []

    def fn():
        calls.append(1)
    t = BM.timed(fn, size=3, repeats=4, warmup=2)
    assert len(calls) == 6 and t.repeats == 4 and t.warmup == 2 and len(t.samples_s) == 4
    assert t.min_s <= t.median_s <= t.max_s and t.size == 3.0
    assert t.as_dict()["spread_fraction"] >= 0.0


def test_peak_memory_reports_both_measures_and_says_which_is_which():
    m = BM.peak_memory(lambda: np.zeros(2_000_000))
    assert m["tracemalloc_peak_bytes"] > 8_000_000            # 2e6 float64
    assert "this call only" in m["tracemalloc_is"]
    assert m["rss_high_water_bytes"] > 0 and "VmHWM" in m["rss_is"]
    # ru_maxrss is still reported, and is labelled with the reason it is not the one
    # to read: Linux inherits it across fork, so in a spawned probe it is the parent's
    assert m["ru_maxrss_bytes"] > 0 and "fork" in m["ru_maxrss_is"]


def test_peak_memory_in_a_child_is_not_the_parents_high_water_mark():
    """The defect: `ru_maxrss` is inherited across `fork`, so a subprocess memory probe
    spawned by a driver holding 300 MB reported at least 300 MB whatever the child did.
    Every per-size RSS figure in the benchmarks came through that path."""
    big = np.zeros((3000, 3000))                              # ~72 MB in this process
    assert big.sum() == 0.0
    child = ("import json,sys;sys.path.insert(0,%r);"
             "from experiments.mlst2026.successor_20260913.common import benchmark as B;"
             "print(json.dumps(B.peak_memory(lambda: None)))" % str(REPO))
    r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stderr
    m = json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1])
    parent = BM._vm_hwm_bytes()
    assert m["rss_high_water_bytes"] < parent, (m["rss_high_water_bytes"], parent)
    assert m["ru_maxrss_bytes"] >= m["rss_high_water_bytes"]


def test_wall_time_range_divides_by_the_worker_count():
    w = BM.wall_time_range(3600.0, 7200.0, (1, 8, 32))
    assert w["1"]["hours_low"] == pytest.approx(1.0) and w["1"]["hours_high"] == pytest.approx(2.0)
    assert w["8"]["hours_low"] == pytest.approx(0.125) and w["32"]["hours_high"] == pytest.approx(2.0 / 32)


def test_estimate_without_a_benchmark_says_so_instead_of_guessing(proto, tmp_path):
    out = tmp_path / "fx"
    assert _main(["run", "--out", str(out), "--fixture"], proto) == 0
    assert _main(["estimate", "--out", str(out)], proto) == 0
    est = json.loads((out / "RESOURCE_ESTIMATE.json").read_text())
    assert est["measured_projection_available"] is False
    assert "BENCHMARK.json" in est["why_no_measured_projection"]
    assert sorted(est["unmeasured_families"]) == ["A", "alg"]   # nothing is invented for them
    assert est["families"]["A"]["measured"] is False
    for field in REQUIRED_ESTIMATE_FIELDS:
        assert field in est, field


def test_estimate_with_a_benchmark_is_not_a_flat_multiple_of_the_fixture_time(proto, tmp_path):
    """The headline must come from the benchmark, not from fixture_seconds x scale."""
    out = tmp_path / "fx"
    assert _main(["run", "--out", str(out), "--fixture"], proto) == 0
    (proto.parent / "BENCHMARK.json").write_text(json.dumps({
        "schema": "mlst-local-package-benchmark/1", "package": "toy",
        "hardware": BM.HARDWARE_BLOCK, "thread_settings": BM.THREAD_ENV,
        "kernels": {"toy_kernel": {"kernel": "toy_kernel", "size_symbol": "n",
                                   "points": [{"size": n, "median_s": 1e-4 * n ** 2, "min_s": 0.9e-4 * n ** 2,
                                               "max_s": 1.1e-4 * n ** 2, "repeats": 3}
                                              for n in (8, 16, 32, 64)],
                                   "fit": BM.fit_power_law([8, 16, 32, 64],
                                                           [1e-4 * n ** 2 for n in (8, 16, 32, 64)]).as_dict(),
                                   "theory": {"key": "eigendecomposition", "exponent": 3.0, "note": "O(d^3)"},
                                   "projection": {"basis": "measured", "seconds_per_call_median": 0.4096}}},
        "families": {"A": {"kernel": "toy_kernel", "calls_per_cell": 1, "basis": "measured", "size": 64,
                           "size_symbol": "n", "fitted_exponent": 2.0, "theoretical_exponent": 3.0,
                           "extrapolation_factor_in_dimension": 1.0,
                           "seconds_per_cell": {"median": 0.4096, "low": 0.36864, "high": 0.45056},
                           "how": "measured at n = 64"},
                     "alg": {"kernel": "toy_kernel", "calls_per_cell": 1, "basis": "projected", "size": 64,
                             "size_symbol": "n", "fitted_exponent": 2.0, "theoretical_exponent": 3.0,
                             "extrapolation_factor_in_dimension": 2.0,
                             "seconds_per_cell": {"median": 0.01, "low": 0.009, "high": 0.011},
                             "how": "projected with the fitted exponent"}},
        "work_accounting": {"estimator_calls_per_cell": 2, "restarts": "none"},
        "reduce": {"measured": True, "cells_read": 4, "seconds_per_cell": 0.01,
                   "wall_seconds_reduce_work": 0.04, "ledger_bytes_per_cell": 900.0,
                   "reduced_output_bytes": 4000, "bound": "compute",
                   "throughput_bytes_per_second": 1e6},
        "storage": {"bytes_per_cell_by_family": {"A": {"bytes": 1000, "basis": "measured", "how": "x"},
                                                 "alg": {"bytes": 500, "basis": "measured", "how": "x"}}},
        "setup_and_transfer": {"transfer": {"archive_bytes": 7_000_000}},
        "notes": ["toy"],
    }) + "\n")
    assert _main(["estimate", "--out", str(out)], proto) == 0
    est = json.loads((out / "RESOURCE_ESTIMATE.json").read_text())
    for field in REQUIRED_ESTIMATE_FIELDS:
        assert field in est, field
    assert est["measured_projection_available"] is True and est["unmeasured_families"] == []
    # 3 pilot cells x 0.4096 s + 1 algebraic cell x 0.01 s
    assert est["cpu_seconds"]["cell_compute_point"] == pytest.approx(3 * 0.4096 + 0.01)
    # the headline is NOT the flat fixture x declared-scale product
    flat = est["cross_check_flat_fixture_scale"]["total_cpu_seconds_point"]
    assert flat > 0 and est["cpu_seconds"]["cell_compute_point"] != pytest.approx(flat)
    assert "NOT the estimate" in est["cross_check_flat_fixture_scale"]["what_this_is"]
    # the projected family names the exponent it used and the factor it spanned
    assert est["families"]["alg"]["basis"] == "projected"
    assert est["families"]["alg"]["fitted_exponent"] == 2.0
    assert est["families"]["alg"]["extrapolation_factor_in_dimension"] == 2.0
    # ranges, not a point; and the wall time is stated per worker count and marked conditional
    assert est["cpu_seconds"]["total_low"] < est["cpu_seconds"]["total_point"] < est["cpu_seconds"]["total_high"]
    assert set(est["wall_time_hours_by_workers"]) == {"1", "8", "32"}
    for w, row in est["wall_time_hours_by_workers"].items():
        assert "CONDITIONAL" in row["conditional"] and row["hours_low"] <= row["hours_high"]
    # reduction, storage, transfer and setup are all present and separated from compute
    assert est["reduction_work"]["measured"] and est["reduction_work"]["total_seconds_projected"] > 0
    assert est["storage"]["total_mb"] > 0 and est["transfer"]["separate_from_computation"]
    assert est["setup"]["separate_from_computation"] and len(est["unknowns"]) >= 5


@pytest.mark.parametrize("pkg", PACKAGES)
def test_shipped_resource_estimate_is_measured_and_scaling_aware(pkg):
    est = json.loads((SUCC / pkg / "RESOURCE_ESTIMATE.json").read_text())
    for field in REQUIRED_ESTIMATE_FIELDS:
        assert field in est, field
    assert est["schema"] == "mlst-local-package-resource-estimate/2"
    assert est["measured_projection_available"] is True
    assert est["unmeasured_families"] == []
    assert "2.10GHz" in est["hardware"] and "OMP_NUM_THREADS=1" in est["hardware"]
    cells, fits, solves = PRODUCTION_COUNTS[pkg]
    assert (est["priced_counts"]["cells"], est["priced_counts"]["fits"],
            est["priced_counts"]["solves"]) == (cells, fits, solves)
    assert "development" not in est["priced_counts"]["by_role"]
    # every family is labelled, and a projected one names its exponent
    for fam, row in est["families"].items():
        assert row["measured"] is True, fam
        assert row["basis"] in ("measured", "projected", "measured-on-substitute"), fam
        if row["basis"] == "projected":
            assert row["fitted_exponent"] is not None and row["extrapolation_factor_in_dimension"] is not None
    # the headline is not the flat fixture x declared-scale product
    flat = est["cross_check_flat_fixture_scale"]["total_cpu_seconds_point"]
    assert est["cpu_seconds"]["cell_compute_point"] != pytest.approx(flat)
    assert est["cpu_seconds"]["total_low"] <= est["cpu_seconds"]["total_point"] <= est["cpu_seconds"]["total_high"]
    assert est["reduction_work"]["measured"] is True
    assert est["peak_memory"]["measured"] is True and est["storage"]["total_mb"] > 0
    assert est["transfer"]["archive_bytes"] and est["setup"]["requirements"]["pinned_lines"]
    assert set(est["wall_time_hours_by_workers"]) == {"1", "8", "32"}
    # the harness around the kernel is measured, priced, and inside the headline total
    c, h = est["cpu_seconds"], est["harness_work"]
    assert h["measured"] is True and h["total_seconds_projected"] > 0
    assert c["harness"] == pytest.approx(h["total_seconds_projected"])
    assert c["total_point"] == pytest.approx(c["cell_compute_point"] + c["harness"] + c["reduction"])
    assert est["total_cpu_seconds_point"] == pytest.approx(c["total_point"])
    # the wall model carries an explicit serial term, and it is not zero
    m = est["wall_time_model"]
    assert "serial_ledger" in m["formula"] and m["serial_ledger_seconds"] > 0
    for w in ("1", "8", "32"):
        row = est["wall_time_hours_by_workers"][w]
        expect = ((c["cell_compute_point"] + m["parallel_harness_seconds"]) / int(w)
                  + m["serial_ledger_seconds"] + m["reduction_seconds"]) / 3600.0
        assert row["hours_point"] == pytest.approx(expect)
        assert row["hours_point"] >= m["serial_floor_hours"]
    # M1: the excludes note must not point at a number that is not in the file
    ex = c["excludes"]
    assert "interpreter start-up" in ex
    if "UNQUANTIFIED" not in ex:
        assert h["per_worker_interpreter_startup_seconds_measured"] > 0
        assert "harness_work.per_worker_interpreter_startup_seconds_measured" in ex
    # m13: the concurrency caveat is its own named unknown
    assert any("concurrent workers" in u for u in est["unknowns"])


@pytest.mark.parametrize("pkg", PACKAGES)
def test_shipped_benchmark_times_every_fitted_kernel_at_four_or_more_sizes(pkg):
    """The claim in README.md is per kernel ("the package's own dominant kernel timed
    at >= 4 sizes"), so the test is per kernel.  `any(...)` would have passed a
    benchmark in which one kernel was scanned properly and the rest were single
    points.  The one honest exception is an axis with fewer than four production
    values: `replicate_by_n_files` runs over n_files in {4, 8, 16} and there is no
    fourth value to run, so it must instead say that every production value was run."""
    b = json.loads((SUCC / pkg / "BENCHMARK.json").read_text())
    assert b["schema"] == "mlst-local-package-benchmark/1" and b["package"] == pkg
    assert b["hardware"] == BM.HARDWARE_BLOCK
    assert b["thread_settings"]["OMP_NUM_THREADS"] == "1"
    scaled = {n: k for n, k in b["kernels"].items() if "fit" in k}
    assert scaled, "no kernel carries a scaling fit"
    for name, k in scaled.items():
        sizes = [r["size"] for r in k["points"]]
        assert len(set(sizes)) == len(sizes), (pkg, name, "a size was benchmarked twice")
        if len(sizes) < 4:
            label = (k.get("projection") or {}).get("label", "")
            assert "every production" in label, (pkg, name, len(sizes), label)
            assert len(sizes) >= 3, (pkg, name, sizes)
        assert "theory" in k and "exponent_check" in k
        assert k["exponent_check"]["theoretical_exponent"] == k["theory"]["exponent"]
        assert k["fit"]["n_points"] == len(sizes)
    # every priced family points at a kernel that exists and was actually run: the
    # defect this catches is a family labelled `measured` against a kernel result
    # that was copied from a different configuration
    for fam, row in b["families"].items():
        kn = row.get("kernel")
        assert kn in b["kernels"], (pkg, fam, kn)
        assert b["kernels"][kn].get("points"), (pkg, fam, kn)
    assert any((r.get("memory") or {}).get("rss_high_water_bytes")
               for k in b["kernels"].values() for r in k.get("points", []))
    assert set(b["storage"]["bytes_per_cell_by_family"]) >= set(b["families"])


@pytest.mark.parametrize("pkg", PACKAGES)
def test_shipped_benchmark_measures_the_per_cell_harness(pkg):
    """The kernel benchmarks time a kernel; production runs a whole cell through
    common/cli.py.  That harness has to be measured, not assumed away."""
    h = json.loads((SUCC / pkg / "BENCHMARK.json").read_text())["harness"]
    assert h["measured"] is True, h.get("why")
    assert h["harness_seconds_per_cell"] > 0
    assert 0 < h["serial_seconds_per_cell"] <= h["harness_seconds_per_cell"]
    assert h["end_to_end"]["cells"] >= 10 and h["end_to_end"]["harness_seconds_per_cell"] > 0
    # the two scale-dependent components are measured at BOTH scales, which is what
    # makes the fixture measurement re-pricable at the production scale
    c = h["components"]
    assert c["ledger_at_production_length"]["existing_records"] > \
           c["ledger_at_fixture_length"]["existing_records"]
    assert c["write_and_schema_at_production_payload"]["payload_bytes"] > 0
    assert h["per_worker_interpreter_startup_seconds_measured"] > 0


@pytest.mark.parametrize("pkg", PACKAGES)
def test_shipped_estimate_is_regenerable_from_the_shipped_benchmark(pkg, tmp_path):
    """`make_handoff._check_measured_estimate` is structural: it cannot tell a fresh
    benchmark paired with a stale estimate from a consistent pair, so a stale estimate
    shipped silently.  This recomputes the estimate from the shipped BENCHMARK.json and
    the enumerated production matrix and requires the shipped one to match.

    The flat cross-check is excluded because it is built from a fixture RUN ROOT (per-cell
    wall times), which is not in the repository; everything that comes from the benchmark
    and the matrix is compared."""
    import importlib
    runner = importlib.import_module(f"experiments.mlst2026.successor_20260913.{pkg}.runner")
    proto = next(iter(sorted((SUCC / pkg).glob("PROTOCOL_*.json"))))
    P = PR.load_protocol(proto)
    out = tmp_path / "regen"
    out.mkdir()
    assert cli._estimate(P, runner.build_matrix(P, False), out) == 0
    fresh = json.loads((out / "RESOURCE_ESTIMATE.json").read_text())
    shipped = json.loads((SUCC / pkg / "RESOURCE_ESTIMATE.json").read_text())
    for key in ("families", "cpu_seconds", "total_cpu_seconds_point", "total_cpu_hours_point",
                "wall_time_hours_by_workers", "wall_time_model", "reduction_work", "harness_work",
                "storage", "peak_memory", "priced_counts", "unmeasured_families",
                "parallelism_assumption", "unknowns"):
        assert fresh[key] == shipped[key], (
            f"{pkg}: RESOURCE_ESTIMATE.json['{key}'] is not what the shipped BENCHMARK.json now produces; "
            f"re-run the `estimate` subcommand")


def test_shipped_resource_report_is_what_the_generator_produces():
    """The report's own header says every number in it comes out of the JSON.  The
    generator's `--check` mode regenerates the report and diffs it; it has no clock in
    it (the stamp is the newest benchmark `generated_utc`), so any difference is a real
    staleness."""
    r = subprocess.run([sys.executable, str(SUCC / "make_resource_report.py"), "--check"],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stdout + r.stderr
