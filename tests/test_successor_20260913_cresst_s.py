"""Local acceptance tests for CRESST-S (L5).

The eight tests the round's README requires of every package, plus the package's own
scientific acceptance: that the recomputation reproduces the archive, that the
cluster-aware reporting matches the archived leave-one-out entries, that a between-run
variance component is refused from two runs, that a robustness cell fails cleanly when
the raw CRESST data root is absent, that the calibration shows lower coverage at two
runs than at eight, and that two injected incorrect scientific outputs are rejected by
the package's own checks rather than by a digest.

The archived inputs live in the MANUSCRIPT repository.  Their location is passed
explicitly (``--archive-root``); it is taken from ``CRESST_S_ARCHIVE_ROOT`` when set and
otherwise from the sibling ``MS`` checkout.  The tests skip if it cannot be found.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.mlst2026.successor_20260913.common import ledger as L  # noqa: E402
from experiments.mlst2026.successor_20260913.common import protocol as PR  # noqa: E402
from experiments.mlst2026.successor_20260913.cresst_s import archive_recompute as AR  # noqa: E402
from experiments.mlst2026.successor_20260913.cresst_s import calibration as CB  # noqa: E402
from experiments.mlst2026.successor_20260913.cresst_s import cluster_report as CR  # noqa: E402
from experiments.mlst2026.successor_20260913.cresst_s import robustness_design as RD  # noqa: E402
from experiments.mlst2026.successor_20260913.cresst_s import runner as RU  # noqa: E402

PROTOCOL = REPO / "experiments" / "mlst2026" / "successor_20260913" / "cresst_s" / "PROTOCOL_CRESST_S_v1.json"


def _archive_root() -> Path | None:
    for cand in (os.environ.get("CRESST_S_ARCHIVE_ROOT"),
                 REPO.parent / "MS" / "evidence" / "data"):
        if not cand:
            continue
        p = Path(cand)
        if (p / AR.ARCHIVE_INPUTS["event_residuals_lossless"]).is_file():
            return p
    return None


ARCHIVE = _archive_root()
needs_archive = pytest.mark.skipif(ARCHIVE is None,
                                   reason="the archived CRESST inputs (MS evidence/data) are not reachable")


def main(args, *, archive=True):
    argv = list(args) + ["--protocol", str(PROTOCOL)]
    if archive and ARCHIVE is not None:
        argv += ["--archive-root", str(ARCHIVE)]
    return RU.main(argv)


@pytest.fixture(scope="module")
def P():
    return PR.load_protocol(PROTOCOL)


@pytest.fixture(scope="module")
def fixture_matrix(P):
    return RU.build_matrix(P, True)


def _cell_id(matrix, **want):
    hits = [c for c in matrix.cells if all(c["params"].get(k) == v for k, v in want.items())]
    assert len(hits) == 1, f"expected one cell for {want}, found {len(hits)}"
    return hits[0]["cell_id"]


@pytest.fixture(scope="module")
def fx_run(tmp_path_factory):
    """One fixture run + reduce, shared by every test that reads the reduced output."""
    if ARCHIVE is None:
        pytest.skip("archive not reachable")
    out = tmp_path_factory.mktemp("cresst_s_fixture")
    assert main(["run", "--out", str(out), "--fixture"]) == 0
    assert main(["reduce", "--out", str(out)]) == 0
    return out


@pytest.fixture(scope="module")
def reduced(fx_run):
    return json.loads((fx_run / "REDUCED.json").read_text())


# --- 1. validate / enumerate compute nothing --------------------------------------------------
def test_validate_and_enumerate_compute_nothing(tmp_path):
    assert main(["validate"], archive=False) == 0
    out = tmp_path / "enum"
    assert main(["enumerate", "--out", str(out)], archive=False) == 0
    assert (out / "run_matrix.jsonl").exists()
    assert not (out / "cells").exists()
    summ = json.loads((out / "run_matrix_summary.json").read_text())
    assert summ["cells"] == 395 and summ["fits"] == 1056 and summ["solves"] == 0
    assert summ["by_role"]["confirmation"]["cells"] == 327
    # the robustness cells that cannot run here are marked in the matrix itself
    rows = [json.loads(l) for l in (out / "run_matrix.jsonl").read_text().splitlines() if l.strip()]
    marked = [r for r in rows if "NOT EXECUTABLE" in (r.get("notes") or "")]
    assert len(marked) == 350 and all(r["family"].startswith("robustness") for r in marked)


def test_help_never_computes():
    with pytest.raises(SystemExit) as e:
        main(["--help"], archive=False)
    assert e.value.code == 0


# --- 2. fixture end to end --------------------------------------------------------------------
@needs_archive
def test_fixture_run_and_reduce_has_the_declared_schema(fx_run, reduced):
    assert reduced["development_only"] is True
    assert reduced["partial"] is False
    assert reduced["reconciliation"]["complete"] is True
    for key in ("archive_recomputation", "archive_intervals", "cluster_report", "calibration",
                "controls", "robustness_design", "scientific_checks", "replicates_by_role", "statement"):
        assert key in reduced, key
    assert "seven run/channel clusters from TWO runs" in reduced["statement"]
    assert "DEVELOPMENT ONLY" in reduced["statement"]
    cells = sorted((fx_run / "cells").glob("*.json"))
    assert len(cells) == 29
    for p in cells:
        js = json.loads(p.read_text())
        assert js["development_only"] is True
        assert set(("kind", "passed", "checks", "summary")) <= set(js["result"])
        assert js["solves_used"] == 0


# --- 3. scientific controls behave --------------------------------------------------------------
@needs_archive
def test_named_scientific_controls_behave(reduced):
    ctrl = reduced["controls"]
    assert set(ctrl) == {"block1_equals_iid", "zero_effect_identity", "event_weight_constant",
                         "empca_uniform_weight_limit", "null_dependence"}
    assert all(c["passed"] for c in ctrl.values()), {k: v["checks"] for k, v in ctrl.items() if not v["passed"]}
    # the cluster layer must add variance to every channel-fixed counterpart
    pairs = reduced["scientific_checks"]["cluster_layer_adds_variance"]
    assert len(pairs) == 5
    assert all(r["passed"] and r["ratio"] >= 2.0 for r in pairs.values())


def test_block_length_one_is_exactly_the_iid_bootstrap():
    r = AR.control_block1_equals_iid()
    assert r["block1_identical_to_iid"] and r["block64_differs_from_iid"]


def test_uniform_weights_reduce_empca_to_pca():
    r = RD.control_uniform_weight_limit(n_events=120, n_samples=48, rank=4, niter=25, tolerance=1e-3)
    assert r["passed"], r
    assert r["relative_difference_flat"] < 1e-3
    assert r["n_fits"] == 2 * RD.FITS_PER_CELL


# --- 4. seed namespaces -------------------------------------------------------------------------
def test_pilot_and_confirmation_seeds_are_disjoint(P):
    m = RU.build_matrix(P, False)
    pilot = {c["seed"] for c in m.cells if c["role"] == "pilot"}
    conf = {c["seed"] for c in m.cells if c["role"] == "confirmation"}
    dev = {c["seed"] for c in m.cells if c["role"] == "development"}
    assert pilot and conf and not (pilot & conf) and not dev
    lo, hi = P["seed_namespaces"]["pilot"]
    assert all(lo <= s <= hi for s in pilot)
    lo, hi = P["seed_namespaces"]["confirmation"]
    assert all(lo <= s <= hi for s in conf)
    # control and algebraic cells are seedless, so they can never borrow a development seed
    assert all(c["seed"] is None for c in m.cells if c["role"] in ("control", "algebraic"))
    fx = RU.build_matrix(P, True)
    dlo, dhi = P["seed_namespaces"]["development"]
    assert all(dlo <= c["seed"] <= dhi for c in fx.cells if c["role"] == "development")


def test_fixture_run_refuses_pilot_and_confirmation_cells(tmp_path):
    assert main(["run", "--out", str(tmp_path / "r"), "--fixture", "--roles", "pilot"], archive=False) == 6
    assert main(["run", "--out", str(tmp_path / "r2"), "--fixture", "--roles", "confirmation"], archive=False) == 6


def test_run_refuses_the_frozen_archive_roots():
    assert main(["run", "--out", str(REPO / "results" / "cresst_s_never")], archive=False) == 6
    assert not (REPO / "results" / "cresst_s_never").exists()


# --- 5. reduce refuses partial, duplicate and tampered runs --------------------------------------
@needs_archive
def test_reduce_refuses_a_partial_run(tmp_path, fx_run, P, fixture_matrix):
    out = tmp_path / "partial"
    shutil.copytree(fx_run, out)
    (out / "REDUCED.json").unlink()
    victim = _cell_id(fixture_matrix, analysis="calibration", scale="fixture", n_runs=4,
                      mc_replicates=120)
    (out / "cells" / f"{victim}.json").unlink()
    assert main(["reduce", "--out", str(out)]) == 3
    assert main(["reduce", "--out", str(out), "--allow-partial"]) == 0


@needs_archive
def test_reduce_refuses_a_duplicated_cell_file(tmp_path, fx_run):
    out = tmp_path / "dup"
    shutil.copytree(fx_run, out)
    src = sorted((out / "cells").glob("*.json"))[0]
    js = json.loads(src.read_text())
    js["cell_id"] = "deadbeefdeadbeef"
    (out / "cells" / "deadbeefdeadbeef.json").write_text(json.dumps(js))
    assert main(["reduce", "--out", str(out)]) == 3


@needs_archive
def test_reduce_refuses_a_tampered_cell_file(tmp_path, fx_run, fixture_matrix):
    out = tmp_path / "tamper"
    shutil.copytree(fx_run, out)
    victim = _cell_id(fixture_matrix, analysis="archive_point_estimates")
    p = out / "cells" / f"{victim}.json"
    js = json.loads(p.read_text())
    js["result"]["summary"]["point_estimates"]["theta_mean_of_channel_fractions"] = 0.42
    p.write_text(json.dumps(js))
    assert main(["reduce", "--out", str(out)]) == 3


# --- 6. an injected failure is charged and the retry accumulates ------------------------------------
@needs_archive
def test_injected_failure_is_charged_and_the_retry_accumulates(tmp_path, fixture_matrix):
    out = tmp_path / "fail"
    victim = _cell_id(fixture_matrix, analysis="science_control", control="block1_equals_iid", scale="fixture")
    assert main(["run", "--out", str(out), "--fixture", "--cells", victim, "--fail-cell", victim]) == 1
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=1400, package="cresst_s", protocol_sha256="x")
    ch = led.charged()
    assert ch["by_state"]["failed"] == 1 and ch["attempts"] == 1
    assert main(["run", "--out", str(out), "--fixture", "--cells", victim]) == 0
    ch = led.charged()
    assert ch["attempts"] == 2 and ch["by_state"]["failed"] == 1 and ch["by_state"]["completed"] == 1


# --- 7. the robustness grid fails cleanly without the raw data root ----------------------------------
def test_production_robustness_cell_fails_with_input_not_available(tmp_path, P):
    m = RU.build_matrix(P, False)
    victim = [c for c in m.cells if c["family"] == "robustness_confirm"][0]
    out = tmp_path / "nodata"
    rc = main(["run", "--out", str(out), "--cells", victim["cell_id"]], archive=False)
    assert rc == 1
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=1400, package="cresst_s", protocol_sha256="x")
    ch = led.charged()
    assert ch["by_state"]["failed"] == 1
    assert ch["fits"] == victim["n_fits"] == RD.FITS_PER_CELL   # the failed attempt stays charged
    rec = [r for r in led.records() if r.get("state") == "failed"][0]
    assert "input not available" in rec["error"]
    assert "<CRESST_DATA_ROOT>" in rec["error"]
    assert not (out / "cells").exists() or not list((out / "cells").glob("*.json"))


def test_missing_archive_root_is_an_input_not_available_failure():
    with pytest.raises(AR.InputNotAvailable) as e:
        AR.resolve_archive(None)
    assert "input not available" in str(e.value)
    assert "--archive-root" in str(e.value)


def test_fixture_data_root_has_the_production_interface(tmp_path):
    idx = RD.write_fixture_data_root(tmp_path / "root", seed=900000, n_events=40, n_noise=40, n_samples=32)
    assert idx["format"] == RD.ROOT_FORMAT and idx["synthetic"] is True
    ch = RD.load_pulse_channel(tmp_path / "root", "33", "36.0")
    assert ch["pulses"].shape == (40, 32) and ch["noise"].shape == (40, 32)
    with pytest.raises(AR.InputNotAvailable):
        RD.load_pulse_channel(tmp_path / "root", "99", "1.0")


# --- 8. resource estimate ------------------------------------------------------------------------
@needs_archive
def test_resource_estimate_from_fixture_timings(fx_run):
    assert main(["estimate", "--out", str(fx_run)]) == 0
    est = json.loads((fx_run / "RESOURCE_ESTIMATE.json").read_text())
    assert est["priced_matrix"] == "production" and est["timing_source_matrix"] == "fixture"
    assert est["unmeasured_families"] == []
    assert est["counts"]["cells"] == 395 and est["counts"]["fits"] == 1056
    # the fixture matrix is the production one plus the eight development cells
    assert est["fixture_counts"]["by_role"]["development"]["cells"] == 8
    assert est["fixture_counts"]["cells"] == est["counts"]["cells"] + 8
    assert est["total_cpu_seconds_point"] > 0
    assert est["retry_allowance"]["fits_headroom_under_cap"] == 1400 - 1056


# --- 9. the recomputation reproduces the archive ------------------------------------------------------
@needs_archive
def test_recomputation_reproduces_the_archived_point_estimates(reduced):
    cmp = reduced["archive_recomputation"]["comparison"]
    assert cmp["all_within_tolerance"] is True
    assert cmp["max_abs_deviation"] <= cmp["tolerance_absolute"] == 1e-12
    per = cmp["per_channel"]
    assert len(per) == 7
    for lab, row in per.items():
        assert row["within_tolerance"], (lab, row)
        # both archived sources agree, not just the re-run
        assert row["fraction_vs_channel_summary_csv"] <= 1e-12
    pe = reduced["archive_recomputation"]["point_estimates"]
    assert pe["n_channels"] == 7 and pe["n_runs"] == 2 and pe["n_evaluation_events"] == 19335
    assert reduced["archive_recomputation"]["archived_inputs"]["event_residuals_lossless"]["matches_protocol"]


@needs_archive
def test_recomputed_interval_endpoints_are_within_the_declared_tolerance(reduced):
    ai = reduced["archive_intervals"]
    assert ai["n_constructions"] == 10 and ai["all_within_tolerance"] is True
    for name, row in ai["constructions"].items():
        assert row["max_abs_deviation"] <= row["tolerance"], (name, row)


@needs_archive
def test_headline_interval_is_bit_identical_at_the_archived_replicate_count():
    """At the archived seed, stream order and replicate count the recomputation is exact."""
    resolved = AR.resolve_archive(ARCHIVE, roles=["event_residuals_lossless", "t13a_results"])
    labels, runs, clusters, times = AR.load_clusters(resolved)
    rec = AR.recompute_interval("headline", clusters, replicates=AR.ARCHIVE_REPLICATES)
    arch = AR.load_reference_results(resolved)["intervals"]["headline"]
    cmp = AR.compare_interval(rec, arch, replicates_archive=AR.ARCHIVE_REPLICATES,
                              z_target=4.0, shape_inflation=2.0)
    assert cmp["bit_identical_to_archive"] is True, cmp["endpoints"]
    assert cmp["max_abs_deviation"] == 0.0


# --- 10. run-stratified and leave-one-out match the archive --------------------------------------------
@needs_archive
def test_leave_one_out_and_run_stratified_match_the_archive(reduced):
    cr = reduced["cluster_report"]
    for report, key, n in (("leave_one_run_out", "dropped_run", 2),
                           ("leave_one_channel_out", "dropped_channel", 7)):
        cmp = cr[report]["comparison"]
        assert cmp["all_within_tolerance"] is True, cmp
        assert len(cmp["rows"]) == n
        assert cmp["max_point_abs_deviation"] <= 1e-12
    # run-stratified: run 33 has two channels, run 35 has five, and each equals the
    # archived leave-one-run-out estimate for the other run
    strata = {s["run"]: s for s in cr["run_stratified"]["strata"]}
    assert strata["33"]["n_channels"] == 2 and strata["35"]["n_channels"] == 5
    ident = reduced["scientific_checks"]["leave_one_run_out_equals_run_stratified"]
    assert ident["passed"] is True
    assert ident["per_dropped_run"]["35"]["stratified_theta"] == pytest.approx(0.1288378444742197, abs=1e-12)
    assert ident["per_dropped_run"]["33"]["stratified_theta"] == pytest.approx(0.2504296735646334, abs=1e-12)
    assert strata["33"]["cluster_interval_is_coarse"] is True


@needs_archive
def test_every_output_of_the_cluster_family_is_labelled_descriptive(reduced):
    cr = reduced["cluster_report"]
    assert "descriptive sensitivity of the archive" in cr["label"]
    for report in RU.CLUSTER_REPORTS:
        assert "descriptive sensitivity of the archive" in cr[report]["label"]
        assert "population inference" in cr[report]["label"]


# --- 11. the between-run component is not identifiable from two runs ------------------------------------
@needs_archive
def test_between_run_variance_is_reported_as_non_identifiable(reduced):
    vd = reduced["cluster_report"]["variance_decomposition"]
    br = vd["between_run"]
    assert vd["n_runs"] == 2
    assert br["identified"] is False and br["identifiable_from_this_design"] is False
    assert br["degrees_of_freedom"] == 1
    assert br["min_runs_required_for_a_reported_component"] == 4
    assert br["chi2_95_width_factor_at_this_df"] > 1000
    assert "not identifiable" in br["statement"].lower()
    # the components that ARE identifiable are reported and add up
    assert vd["within_channel"]["mean_bootstrap_variance"] > 0
    assert vd["between_channel"]["moment_estimate"] > 0
    total = vd["total_observed_dispersion_of_channel_fractions"]
    assert vd["between_channel"]["moment_estimate"] + vd["within_channel"]["mean_bootstrap_variance"] \
        == pytest.approx(total, rel=1e-9)


def test_the_gate_rejects_an_attributed_between_run_component():
    vd = {"n_runs": 2, "within_channel": {"mean_bootstrap_variance": 0.01},
          "between_channel": {"moment_estimate": 0.03},
          "between_run": {"n_runs": 2, "identified": True, "identifiable_from_this_design": False}}
    problems = CR.check_variance_decomposition(vd, min_runs=4)
    assert problems and "identified between-run component" in problems[0]
    vd["between_run"]["identified"] = False
    assert CR.check_variance_decomposition(vd, min_runs=4) == []


# --- 12. calibration ---------------------------------------------------------------------------------
@needs_archive
def test_calibration_coverage_is_lower_at_two_runs_than_at_eight(reduced):
    curves = reduced["calibration"]["coverage_curves"]
    key = "calib_confirm|C2|sd_run0.3"
    assert key in curves, sorted(curves)
    c = curves[key]
    assert c["n_runs_evaluated"] == [2, 4, 8]
    assert c["coverage_increases_with_runs"] is True
    # fixed seeds; a loose tolerance on a property of the procedure, not a tuned threshold
    assert c["coverage_gain_fewest_to_most_runs"] > 0.05
    assert c["coverage_at_fewest_runs"] < 0.95 - 0.05
    for n, row in c["by_n_runs"].items():
        w = row["coverage"]
        assert w["half_width"] is not None and w["lo"] <= w["rate"] <= w["hi"], n


def test_calibration_analytic_diagnostic_matches_its_derivation():
    d = CB.CalibrationDesign(n_runs=2, channels_per_run=5, n_events=50, sd_run=0.3, sd_channel=0.1,
                             theta=0.22, sigma_p=0.7, sigma_e=0.5, mc_replicates=1, bootstrap_replicates=1)
    a = CB.implied_variance_ratio(d)
    # true var = sd_run^2/G + sd_chan^2/(GC); E[s^2] = sd_chan^2 + sd_run^2 C(G-1)/(GC-1)
    assert a["true_variance_of_equal_channel_mean"] == pytest.approx(0.09 / 2 + 0.01 / 10)
    assert a["expected_bootstrap_implied_variance"] == pytest.approx((0.01 + 0.09 * 5 * 1 / 9) / 10)
    assert a["true_over_implied"] > 1.0


def test_wilson_interval_brackets_the_rate():
    w = CB.wilson(60, 100)
    assert w["lo"] < 0.6 < w["hi"] and w["n"] == 100 and w["half_width"] > 0
    assert CB.wilson(0, 0)["rate"] is None


# --- 13. mutation: injected incorrect scientific outputs are rejected -------------------------------------
@needs_archive
def test_mutation_events_only_bootstrap_called_hierarchical_is_rejected(tmp_path, monkeypatch, fixture_matrix):
    """Resample events but not channels, and still call the result hierarchical."""
    ids = [_cell_id(fixture_matrix, analysis="archive_interval", construction=c)
           for c in ("headline", "channel_fixed")]
    clean = tmp_path / "clean"
    assert main(["run", "--out", str(clean), "--fixture", "--cells", *ids]) == 0
    assert main(["reduce", "--out", str(clean), "--allow-partial"]) == 0, "the un-mutated pair must reduce"
    monkeypatch.setenv("CRESST_S_TEST_MUTATION", "events_only_hierarchical")
    bad = tmp_path / "mutated"
    assert main(["run", "--out", str(bad), "--fixture", "--cells", *ids]) == 0
    assert main(["reduce", "--out", str(bad), "--allow-partial"]) == 4
    js = json.loads((bad / "cells" / f"{ids[0]}.json").read_text())
    chk = js["result"]["checks"]["channel_layer_matches_its_label"]
    assert chk["passed"] is False
    assert chk["mean_distinct_channels_per_replicate"] == pytest.approx(7.0)
    # and the digest is intact: the rejection came from the science, not from tampering
    assert js["result"]["summary"]["mutation"] == "events_only_hierarchical"


@needs_archive
def test_mutation_between_run_variance_from_two_runs_is_rejected(tmp_path, monkeypatch, fixture_matrix):
    vid = _cell_id(fixture_matrix, analysis="cluster_report", report="variance_decomposition", scale="fixture")
    monkeypatch.setenv("CRESST_S_TEST_MUTATION", "between_run_from_two_runs")
    bad = tmp_path / "vd_mutated"
    assert main(["run", "--out", str(bad), "--fixture", "--cells", vid]) == 0
    assert main(["reduce", "--out", str(bad), "--allow-partial"]) == 4
    js = json.loads((bad / "cells" / f"{vid}.json").read_text())
    assert js["result"]["checks"]["between_run_component_not_reported_as_identified"]["passed"] is False
    assert js["result"]["summary"]["payload"]["between_run"]["identified"] is True


# --- protocol and design invariants -----------------------------------------------------------------
def test_protocol_pins_the_archived_streams_and_constants(P):
    assert list(P["archive"]["stream_order"])[:13] == list(
        __import__("experiments.mlst2026.t13a_cresst_channel_bootstrap", fromlist=["x"]).STREAM_ORDER)
    assert P["archive"]["seed"] == 20260713 and P["archive"]["replicates"] == 10000
    assert P["robustness"]["primary_setting"] == {"rank": 6, "covariance_model": "stationary_psd",
                                                  "shrinkage": 0.1, "time_cut": "full"}
    assert P["budget"]["estimator_fits_cap"] >= 1056
    assert "no cresst raw data is present" in P["provenance"]["no_cresst_raw_data_present"].lower()


def test_declared_scales_are_the_mechanically_derived_ones(P):
    RU._check_declared_scales(P)   # raises ProtocolError otherwise
    assert set(P["budget"]["fixture_to_production_scale"]) == set(RU.declared_scales(P))


def test_robustness_grid_is_paired_and_marked_unexecutable(P):
    rows = RD.grid_rows(P, "confirmation")
    assert len(rows) == 315
    assert all(r["executable_here"] is False for r in rows)
    plan = RD.contrast_plan(rows)
    assert plan["n_contrast_groups"] == 21
    assert plan["groups_without_a_reference_setting"] == []
    assert plan["n_paired_contrasts"] == 21 * 14
    assert plan["n_time_cut_contrasts"] == 14
    # every setting in a group shares one seed, so a paired contrast sees one event partition
    m = RU.build_matrix(P, False)
    groups: dict[str, set] = {}
    for c in m.cells:
        if c["family"] == "robustness_confirm":
            groups.setdefault(c["params"]["contrast_group"], set()).add(c["seed"])
    assert groups and all(len(s) == 1 for s in groups.values())
    assert len({next(iter(s)) for s in groups.values()}) == 21


# --- the fit-accounting guard is a real check ------------------------------------------------
def _ledger_records(out):
    return [json.loads(l) for l in (out / "EXECUTION_LEDGER.jsonl").read_text().splitlines() if l.strip()]


def test_fit_accounting_guard_can_fail(tmp_path, monkeypatch, P):
    """The estimator control reports the ``classic_pca``/``empca`` invocations
    instrumented in ``robustness_design.fit_and_score``.  Declare one more in the
    matrix and the run is refused (exit 4), charged, with no cell file written."""
    real = RU.build_matrix
    target = next(c for c in real(P, True).cells
                  if c["params"].get("control") == "empca_uniform_weight_limit")
    assert target["n_fits"] == 2 * RD.FITS_PER_CELL

    def misdeclared(P_, fixture):
        m = real(P_, fixture)
        for c in m.cells:
            if c["cell_id"] == target["cell_id"]:
                c["n_fits"] = int(c["n_fits"]) + 1
        return m

    monkeypatch.setattr(RU, "build_matrix", misdeclared)
    out = tmp_path / "misdeclared"
    assert main(["run", "--out", str(out), "--fixture", "--cells", target["cell_id"]],
                archive=False) == 4
    recs = _ledger_records(out)
    assert recs[-1]["state"] == "failed" and "accounting mismatch" in recs[-1]["error"]
    assert not list((out / "cells").glob("*.json"))


def test_estimator_control_reports_the_instrumented_fit_count(tmp_path, P):
    target = next(c for c in RU.build_matrix(P, True).cells
                  if c["params"].get("control") == "empca_uniform_weight_limit")
    out = tmp_path / "counted"
    assert main(["run", "--out", str(out), "--fixture", "--cells", target["cell_id"]],
                archive=False) == 0
    js = json.loads((out / "cells" / f"{target['cell_id']}.json").read_text())
    assert js["fits_used"] == js["fits_observed"]["total"] == 2 * RD.FITS_PER_CELL
    assert js["fits_observed"]["by_label"] == {"classic_pca": 2,
                                               "empca_restart": 2 * RD.EMPCA_RESTARTS}
