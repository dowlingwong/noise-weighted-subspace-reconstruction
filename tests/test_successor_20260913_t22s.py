"""T22-S (L4) local acceptance tests: the eight README tests plus the package's
own scientific checks, all on synthetic fixtures.  No instrument file exists in
this environment and none is opened; the matched interface is exercised on a
synthetic HDF5 file with the TIDMAD dataset layout read through the archived
opener under an exposure allowance."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.mlst2026 import t22_tidmad_narrowband as t22  # noqa: E402
from experiments.mlst2026 import tidmad_endpoint as te  # noqa: E402
from experiments.mlst2026.successor_20260913.common import ledger as L, protocol as PR, results as R  # noqa: E402
from experiments.mlst2026.successor_20260913.t22s import matched_stage as ms  # noqa: E402
from experiments.mlst2026.successor_20260913.t22s import pipeline as pl  # noqa: E402
from experiments.mlst2026.successor_20260913.t22s import runner as RN  # noqa: E402
from experiments.mlst2026.successor_20260913.t22s import simulator as sim  # noqa: E402

PROTO = RN.PROTOCOL_DEFAULT
PILOT_FAMILIES = ["--families", "sim_pilot", "control", "matched_fixture"]


def main(*argv):
    return RN.main([argv[0], "--protocol", str(PROTO)] + list(argv[1:]))


def _cells(out):
    return {k: v[0][1] for k, v in R.read_cells(out).items()}


@pytest.fixture(scope="module")
def pilot_root(tmp_path_factory):
    """Pilot fixture run + reduce (writes FROZEN_CALIBRATION.json); shared read-only."""
    out = tmp_path_factory.mktemp("t22s") / "pilot"
    assert main("run", "--out", str(out), "--fixture", *PILOT_FAMILIES) == 0
    assert main("reduce", "--out", str(out), "--allow-partial") == 0
    return out


@pytest.fixture(scope="module")
def full_root(tmp_path_factory, pilot_root):
    """Pilot + confirmation fixture run, fully reduced."""
    out = tmp_path_factory.mktemp("t22s") / "full"
    shutil.copytree(pilot_root, out)
    assert main("run", "--out", str(out), "--fixture", "--families", "sim_confirm") == 0
    assert main("reduce", "--out", str(out)) == 0
    return out


def _copy(src, tmp_path, name):
    dst = tmp_path / name
    shutil.copytree(src, dst)
    return dst


# --- 1. validate / enumerate compute nothing -----------------------------------------
def test_validate_and_enumerate_compute_nothing(tmp_path):
    assert main("validate") == 0
    out = tmp_path / "enum"
    assert main("enumerate", "--out", str(out)) == 0
    assert (out / "run_matrix.jsonl").exists() and not (out / "cells").exists()
    summ = json.loads((out / "run_matrix_summary.json").read_text())
    assert summ["fits"] == 0 and summ["by_role"]["pilot"]["cells"] == 32
    assert summ["by_role"]["confirmation"]["cells"] == 69 * 10
    assert summ["by_family"]["sim_confirm"]["cells"] == 690


def test_protocol_is_v2_with_provenance_and_fixed_tolerances():
    P = PR.load_protocol(PROTO)
    assert P["schema"] == PR.SCHEMA and P["package"] == "t22s"
    assert "no_prior_t22s_protocol" in P["provenance"] and "12 September" in P["provenance"]["no_prior_t22s_protocol"]
    assert P["calibration_tolerances"]["coverage"] == [0.925, 0.975] and P["calibration_tolerances"]["type1"] == [0.0, 0.07]
    assert P["seed_namespaces"] == {"development": [900000, 900999], "pilot": [7000, 7199], "confirmation": [8000, 8399]}
    assert P["budget"]["estimator_fits_cap"] == 64
    # the Monte-Carlo count meets the declared Wilson requirement
    assert RN.wilson(50, P["matrix"]["production"]["mc_replicates_per_scenario"])["half_width"] <= 0.015
    assert RN.wilson_n_for_half_width(0.05, 0.015) <= P["matrix"]["production"]["mc_replicates_per_scenario"]


# --- 2. fixture end to end with the declared schema --------------------------------------
def test_fixture_end_to_end_has_declared_schema(full_root):
    red = json.loads((full_root / "REDUCED.json").read_text())
    assert red["development_only"] is True and red["partial"] is False and red["reconciliation"]["complete"]
    for key in ("pilot", "frozen_calibration", "confirmation", "controls", "matched_fixture", "statement", "tolerances"):
        assert key in red
    assert "cannot establish the adequacy" in red["statement"]
    conf = red["confirmation"]
    assert conf["complete"] and set(conf["scenarios"]) == set(s.name for s in sim.development_scenarios())
    s = next(iter(conf["scenarios"].values()))
    for key in ("withholding", "contrasts", "claims", "any_supported", "dependence_verdicts"):
        assert key in s
    c = s["contrasts"]["raw-bin|empca"]
    for key in ("coverage_two_stage", "coverage_two_stage_given_not_withheld", "coverage_t_interval", "mean_length_two_stage",
                "mean_length_t_interval", "bias_mean", "bias_sign", "coverage_within_tolerance"):
        assert key in c
    assert all("supported_given_not_withheld" in v for v in s["claims"].values())
    for js in _cells(full_root).values():
        R.check_cell_schema(js, RN.result_required)
        assert js["development_only"] and 900000 <= (js["seed"] or 900000) <= 900999


# --- 3. scientific controls ----------------------------------------------------------------
def test_controls_behave_as_declared(pilot_root):
    red = json.loads((pilot_root / "REDUCED.json").read_text())
    ctrl = red["controls"]
    assert set(ctrl) == {"simulator_estimand", "ar1_autocorrelation", "iid_bootstrap_se", "dependence_gate", "holm_arithmetic"}
    assert all(v["passed"] for v in ctrl.values()), {k: v["checks"] for k, v in ctrl.items() if not v["passed"]}
    gate = ctrl["dependence_gate"]["checks"]
    assert set(gate["gate_withholds_every_claim"]["verdicts"].values()) == {"inconclusive"}
    assert gate["positive_control_supported_when_stabilised"]["verdicts"]["denoising_benefit"] == "supported"
    m = red["matched_fixture"][0]
    assert m["passed"] and m["fits_used"] == RN.MATCHED_FITS and m["exposure_status"] == "synthetic-fixture"
    assert "already exposed data must be labelled" in m["label"]


def test_null_scenario_phi0_n16_coverage_near_nominal():
    sc = sim.make_scenario("null-partial", 16, 0.0)
    recs = [pl.run_replicate(900200 * 10_000 + j, sc, file_replicates=80, hier_replicates=400, curve_replicates=60)
            for j in range(30)]
    cov = np.mean([r["contrasts"]["raw-bin|empca"]["covers"] for r in recs])
    cov_pca = np.mean([r["contrasts"]["pca|empca"]["covers"] for r in recs])
    assert cov >= 0.80 and cov_pca >= 0.80, (cov, cov_pca)   # 30 replicates: >= 24 hits (nominal 0.95)
    # the denoising claim is null here: it must not be 'supported' more than rarely
    assert np.mean([r["verdicts"]["denoising_benefit"]["verdict"] == "supported" for r in recs]) <= 0.2
    assert all(not pl.check_record(r, sc) for r in recs)
    assert all(abs(r["contrasts"]["raw-bin|empca"]["true_mu"]) == 0.0 for r in recs)


def test_strongly_dependent_scenario_is_withheld_regardless_of_effect():
    sc = sim.make_scenario("alt-large", 4, 0.6, drift_sd=0.3)
    for j in range(3):
        r = pl.run_replicate(900300 * 10_000 + j, sc, file_replicates=40, hier_replicates=200, curve_replicates=40)
        assert r["dependence"]["verdict"] == "inconclusive" and r["withheld"]
        assert all(v["verdict"] == "inconclusive" for v in r["verdicts"].values())
        # the declared effect is the largest in the catalogue: the withholding is the gate, not the effect
        assert r["contrasts"]["raw-bin|empca"]["true_mu"] == sim.EFFECTS["alt-large"][0] * 4.0 / sc.w > 0


def test_simulator_estimand_is_exact_conditional_on_files():
    sc = sim.make_scenario("alt-medium", 8, 0.3, tau=0.0)
    rng = np.random.default_rng(3)
    d, Ns = [], []
    for i in range(40):
        f = sim.simulate_file(rng, sc, f"f{i}", sc.lam_emp)
        Ns.append(f.N)
        d.append(te.squared_errors(f.eval_read, f.truth_eval, sc.w) - te.squared_errors(f.generated_estimates["empca"], f.truth_eval, sc.w))
    d = np.concatenate(d)
    target = sc.true_mu()["raw-bin|empca"] * np.mean(Ns)
    assert abs(d.mean() - target) <= 4 * d.std(ddof=1) / np.sqrt(d.size)


# --- 4. namespaces and fixture refusal ----------------------------------------------------
def test_namespaces_disjoint_and_fixture_refuses_pilot_confirmation(tmp_path):
    P = PR.load_protocol(PROTO)
    m = RN.build_matrix(P, fixture=False)
    seeds = {}
    for c in m.cells:
        if c["seed"] is not None:
            seeds.setdefault(c["role"], set()).add(c["seed"])
    assert seeds["pilot"] <= set(range(7000, 7200)) and seeds["confirmation"] <= set(range(8000, 8400))
    assert not (seeds["pilot"] & seeds["confirmation"])
    mf = RN.build_matrix(P, fixture=True)
    dev = {c["seed"] for c in mf.cells if c["role"] == "development"}
    assert dev and dev <= set(range(900000, 901000))
    out = tmp_path / "refuse"
    (out).mkdir()
    (out / RN.FROZEN_NAME).write_text("{}")   # so the refusal below is the role check, not the freeze check
    assert main("run", "--out", str(out), "--fixture", "--roles", "confirmation") == 6
    assert main("run", "--out", str(out), "--fixture", "--roles", "pilot") == 6
    assert not (out / "cells").exists()


# --- 5. reduce refusals -------------------------------------------------------------------
def test_reduce_refuses_partial_duplicate_and_tampered(pilot_root, full_root, tmp_path):
    assert main("reduce", "--out", str(pilot_root)) == 3          # confirmation development cells missing
    dup = _copy(full_root, tmp_path, "dup")
    p = sorted((dup / "cells").glob("*.json"))[0]
    js = json.loads(p.read_text())
    js["cell_id"] = "deadbeefdeadbeef"
    (dup / "cells" / "deadbeefdeadbeef.json").write_text(json.dumps(js))
    assert main("reduce", "--out", str(dup)) == 3
    tam = _copy(full_root, tmp_path, "tamper")
    p = [q for q in sorted((tam / "cells").glob("*.json")) if json.loads(q.read_text())["family"] == "sim_confirm"][0]
    js = json.loads(p.read_text())
    js["result"]["records"][0]["contrasts"]["raw-bin|empca"]["ci95_hi"] += 1.0
    p.write_text(json.dumps(js))
    assert main("reduce", "--out", str(tam)) == 3


def test_frozen_calibration_is_deterministic_and_never_overwritten(pilot_root, tmp_path):
    fz = json.loads((pilot_root / RN.FROZEN_NAME).read_text())
    assert fz["schema"] == RN.FROZEN_SCHEMA and fz["development_only"] is True
    assert fz["tolerances"] == PR.load_protocol(PROTO)["calibration_tolerances"]
    assert fz["block_grid_evaluation"] == list(pl.BLOCKS_EVAL)
    again = _copy(pilot_root, tmp_path, "again")
    assert main("reduce", "--out", str(again), "--allow-partial") == 0
    assert json.loads((again / RN.FROZEN_NAME).read_text()) == fz
    edited = _copy(pilot_root, tmp_path, "edited")
    fz2 = dict(fz, setting="archived", replicates={"file": 1, "hier": 1, "curve": 1})
    (edited / RN.FROZEN_NAME).write_text(json.dumps(fz2))
    assert main("reduce", "--out", str(edited), "--allow-partial") == 3


# --- 6. injected failure ------------------------------------------------------------------
def test_injected_failure_is_charged_and_retry_accumulates(tmp_path):
    P = PR.load_protocol(PROTO)
    m = RN.build_matrix(P, fixture=True)
    victim = [c for c in m.cells if c["family"] == "control" and c["params"]["control"] == "holm_arithmetic"][0]["cell_id"]
    out = tmp_path / "fail"
    assert main("run", "--out", str(out), "--fixture", "--families", "control", "--fail-cell", victim) == 1
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=64, package="t22s", protocol_sha256=P["_sha256"])
    ch = led.charged()
    assert ch["by_state"]["failed"] == 1
    assert main("run", "--out", str(out), "--fixture", "--families", "control") == 0
    ch = led.charged()
    assert ch["by_state"]["failed"] == 1 and ch["by_state"]["completed"] == 5 and ch["attempts"] == 6


# --- 7. mutation: rejected by the scientific checks -------------------------------------------
def test_mutation_dropping_the_dependence_gate_is_rejected(pilot_root, tmp_path, monkeypatch):
    P = PR.load_protocol(PROTO)
    m = RN.build_matrix(P, fixture=True)
    drift = [c for c in m.cells if c["family"] == "sim_confirm" and c["params"]["scenario"]["drift_sd"] > 0][0]
    out = _copy(pilot_root, tmp_path, "mut")
    monkeypatch.setenv("T22S_TEST_MUTATION", "drop_dependence_gate")
    assert main("run", "--out", str(out), "--fixture", "--cells", drift["cell_id"]) == 0   # the cell itself completes
    monkeypatch.delenv("T22S_TEST_MUTATION")
    js = _cells(out)[drift["cell_id"]]
    sc = sim.Scenario(**drift["params"]["scenario"])
    probs = [p for rec in js["result"]["records"] for p in pl.check_record(rec, sc)]
    assert probs and any("gate dropped" in p for p in probs)
    assert js["result"]["passed"] is False
    assert main("reduce", "--out", str(out), "--allow-partial") == 4   # the digest is intact; the science is not


def test_mutation_files_only_bootstrap_is_rejected_by_the_variance_identity():
    sc = sim.make_scenario("alt-medium", 4, 0.6, tau=0.0)
    reps = {"file": 60, "hier": 3000, "curve": 40}
    clean = RN._replicate_with_mutation(900400 * 10_000, sc, reps, None)
    assert not pl.check_record(clean, sc)
    mutated = RN._replicate_with_mutation(900400 * 10_000, sc, reps, "files_only_bootstrap")
    probs = pl.check_record(mutated, sc)
    assert probs and any("two-stage bootstrap variance" in p for p in probs), probs


def test_record_checks_catch_a_wrong_sign_and_a_wrong_coverage_flag():
    sc = sim.make_scenario("alt-large", 4, 0.0)
    rec = pl.run_replicate(900500 * 10_000, sc, file_replicates=40, hier_replicates=200, curve_replicates=30)
    assert not pl.check_record(rec, sc)
    bad = json.loads(json.dumps(rec))
    bad["contrasts"]["raw-bin|empca"]["covers"] = not bad["contrasts"]["raw-bin|empca"]["covers"]
    assert any("coverage indicator" in p for p in pl.check_record(bad, sc))
    bad = json.loads(json.dumps(rec))
    bad["contrasts"]["raw-bin|empca"]["true_mu"] *= -1.0
    assert any("estimand" in p for p in pl.check_record(bad, sc))


# --- 8. resource estimate ---------------------------------------------------------------------
def test_resource_estimate_from_fixture_timings(full_root):
    assert main("estimate", "--out", str(full_root)) == 0
    est = json.loads((full_root / "RESOURCE_ESTIMATE.json").read_text())
    assert est["unmeasured_families"] == []
    assert est["families"]["sim_confirm"]["measured"] and est["total_cpu_seconds_point"] > 0
    assert est["retry_allowance"]["fits_headroom_under_cap"] == 64  # production matrix has no estimator fits; fixture counts are recorded separately
    assert est["priced_matrix"] == "production" and est["fixture_counts"]["fits"] == RN.MATCHED_FITS
    P = PR.load_protocol(PROTO)
    assert P["budget"]["fixture_to_production_scale"] == RN.declared_scales(P)


# --- validation refused without the freeze ---------------------------------------------------
def test_validation_refused_without_frozen_calibration(tmp_path):
    out = tmp_path / "nofreeze"
    assert main("run", "--out", str(out), "--fixture", "--families", "sim_confirm") == 6
    assert main("run", "--out", str(out), "--fixture") == 6
    assert not out.exists() or not (out / "cells").exists()
    # the in-cell guard (defence in depth): a confirmation-type cell selected by id fails and is charged
    P = PR.load_protocol(PROTO)
    m = RN.build_matrix(P, fixture=True)
    cell = [c for c in m.cells if c["family"] == "sim_confirm"][0]
    from experiments.mlst2026.successor_20260913.common import cli
    rc = cli.main_template(package="t22s", description="t", build_matrix=RN.build_matrix, run_cell=RN.run_cell, reduce_run=RN.reduce_run,
                           result_required=RN.result_required, source_files=RN.SOURCE_FILES,
                           argv=["run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--cells", cell["cell_id"]])
    assert rc == 1   # the command-line pre-check is bypassed here on purpose; the cell itself refuses
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=64, package="t22s", protocol_sha256=P["_sha256"])
    atts = list(led.attempts().values())
    assert len(atts) == 1 and atts[0]["state"] == "failed" and "validation refused" in atts[0]["error"]


# --- part B: matched interface, feasibility, inventory -------------------------------------------
def test_matched_scoring_uses_identical_windows_and_labels(tmp_path):
    cfg = RN.MATCHED_FIXTURE
    path = tmp_path / "abra_fixture_0001.h5"
    ms.write_tidmad_fixture(path, seed=11, w=cfg["w"], n_cal=cfg["n_cal"], n_fit=cfg["n_fit"], n_eval=cfg["n_eval"],
                            guard=cfg["guard"], k_inj=cfg["k_inj"])
    import h5py
    with h5py.File(path, "r") as h:
        assert h[t22.SQUID].dtype == np.float32 and h[t22.REFERENCE].dtype == np.float32
        assert h[t22.SQUID].shape[0] > 200_000
    alloc = ms.fixture_allocation(w=cfg["w"], n_cal=cfg["n_cal"], n_fit=cfg["n_fit"], n_eval=cfg["n_eval"], guard=cfg["guard"])
    st = ms.load_fixture_stretch(path, 0, alloc)      # archived open_instrument_file/load_stretch under an allowance
    assert st["eval_read"].shape == (cfg["n_eval"], cfg["w"] // 2 + 1)
    k = cfg["k_inj"]
    N = te.mean_bin_power(st["fit_read"])
    N[k] = te.offline_noise_floor(st["fit_read"], k)
    out = ms.matched_scores(st, k_inj=k, B=9, w=cfg["w"], N_bins=N, exposure_status="synthetic-fixture")
    pr = out["per_representation"]
    assert pr["whole-window"]["evaluation_window_index"] == pr["band"]["evaluation_window_index"] == list(range(cfg["n_eval"]))
    assert pr["whole-window"]["label_sha256"] == pr["band"]["label_sha256"] == out["shared"]["label_sha256"]
    across = out["whole_window_vs_band_same_method"]
    assert out["shared_methods_identical_across_representations"]
    assert all(across[m]["max_abs_per_window"] == 0.0 for m in ms.SHARED_METHODS)
    assert any(across[m]["max_abs_per_window"] > 0.0 for m in ms.LEARNED)     # representation is the only difference
    assert out["band_selected_here"] is False and out["n_fits"] == 16
    assert "already exposed data must be labelled" in out["label"]
    with pytest.raises(ValueError):
        ms.matched_scores(st, k_inj=k, B=9, w=cfg["w"], N_bins=N, exposure_status="whatever")
    # the archived exposure guard refuses a region the allowance does not declare
    entry = {"dataset_id": path.name, "region": {"start_sample": 0, "n_samples": alloc.stretch}}
    with t22.exposure_allowance([entry]):
        with pytest.raises(SystemExit):
            t22.load_stretch(path, 4096, alloc)
    with t22.exposure_allowance([{"dataset_id": "abra_validation_0012.h5", "region": {"start_sample": 0, "n_samples": 10}}]):
        with pytest.raises(SystemExit):
            t22.load_stretch(path, 0, alloc)


def test_feasibility_failure_below_minimum_and_on_unstable_dependence():
    rules = PR.load_protocol(PROTO)["inference"]["feasibility"]
    r = ms.feasibility(None, 3, rules=rules)
    assert not r and r["status"] == "feasibility-failure" and any("too few independent units" in x for x in r["reasons"])
    good_table = {"4": {"type1_within_tolerance": True, "coverage_within_tolerance": True, "withholding_rate_max": 0.2,
                        "power_medium_effect_min": 0.6}}
    calib = {"complete": True, "feasibility_table": good_table}
    ok = ms.feasibility(calib, 4, rules=rules, dependence_verdict="stabilised")
    assert ok and ok["feasible"] and ok["reasons"] == []
    bad = ms.feasibility(calib, 4, rules=rules, dependence_verdict="inconclusive")
    assert not bad and any("dependence assessment unstable" in x for x in bad["reasons"])
    wh = ms.feasibility({"complete": True, "feasibility_table": {"4": dict(good_table["4"], withholding_rate_max=0.9)}}, 4, rules=rules)
    assert not wh and any("withholding rate" in x for x in wh["reasons"])
    assert "labelled accordingly" in wh["label_required"]


def test_exposure_inventory_reads_the_archive_and_declares_what_it_cannot_derive():
    inv = ms.exposure_inventory()
    assert inv["release_file_list"]["derivable_from_archive"] is False
    assert inv["release_file_list"]["files_not_yet_exposed"] is None
    hold = [r for r in inv["regions"] if r["file"] == "abra_validation_0015.h5" and r["stage"] == "final-test" and r["run_tree"] == "t22_2026-09-11"]
    assert hold and hold[0]["start_sample"] == 37027840 and hold[0]["n_samples"] == 46825472
    t10 = [r for r in inv["regions"] if r["file"] == "abra_validation_0015.h5" and r["stage"] == "final-test" and r["run_tree"] == "mlst_2026-09-10"]
    assert t10 and t10[0]["start_sample"] == 0 and t10[0]["n_samples"] == 36995072
    assert "abra_science_0000.h5" in inv["files_known_to_archive"] and len(inv["files_known_to_archive"]) == 22
    pf = inv["per_file"]["abra_validation_0015.h5"]
    assert pf["exposed_regions_merged"][0] == (0, 36995072) and pf["never_read_regions"][-1][1] == pf["n_total_samples"]
    md = ms.inventory_markdown(inv)
    assert "not an independent unit" in md and "unresolved input" in md
    committed = REPO / "experiments" / "mlst2026" / "successor_20260913" / "t22s" / "EXPOSURE_INVENTORY.md"
    assert committed.exists() and committed.read_text() == md
