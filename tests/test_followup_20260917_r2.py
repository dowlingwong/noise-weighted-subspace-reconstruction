"""R2 (T20-S v3, theta=0.12 scaling and crossover) package tests.

Everything here runs in the development seed namespace and is exercised on the
fixture protocol block (d = 12).  No test is a scientific result and no fixture
output is citable.

Covered, as the R2 package requires:

* validate/enumerate compute nothing;
* a fixture end-to-end run and reduce with the declared schema, the two
  endpoints and the declared multiplicity rule;
* the seed namespaces are disjoint and a fixture run refuses pilot/confirmation
  seeds;
* a confirmation reduce refuses without the frozen model;
* a scientific mutation test catching a double-applied ``a_ref`` offset (the
  September reporting trap), enforced at reduce with exit 4;
* the 2.7%-vs-``c/N^alpha`` old selection margin distinction;
* the resource estimate, honestly labelled (no BENCHMARK.json is shipped);
* the optional costed four-geometry expansion is declared and costed but not
  required.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.mlst2026.followup_20260917.r2_t20_theta012 import analysis as AN  # noqa: E402
from experiments.mlst2026.followup_20260917.r2_t20_theta012 import runner as RN  # noqa: E402
from experiments.mlst2026.successor_20260913.common import ledger as L  # noqa: E402
from experiments.mlst2026.successor_20260913.common import matrix as M  # noqa: E402
from experiments.mlst2026.successor_20260913.common import protocol as PR  # noqa: E402
from experiments.mlst2026.successor_20260913.common import results as R  # noqa: E402

PKG = REPO / "experiments" / "mlst2026" / "followup_20260917" / "r2_t20_theta012"
PROTO = PKG / "PROTOCOL_R2_T20_THETA012_v1.json"


def _main(argv):
    return RN.main([argv[0], "--protocol", str(PROTO)] + argv[1:])


@pytest.fixture(scope="module")
def P():
    return PR.load_protocol(PROTO)


def _pilot_fixture(out: Path):
    assert _main(["run", "--out", str(out), "--fixture", "--families", "algebraic", "pilot", "control"]) == 0
    assert _main(["reduce", "--out", str(out), "--allow-partial"]) == 0
    assert (out / RN.FROZEN_NAME).exists()


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("r2_full")
    _pilot_fixture(out)
    assert _main(["run", "--out", str(out), "--fixture", "--families", "confirmation"]) == 0
    assert _main(["reduce", "--out", str(out)]) == 0
    return out


# ---------------------------------------------------------------------------------------
# validate / enumerate compute nothing
# ---------------------------------------------------------------------------------------
def test_validate_and_enumerate_compute_nothing(tmp_path, capsys):
    assert _main(["validate"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert js["counts"]["cells"] == 966 and js["counts"]["fits"] == 4800 and js["counts"]["solves"] == 105
    assert js["counts"]["by_role"]["pilot"]["cells"] == 384
    assert js["counts"]["by_role"]["confirmation"]["cells"] == 576
    assert js["counts"]["by_role"]["algebraic"]["cells"] == 3
    assert js["counts"]["by_role"]["control"]["cells"] == 3
    assert js["fits_within_cap"] and js["cap"] == 7000
    out = tmp_path / "enum"
    assert _main(["enumerate", "--out", str(out)]) == 0
    assert (out / "run_matrix.jsonl").exists() and not (out / "cells").exists()
    rows = [json.loads(l) for l in (out / "run_matrix.jsonl").read_text().splitlines()]
    fit_rows = [r for r in rows if r["params"].get("kind") == "fit"]
    assert all(r["n_fits"] == 1 + r["params"]["R"] == 5 for r in fit_rows)
    assert sorted({r["params"]["N"] for r in fit_rows}) == [400, 800, 1000, 1250, 1600, 3200, 6400, 12800]
    assert sorted({r["params"]["theta"] for r in fit_rows}) == [0.0, 0.12, 0.25]


def test_module_entry_never_computes():
    import os
    cp = subprocess.run([sys.executable, "-m", "experiments.mlst2026.followup_20260917.r2_t20_theta012",
                         "validate", "--protocol", str(PROTO)], cwd=str(REPO),
                        env=dict(os.environ, PYTHONPATH=str(REPO)), capture_output=True, text=True, timeout=120)
    assert cp.returncode == 0, cp.stderr
    assert json.loads(cp.stdout)["package"] == "r2_t20_theta012"


# ---------------------------------------------------------------------------------------
# fixture end to end with the declared schema and the two endpoints
# ---------------------------------------------------------------------------------------
def test_fixture_end_to_end_schema(full_run, P):
    red = json.loads((full_run / "REDUCED.json").read_text())
    assert red["development_only"] is True and red["fixture"] is True
    assert red["phase"] == "confirmation" and red["confirmation_complete"] and red["pilot_complete"]
    assert red["reconciliation"]["complete"]
    for key in ("invariants", "controls", "grid", "confirmation", "frozen_choice", "excluded_thetas",
                "nonseparability_moves", "old_selection_margins", "offset_convention"):
        assert key in red, key
    assert red["grid"]["old_N"] == [400, 800, 1000, 1250, 1600, 3200]
    assert red["grid"]["extended_N"] == [6400, 12800]
    cells = R.read_cells(full_run)
    kinds = {}
    for files in cells.values():
        for _, js in files:
            R.check_cell_schema(js, RN.RESULT_REQUIRED)
            assert js["development_only"]
            kinds[js["result"]["kind"]] = kinds.get(js["result"]["kind"], 0) + 1
    assert kinds == {"algebraic": 2, "fit": 64, "control": 3}
    conf = red["confirmation"]
    assert "endpoint_a_old_prediction" in conf and "endpoint_b_new_model" in conf
    assert conf["multiplicity"]["all_of_rule"]
    assert conf["multiplicity"]["m_claims"] >= 1
    for th, ea in conf["endpoint_a_old_prediction"].items():
        assert "extrapolated_flags" in ea and ea["old_grid_max"] == 3200
        assert sum(ea["extrapolated_flags"]) == 2                 # N = 6400, 12800
    # every fit cell records x_U, the contrast, the a interval, restart spread and unresolved flags
    for files in cells.values():
        for _, js in files:
            r = js["result"]
            if r["kind"] != "fit":
                continue
            for k in ("d", "x_U", "a_lower_bound", "a_best_found", "a_interval", "restart_spread_risk",
                      "optimization_unresolved", "identity_residual"):
                assert k in r, k
            assert abs(r["identity_residual"]) < 1e-9 and r["x_U"] >= -1e-9
            assert r["selected_restart"] == int(np.argmin(r["restart_train_losses"]))
    # reduce is deterministic
    before = (full_run / "REDUCED.json").read_bytes()
    assert _main(["reduce", "--out", str(full_run)]) == 0
    assert (full_run / "REDUCED.json").read_bytes() == before


def test_control_cells_passed(full_run):
    for files in R.read_cells(full_run).values():
        for _, js in files:
            if js["result"]["kind"] == "control":
                assert js["result"]["passed"] is True, js["result"]["checks"]


# ---------------------------------------------------------------------------------------
# namespaces disjoint; a fixture run refuses pilot/confirmation seeds
# ---------------------------------------------------------------------------------------
def test_namespaces_disjoint_and_fixture_refuses_pilot_confirmation(P, tmp_path):
    m = RN.build_matrix(P, False)
    seeds = {role: {c["seed"] for c in m.cells if c["role"] == role} for role in ("pilot", "confirmation")}
    assert seeds["pilot"] and seeds["confirmation"] and not (seeds["pilot"] & seeds["confirmation"])
    assert all(PR.namespace_of_seed(P, s) == "pilot" for s in seeds["pilot"])
    assert all(PR.namespace_of_seed(P, s) == "confirmation" for s in seeds["confirmation"])
    assert M.check_seed_roles(m, P) == [] and M.check_seed_roles(RN.build_matrix(P, True), P) == []
    # fresh namespaces, disjoint from the old t20s and from R1
    assert P["seed_namespaces"]["development"] == [921000, 921999]
    assert P["seed_namespaces"]["pilot"] == [23000, 23299]
    assert P["seed_namespaces"]["confirmation"] == [24000, 24499]
    fx = RN.build_matrix(P, True)
    assert {c["role"] for c in fx.cells} == {"development", "algebraic", "control"}
    assert all(PR.namespace_of_seed(P, c["seed"]) == "development" for c in fx.cells if c["seed"] is not None)
    assert _main(["run", "--out", str(tmp_path / "a"), "--fixture", "--roles", "pilot"]) == 6
    assert _main(["run", "--out", str(tmp_path / "b"), "--fixture", "--roles", "confirmation"]) == 6
    assert not (tmp_path / "a" / "cells").exists()


# ---------------------------------------------------------------------------------------
# confirmation refuses without the frozen model
# ---------------------------------------------------------------------------------------
def test_confirmation_refused_without_freeze(tmp_path, P):
    out = tmp_path / "gate"
    assert _main(["run", "--out", str(out), "--fixture", "--families", "algebraic", "pilot", "control"]) == 0
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=P["budget"]["estimator_fits_cap"],
                            package=RN.PACKAGE, protocol_sha256=P["_sha256"])
    before = led.charged()["attempts"]
    # no freeze yet: the confirmation run is refused BEFORE any attempt is charged
    assert _main(["run", "--out", str(out), "--fixture", "--families", "confirmation"]) == 6
    assert led.charged()["attempts"] == before
    assert not any(json.loads(p.read_text())["params"].get("phase") == "confirmation"
                   for p in (out / "cells").glob("*.json"))
    # the pilot reduce writes the freeze; then confirmation runs; removing the freeze refuses the reduce
    assert _main(["reduce", "--out", str(out), "--allow-partial"]) == 0
    fz = json.loads((out / RN.FROZEN_NAME).read_text())
    assert fz["schema"] == RN.FROZEN_SCHEMA and fz["protocol_sha256"] == P["_sha256"]
    assert fz["development_only"] is True and fz["confirmation_cells_present_at_freeze"] == 0
    assert _main(["run", "--out", str(out), "--fixture", "--families", "confirmation"]) == 0
    moved = out / "FROZEN_MODEL.json.moved"
    (out / RN.FROZEN_NAME).rename(moved)
    assert _main(["reduce", "--out", str(out)]) == 3
    moved.rename(out / RN.FROZEN_NAME)
    assert _main(["reduce", "--out", str(out)]) == 0


# ---------------------------------------------------------------------------------------
# offset handling: the double-application trap is caught
# ---------------------------------------------------------------------------------------
def test_offset_applied_once_rejects_double_application(P):
    old = P["old_frozen_prediction"]
    params = next(v for k, v in old["per_theta"].items() if abs(float(k) - 0.12) <= 1e-12)
    N = np.asarray(old["old_grid"], float)
    off = float(params["offset_a_ref"])
    base, pred = AN.old_endpoint_prediction(params, N, off)
    # the correct, single-application prediction is accepted
    AN.check_offset_applied_once(params=params, N=N, offset=off, prediction=pred)
    # the September trap: base + 2 a_ref (or a stored prediction re-offset) is rejected
    with pytest.raises(ValueError, match="exactly once|twice"):
        AN.check_offset_applied_once(params=params, N=N, offset=off, prediction=base + 2.0 * off)
    with pytest.raises(ValueError):
        AN.check_offset_applied_once(params=params, N=N, offset=off, prediction=base + 3.0 * off)
    # a prediction built with a DIFFERENT current a_ref is not silently accepted as the same offset
    _, pred_other = AN.old_endpoint_prediction(params, N, off + 0.5)
    with pytest.raises(ValueError):
        AN.check_offset_applied_once(params=params, N=N, offset=off, prediction=pred_other)
    # the guard works for every new-model form, not just c/N + d/N^2
    N2 = np.asarray([400.0, 800.0, 1600.0])
    for model, params2 in (("c_over_N", {"model": "c_over_N", "c": -100.0}),
                           ("c_over_N_alpha", {"model": "c_over_N_alpha", "c": -100.0, "alpha": 1.2}),
                           ("c_over_N_d_over_N2", {"model": "c_over_N_d_over_N2", "c": -100.0, "d": 1e5})):
        b = AN.predict(params2, N2)
        AN.check_offset_applied_once(params=params2, N=N2, offset=off, prediction=b + off)
        with pytest.raises(ValueError):
            AN.check_offset_applied_once(params=params2, N=N2, offset=off, prediction=b + 2.0 * off)


def test_endpoint_a_reproduces_the_archived_stored_prediction(P):
    """At the production geometry the current a_ref equals the archived offset, and endpoint (a)
    rebuilt from the original (c, d) with a_ref applied once reproduces the t20s stored
    prediction_at_grid (which itself already contained a_ref).  Adding a_ref twice does not."""
    D = RN.design(P, False)
    old = P["old_frozen_prediction"]["per_theta"]
    N = np.asarray(P["grid_catalogue"]["old_N"], float)
    for th in (0.12, 0.25):
        ref = RN._reference(D, float(th))
        assert ref["a_best_found"] == pytest.approx(old[str(th)]["offset_a_ref"], abs=1e-9)
        base, pred = AN.old_endpoint_prediction(old[str(th)], N, ref["a_best_found"])
        assert np.max(np.abs(pred - np.asarray(old[str(th)]["stored_prediction_at_grid"]))) < 1e-9
        doubled = base + 2.0 * ref["a_best_found"]
        assert np.max(np.abs(doubled - np.asarray(old[str(th)]["stored_prediction_at_grid"]))) > 0.1


def test_double_applied_offset_rejected_by_reduce(full_run, tmp_path):
    out = tmp_path / "mut"
    shutil.copytree(full_run, out)
    fz = json.loads((out / RN.FROZEN_NAME).read_text())
    touched = 0
    for name, s in fz["new_selection"].items():
        if s["selected"] == "none":
            continue
        for th, entry in s["frozen_per_theta"].items():
            off = float(entry["offset_a_ref"])
            if abs(off) <= 1e-9:
                continue
            entry["prediction_at_grid"] = [p + off for p in entry["prediction_at_grid"]]
            touched += 1
    assert touched > 0, "the fixture freeze must contain at least one non-zero offset to exercise the trap"
    (out / RN.FROZEN_NAME).write_text(json.dumps(fz))
    assert _main(["reduce", "--out", str(out)]) == 4
    # the same trap must also fail the direct control on the frozen entry
    entry = fz["new_selection"]["d_minus_a_ref"]["frozen_per_theta"]["0.12"]
    with pytest.raises(RN.ScienceError):
        RN.check_offset_convention(entry, np.asarray([400, 800, 1600]), where="test")


# ---------------------------------------------------------------------------------------
# the 2.7% margin was against none; the c/N^alpha margin is smaller and distinct
# ---------------------------------------------------------------------------------------
def test_old_selection_margins_distinguish_none_and_alpha(P):
    m = AN.old_selection_margins(P["old_frozen_prediction"]["cv_errors"])
    assert m["selected"] == "c_over_N_d_over_N2"
    assert m["margin_vs_none_rel"] == pytest.approx(0.02672759246226115, abs=1e-12)
    assert m["margin_vs_c_over_N_alpha_rel"] == pytest.approx(0.004267072115431259, abs=1e-12)
    assert m["margin_vs_none_abs"] == pytest.approx(0.007950374567227136, abs=1e-15)
    assert m["margin_vs_c_over_N_alpha_abs"] == pytest.approx(0.0012406499842037566, abs=1e-15)
    # the reported 2.7% figure is the margin over none, NOT the smaller margin over c/N^alpha
    assert m["margin_vs_none_rel"] > 5.0 * m["margin_vs_c_over_N_alpha_rel"]
    assert m["margin_vs_none_abs"] > 4.0 * m["margin_vs_c_over_N_alpha_abs"]
    assert m["none_minus_c_over_N_alpha_rel"] == pytest.approx(0.02255677171844381, abs=1e-12)


# ---------------------------------------------------------------------------------------
# resource estimate (honestly labelled: no BENCHMARK.json is shipped)
# ---------------------------------------------------------------------------------------
def test_resource_estimate_from_fixture_timings(full_run):
    assert _main(["estimate", "--out", str(full_run)]) == 0
    est = json.loads((full_run / "RESOURCE_ESTIMATE.json").read_text())
    assert est["measured_projection_available"] is False
    assert "BENCHMARK.json" in est["why_no_measured_projection"]
    assert est["unmeasured_families"], "with no BENCHMARK.json every family must be reported unmeasured"
    assert est["priced_matrix"] == "production"
    assert est["priced_counts"]["cells"] == 966 and est["priced_counts"]["fits"] == 4800
    assert est["retry_allowance"]["fits_headroom_under_cap"] == 7000 - 4800
    assert est["fixture_counts"]["fits"] == 192
    assert est["total_cpu_seconds_point"] >= 0.0
    flat = est["cross_check_flat_fixture_scale"]
    assert all(v.get("measured") for v in flat["families"].values())
    assert flat["total_cpu_seconds_point"] > 0.0


# ---------------------------------------------------------------------------------------
# the optional four-geometry expansion is declared and costed, not required
# ---------------------------------------------------------------------------------------
def test_optional_expansion_declared_and_costed(P):
    required = RN.build_matrix(P, False)
    optional = RN._build_matrix(P, False, include_optional=True)
    assert sorted({c["params"].get("theta") for c in required.cells if c["params"].get("kind") == "fit"}) == [0.0, 0.12, 0.25]
    assert 0.4 not in {c["params"].get("theta") for c in required.cells if c["params"].get("kind") == "fit"}
    assert 0.4 in {c["params"].get("theta") for c in optional.cells if c["params"].get("kind") == "fit"}
    rc, oc = required.counts(), optional.counts()
    assert (rc["cells"], rc["fits"], rc["solves"]) == (966, 4800, 105)
    assert (oc["cells"], oc["fits"], oc["solves"]) == (1287, 6400, 140)
    assert oc["fits"] <= P["budget"]["estimator_fits_cap"]
    decl = P["matrix"]["production"]["geometries"]["optional_costed_expansion"]
    assert decl["theta"] == 0.4 and decl["extra_fits"] == oc["fits"] - rc["fits"] == 1600


# ---------------------------------------------------------------------------------------
# production role path (pilot/confirmation roles, bootstrap bands, theta=0.25) on a shrunken copy
# ---------------------------------------------------------------------------------------
def test_production_role_path_with_shrunken_protocol(tmp_path):
    P = json.loads(PROTO.read_text())
    P["matrix"]["production"].update({
        "C": 3, "T": 4, "k_c": 2, "k_t": 2,
        "spectrum": {"top": [6.0, 5.0, 4.0, 3.0], "gap": 1.5, "tail_min": 0.05},
        "geometries": {"required": [0.0, 0.12, 0.25], "optional_costed_expansion": None},
        "N_core": [400, 800], "N_anchor": [1600],
        "R": 2, "R_pop": 4, "als_n_iter": 60, "pop_als_n_iter": 200,
        "pilot_seeds": [23000, 23008], "confirmation_seeds": [24000, 24008]})
    P["grid_catalogue"] = {"old_N": [400, 800], "extended_N": [1600], "N": [400, 800, 1600],
                           "old_grid_max": 800}
    proto = tmp_path / "PROTO_SHRUNK.json"
    proto.write_text(json.dumps(P))
    out = tmp_path / "prod"

    def run(*a):
        return RN.main([a[0], "--protocol", str(proto)] + list(a[1:]))

    # confirmation cannot be drawn before the freeze exists (nothing created)
    assert run("run", "--out", str(out), "--families", "confirmation") == 6 and not out.exists()
    assert run("run", "--out", str(out), "--families", "algebraic", "pilot", "control") == 0
    assert run("reduce", "--out", str(out)) == 3                       # confirmation cells missing, no --allow-partial
    assert run("reduce", "--out", str(out), "--allow-partial") == 0     # pilot freeze
    fz = json.loads((out / RN.FROZEN_NAME).read_text())
    assert fz["development_only"] is False and fz["pilot_cells"] == 81
    assert run("run", "--out", str(out), "--families", "confirmation") == 0
    assert run("reduce", "--out", str(out)) == 0
    red = json.loads((out / "REDUCED.json").read_text())
    assert red["phase"] == "confirmation" and red["development_only"] is False and red["partial"] is False
    assert red["reconciliation"]["complete"]
    assert red["ledger"]["charged"]["fits"] == 162 * 3 and red["ledger"]["charged"]["solves"] == 21
    roles = {js["role"] for files in R.read_cells(out).values() for _, js in files}
    assert roles == {"pilot", "confirmation", "algebraic", "control"}
    # 9 confirmation seeds >= min_n = 8: the studentized bootstrap (not the fallback) is the band
    assert red["confirmation"]["per_theta"]["0.25"]["d_simultaneous"]["method"] == "max_abs_t_bootstrap"
    # endpoint (a) covers the focal and the previously bracketed reference geometry
    ea = red["confirmation"]["endpoint_a_old_prediction"]
    assert "0.12" in ea and "0.25" in ea and ea["0.25"]["scored"] is True
    assert "extrapolated_flags" in ea["0.25"]
    mp = red["confirmation"]["multiplicity"]
    assert abs(mp["alpha_claim"] * mp["m_claims"] - mp["alpha"]) < 1e-12
    assert mp["m_claims"] >= 2
    grid_max = max(red["grid"]["N"])
    for th, p in red["confirmation"]["per_theta"].items():
        cr = p["crossing_report"]
        if cr["no_crossing_within_grid"]:
            assert cr["grid_bound_N"] == grid_max
        else:
            assert cr["grid_bound_N"] is None and cr["brackets_N"]


# ---------------------------------------------------------------------------------------
# analysis primitives (the NEW selection, bands and profiles)
# ---------------------------------------------------------------------------------------
def test_analysis_selection_band_and_profile():
    rng = np.random.default_rng(0)
    N = np.array([400, 800, 1000, 1250, 1600, 3200], float)
    truth = 2.0 - 2200.0 / N
    Y = truth[None, :] + 0.2 * rng.standard_normal((24, N.size))
    band = AN.simultaneous_band(Y, seed=1, replicates=500)
    assert band["method"] == "max_abs_t_bootstrap" and band["q_max_t"] > 2.0
    assert len(band["maxt_draws"]) >= 250
    prof = AN.archived_sign_profile(band)
    assert prof["signs"][0] == "-" and prof["signs"][-1] == "+" and len(prof["adjacent_brackets"]) >= 1
    small = AN.simultaneous_band(Y[:4], seed=1, replicates=200)
    assert small["method"] == "bonferroni_t_fallback"
    # 'none' is a valid outcome: a zigzag (non-scaling) target is tracked by the per-N bracket and no
    # parametric scaling model beats it; a clean c/N target selects a parametric candidate
    seeds = list(range(23000, 23016))
    zig = np.where(np.arange(N.size) % 2 == 0, -3.0, 3.0)
    Yn = {0: zig[None, :] + 0.2 * rng.standard_normal((16, N.size)),
          1: zig[None, :] + 0.2 * rng.standard_normal((16, N.size))}
    sel_none = AN.select_model(Yn, N.tolist(), seeds)
    assert sel_none["selected"] == "none"
    assert sel_none["relative_improvement"] < AN.MIN_RELATIVE_IMPROVEMENT
    Yc = {0: 3000.0 / N[None, :] + 0.05 * rng.standard_normal((16, N.size))}
    sel = AN.select_model(Yc, N.tolist(), seeds)
    assert sel["selected"] in ("c_over_N", "c_over_N_alpha", "c_over_N_d_over_N2")
    assert sel["cv_error"][sel["selected"]] < sel["cv_error"]["none"]
    A, B = AN.split_halves(seeds)
    assert A == seeds[:8] and B == seeds[8:]
    p = AN.fit_model("c_over_N_d_over_N2", N, 3000 / N + 5e5 / N ** 2)
    assert abs(p["c"] - 3000) < 1e-6 and abs(p["d"] - 5e5) < 1e-3


def test_holm_is_a_step_down():
    h = AN.holm({"a": 0.001, "b": 0.04, "c": 0.5})
    assert h["order"][0] == "a"
    assert h["adjusted"]["a"] == pytest.approx(0.003)
    assert h["adjusted"]["b"] == pytest.approx(0.08)
    assert h["reject_inadequacy_0.05"]["a"] is True
    assert h["reject_inadequacy_0.05"]["b"] is False


# --- A3: simulator-truth audit for the inherited generator ---------------------------
def test_truth_audit_passes_on_the_fixture_design():
    from experiments.mlst2026.followup_20260917.r2_t20_theta012 import truth_audit as TA
    P = PR.load_protocol(PROTO)
    rec = TA.audit_generator(P, fixture=True)
    assert rec["all_passed"], rec["checks"]
    assert rec["checks"]["no_clipping_in_source"]["passed"]
    assert rec["checks"]["spectrum_fixed"]["passed"]
    assert rec["checks"]["nonseparability_moves"]["passed"]
    assert "heavy_tail_initialisation" in rec["stochastic_audit"]


def test_committed_production_truth_audit_receipt():
    import json
    rec = json.loads((PKG / "TRUTH_AUDIT.json").read_text())
    assert rec["fixture"] is False and rec["all_passed"] is True
    assert rec["thetas"] == [0.0, 0.12, 0.25]
    assert rec["checks"]["no_clipping_in_source"]["tokens_found"] == []
