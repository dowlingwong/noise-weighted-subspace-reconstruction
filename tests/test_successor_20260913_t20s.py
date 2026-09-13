"""T20-S (L2) package tests: the eight README tests with the real reduced runner
and the fixture protocol block, plus the package-specific scientific tests
(invariants, theta = 0 control, bound order, identity, frozen-selection gate,
mutation).  Everything runs in the development seed namespace.  Nothing here is
a scientific result."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.mlst2026.successor_20260913.common import ledger as L, matrix as M, protocol as PR, results as R  # noqa: E402
from experiments.mlst2026.successor_20260913.t20s import analysis as AN, generator as G, runner as RN  # noqa: E402

PKG = REPO / "experiments" / "mlst2026" / "successor_20260913" / "t20s"
PROTO = PKG / "PROTOCOL_T20S_v2.json"


def _main(argv):
    return RN.main([argv[0], "--protocol", str(PROTO)] + argv[1:])


def _pilot_fixture(out: Path):
    assert _main(["run", "--out", str(out), "--fixture", "--families", "pilot", "algebraic"]) == 0
    assert _main(["reduce", "--out", str(out), "--allow-partial"]) == 0
    assert (out / RN.FROZEN_NAME).exists()


def _full_fixture(out: Path):
    _pilot_fixture(out)
    assert _main(["run", "--out", str(out), "--fixture", "--families", "confirmation"]) == 0
    assert _main(["reduce", "--out", str(out)]) == 0


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("t20s_full")
    _full_fixture(out)
    return out


@pytest.fixture(scope="module")
def P():
    return PR.load_protocol(PROTO)


# ---------------------------------------------------------------------------------------
# README test 1: validate / enumerate compute nothing
# ---------------------------------------------------------------------------------------
def test_validate_and_enumerate_compute_nothing(tmp_path, capsys):
    assert _main(["validate"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert js["counts"]["fits"] == 4800 and js["counts"]["solves"] == 140 and js["counts"]["cells"] == 964
    assert js["counts"]["by_role"]["pilot"]["cells"] == 384 and js["counts"]["by_role"]["confirmation"]["cells"] == 576
    assert js["counts"]["by_role"]["algebraic"]["cells"] == 4 and js["fits_within_cap"]
    out = tmp_path / "enum"
    assert _main(["enumerate", "--out", str(out)]) == 0
    assert (out / "run_matrix.jsonl").exists() and not (out / "cells").exists()
    rows = [json.loads(l) for l in (out / "run_matrix.jsonl").read_text().splitlines()]
    fit_rows = [r for r in rows if r["params"]["kind"] == "fit"]
    assert all(r["n_fits"] == 1 + r["params"]["R"] == 5 for r in fit_rows)
    assert sorted({r["params"]["N"] for r in fit_rows}) == [400, 800, 1000, 1250, 1600, 3200]
    assert sorted({r["params"]["theta"] for r in fit_rows}) == [0.0, 0.12, 0.25, 0.40]


def test_help_and_module_entry_never_compute():
    with pytest.raises(SystemExit) as e:
        _main(["--help"])
    assert e.value.code == 0
    env = dict(os.environ, PYTHONPATH=str(REPO))
    cp = subprocess.run([sys.executable, "-m", "experiments.mlst2026.successor_20260913.t20s", "validate", "--protocol", str(PROTO)],
                        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120)
    assert cp.returncode == 0, cp.stderr
    assert json.loads(cp.stdout)["package"] == "t20s"


def test_run_refuses_archive_roots():
    rc = _main(["run", "--out", str(REPO / "results" / "t20s_should_never_exist"), "--fixture", "--families", "algebraic"])
    assert rc == 6 and not (REPO / "results" / "t20s_should_never_exist").exists()


# ---------------------------------------------------------------------------------------
# README test 2: fixture end to end with the declared schema
# ---------------------------------------------------------------------------------------
def test_fixture_end_to_end_schema(full_run, P):
    red = json.loads((full_run / "REDUCED.json").read_text())
    assert red["development_only"] is True and red["reconciliation"]["complete"] and red["phase"] == "confirmation"
    assert red["pilot_complete"] and red["confirmation_complete"] and red["partial"] is False
    for key in ("invariants", "controls", "grid", "confirmation", "frozen_choice", "excluded_thetas", "nonseparability_moves"):
        assert key in red
    cells = R.read_cells(full_run)
    kinds = {}
    for files in cells.values():
        for _, js in files:
            R.check_cell_schema(js, RN.RESULT_REQUIRED)
            kinds[js["result"]["kind"]] = kinds.get(js["result"]["kind"], 0) + 1
            assert js["development_only"]
    assert kinds == {"algebraic": 2, "fit": 16}
    fit = next(js["result"] for files in cells.values() for _, js in files if js["result"]["kind"] == "fit")
    for k in ("R_U_hat", "R_F_hat", "d", "x_U", "x_F_ref", "a_lower_bound", "a_best_found", "a_interval", "identity_residual",
              "restart_train_losses", "restart_pop_risks", "selected_restart", "selected_by", "population_best_restart",
              "selection_matches_population_best", "restart_spread_loss", "restart_spread_risk", "optimization_unresolved"):
        assert k in fit, k
    assert fit["selected_by"] == "training_loss" and len(fit["restart_train_losses"]) == fit["R"] == 2
    conf = red["confirmation"]
    for th, per in conf["per_theta"].items():
        assert per["sign_profile"]["profile"] is not None and "by_N" in per and per["d_simultaneous"]["method"] in ("max_abs_t_bootstrap", "bonferroni_t_fallback")
    # reduce is deterministic
    a = (full_run / "REDUCED.json").read_bytes()
    assert _main(["reduce", "--out", str(full_run)]) == 0
    assert (full_run / "REDUCED.json").read_bytes() == a


def test_resource_estimate_from_fixture_timings(full_run):
    assert _main(["estimate", "--out", str(full_run)]) == 0
    est = json.loads((full_run / "RESOURCE_ESTIMATE.json").read_text())
    assert est["unmeasured_families"] == [] and est["total_cpu_seconds_point"] > 0
    assert set(est["families"]) == {"pilot", "confirmation", "algebraic"}
    assert est["retry_allowance"]["fits_headroom_under_cap"] == 5400 - 4800  # common prices the PRODUCTION matrix
    assert est["priced_matrix"] == "production" and est["fixture_counts"]["fits"] == 48
    proj = est["production_projection"]                                     # full production matrix x scaled fixture timings
    assert proj["production_counts"]["cells"] == 964 and proj["production_counts"]["fits"] == 4800
    assert proj["unmeasured_families"] == [] and proj["retry_allowance"]["fits_headroom_under_cap"] == 600
    assert proj["total_cpu_seconds_point"] == pytest.approx(sum(v["production_cpu_seconds"] for v in proj["families"].values()))
    assert proj["families"]["pilot"]["cells"] == 384 and proj["families"]["confirmation"]["cells"] == 576


# ---------------------------------------------------------------------------------------
# README test 3 + package science: invariants, controls, bound order, identity
# ---------------------------------------------------------------------------------------
def test_generator_invariants_spectrum_fixed_and_nonseparability_moves(P):
    D = RN.design(P, True)
    g = RN._geometry(D)
    thetas = [0.0, 0.1, 0.3, 0.6]
    inv = G.invariants(g, thetas, R_pop=D["R_pop"], n_iter=D["pop_als_n_iter"], tol=D["pop_als_tol"], spec=D["spectrum"])
    assert inv["excluded_thetas"] == [] and inv["nonseparability_moves"]
    kmc = [inv["per_theta"][t]["partial_trace"]["k_minus_cap"] for t in thetas]
    for t in thetas:
        r = inv["per_theta"][t]
        assert r["eigenvalues_max_abs_dev"] <= 1e-10
        assert abs(r["trace_K"] - g.mu.sum()) < 1e-8 and r["numerical_rank"] == g.d
        assert r["a_lower_bound"] <= r["a_best_found"] + 1e-9 and r["a_best_found"] >= -1e-9
        assert abs(r["a_lower_bound"] - r["Delta_k"] * r["partial_trace"]["k_minus_cap"]) < 1e-9
    assert kmc[0] == 0.0 and all(kmc[i] < kmc[i + 1] for i in range(len(kmc) - 1))
    # V(theta) is orthogonal and K(theta) has exactly the declared spectrum at production size too
    Dp = RN.design(P, False)
    gp = RN._geometry(Dp)
    for th in Dp["thetas"]:
        V = gp.V(th)
        assert np.max(np.abs(V.T @ V - np.eye(gp.d))) < 1e-12
    w = np.sort(np.linalg.eigvalsh(gp.K(0.25)))[::-1]
    assert np.max(np.abs(w - gp.mu)) <= 1e-10


def test_theta0_control_lower_bound_zero_and_best_found_zero(P):
    D = RN.design(P, True)
    ref = RN._reference(D, 0.0)
    assert ref["a_lower_bound"] == 0.0 and abs(ref["a_best_found"]) <= 1e-8
    assert ref["partial_trace"]["cap"] == pytest.approx(D["k"], abs=1e-9)
    # the top-k eigenspace at theta = 0 IS the product subspace
    B = G.kron_basis(RN._geometry(D).U_c0, RN._geometry(D).U_t0)
    P_prod = B @ B.T
    P_U = ref["B_U_star"] @ ref["B_U_star"].T
    assert np.max(np.abs(P_prod - P_U)) < 1e-9


def test_controls_and_identity_in_fixture_cells(full_run):
    red = json.loads((full_run / "REDUCED.json").read_text())
    assert red["controls"]["theta0_a_zero"] is True and red["controls"]["spectrum_fixed"] is True
    assert red["invariants"]["0.0"]["a_interval"] == [0.0, pytest.approx(0.0, abs=1e-8)]
    lo, hi = red["invariants"]["0.3"]["a_interval"]
    assert 0 < lo <= hi
    cells = R.read_cells(full_run)
    for files in cells.values():
        js = files[0][1]
        r = js["result"]
        if r["kind"] != "fit":
            continue
        # identity with the reference definitions: d = a_best + x_F_ref - x_U
        assert abs(r["d"] - (r["a_ref"] + r["x_F_ref"] - r["x_U"])) < 1e-9 and abs(r["identity_residual"]) < 1e-9
        assert r["x_U"] >= -1e-9 and r["a_lower_bound"] <= r["a_best_found"] + 1e-9
        assert r["selected_restart"] == int(np.argmin(r["restart_train_losses"]))
        if r["theta"] == 0.0:
            assert r["a_lower_bound"] == 0.0 and r["a_best_found"] <= 1e-8


# ---------------------------------------------------------------------------------------
# README test 4: namespaces disjoint; a fixture run refuses pilot/confirmation cells
# ---------------------------------------------------------------------------------------
def test_namespaces_disjoint_and_fixture_refuses_pilot_confirmation(P, tmp_path):
    m = RN.build_matrix(P, False)
    seeds = {role: {c["seed"] for c in m.cells if c["role"] == role} for role in ("pilot", "confirmation")}
    assert seeds["pilot"] and seeds["confirmation"] and not (seeds["pilot"] & seeds["confirmation"])
    assert all(PR.namespace_of_seed(P, s) == "pilot" for s in seeds["pilot"])
    assert all(PR.namespace_of_seed(P, s) == "confirmation" for s in seeds["confirmation"])
    assert M.check_seed_roles(m, P) == [] and M.check_seed_roles(RN.build_matrix(P, True), P) == []
    fx = RN.build_matrix(P, True)
    assert {c["role"] for c in fx.cells} == {"development", "algebraic"}
    assert all(PR.namespace_of_seed(P, c["seed"]) == "development" for c in fx.cells if c["seed"] is not None)
    assert _main(["run", "--out", str(tmp_path / "a"), "--fixture", "--roles", "pilot"]) == 6
    assert _main(["run", "--out", str(tmp_path / "b"), "--fixture", "--roles", "confirmation"]) == 6
    assert not (tmp_path / "a" / "cells").exists()


# ---------------------------------------------------------------------------------------
# README test 5: reduce refuses partial, duplicated and tampered cells (exit 3)
# ---------------------------------------------------------------------------------------
def test_reduce_refuses_partial_duplicate_tampered(tmp_path):
    out = tmp_path / "r"
    # partial pilot (2 of 8 pilot cells) -> the reducer refuses to freeze even with --allow-partial? no: the
    # common layer accepts --allow-partial but the package refuses an incomplete pilot (exit 3)
    assert _main(["run", "--out", str(out), "--fixture", "--families", "pilot", "algebraic", "--max-cells", "4"]) == 0
    assert _main(["reduce", "--out", str(out)]) == 3
    assert _main(["reduce", "--out", str(out), "--allow-partial"]) == 3
    assert not (out / RN.FROZEN_NAME).exists()
    assert _main(["run", "--out", str(out), "--fixture", "--families", "pilot", "algebraic"]) == 0
    assert _main(["reduce", "--out", str(out), "--allow-partial"]) == 0
    # duplicate: a copied cell file under a new id
    src = sorted((out / "cells").glob("*.json"))[0]
    js = json.loads(src.read_text())
    js["cell_id"] = "deadbeefdeadbeef"
    dup = out / "cells" / "deadbeefdeadbeef.json"
    dup.write_text(json.dumps(js))
    assert _main(["reduce", "--out", str(out), "--allow-partial"]) == 3
    dup.unlink()
    # tampered: a changed value inside an attributed file -> digest mismatch
    js = json.loads(src.read_text())
    js["result"]["d"] = 123.0
    src.write_text(json.dumps(js))
    assert _main(["reduce", "--out", str(out), "--allow-partial"]) == 3


# ---------------------------------------------------------------------------------------
# README test 6: injected failure is charged; the retry accumulates
# ---------------------------------------------------------------------------------------
def test_failure_charged_and_retry_accumulates(P, tmp_path):
    out = tmp_path / "f"
    m = RN.build_matrix(P, True)
    victim = [c for c in m.cells if c["family"] == "pilot"][0]["cell_id"]
    assert _main(["run", "--out", str(out), "--fixture", "--families", "pilot", "algebraic", "--fail-cell", victim]) == 1
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=P["budget"]["estimator_fits_cap"], package="t20s", protocol_sha256="x")
    ch = led.charged()
    assert ch["by_state"]["failed"] == 1 and ch["fits"] >= 3
    assert _main(["run", "--out", str(out), "--fixture", "--families", "pilot", "algebraic"]) == 0
    ch2 = led.charged()
    assert ch2["by_state"]["failed"] == 1 and ch2["by_state"]["completed"] == 10 and ch2["fits"] == 8 * 3 + 3  # 8 pilot cells + the failed attempt


# ---------------------------------------------------------------------------------------
# frozen selection gate
# ---------------------------------------------------------------------------------------
def test_confirmation_refused_without_frozen_selection(tmp_path, P):
    out = tmp_path / "gate"
    assert _main(["run", "--out", str(out), "--fixture", "--families", "pilot", "algebraic"]) == 0
    # before the pilot reduce: refused before any attempt is charged
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=P["budget"]["estimator_fits_cap"], package="t20s", protocol_sha256="x")
    before = led.charged()["attempts"]
    assert _main(["run", "--out", str(out), "--fixture", "--families", "confirmation"]) == 6
    assert _main(["run", "--out", str(out), "--fixture"]) == 6                      # unrestricted fixture run also selects confirmation
    assert led.charged()["attempts"] == before
    assert not any(json.loads(p.read_text())["params"].get("phase") == "confirmation" for p in (out / "cells").glob("*.json"))
    # the in-cell guard also refuses (charged failed attempt) if the file vanishes between the precheck and the cell
    import experiments.mlst2026.successor_20260913.t20s.runner as runner_mod
    orig = runner_mod._confirmation_precheck
    runner_mod._confirmation_precheck = lambda argv: None
    try:
        assert _main(["run", "--out", str(out), "--fixture", "--families", "confirmation"]) == 1
    finally:
        runner_mod._confirmation_precheck = orig
    assert led.charged()["by_state"]["failed"] == 1
    # confirmation reduce without the file is refused too
    assert _main(["reduce", "--out", str(out), "--allow-partial"]) == 0   # pilot reduce: freezes
    fz = json.loads((out / RN.FROZEN_NAME).read_text())
    assert fz["confirmation_cells_present_at_freeze"] == 0 and fz["protocol_sha256"] == P["_sha256"]
    assert fz["selection"]["x_U"]["selected"] in AN.CANDIDATES and fz["selection"]["d_minus_a_ref"]["selected"] in AN.CANDIDATES
    assert _main(["run", "--out", str(out), "--fixture", "--families", "confirmation"]) == 0
    (out / RN.FROZEN_NAME).rename(out / "FROZEN_SELECTION.json.moved")
    assert _main(["reduce", "--out", str(out)]) == 3
    (out / "FROZEN_SELECTION.json.moved").rename(out / RN.FROZEN_NAME)
    assert _main(["reduce", "--out", str(out)]) == 0
    red = json.loads((out / "REDUCED.json").read_text())
    assert red["phase"] == "confirmation" and red["frozen_choice"] == {k: v["selected"] for k, v in fz["selection"].items()}
    # a frozen file for a different protocol is refused before charging
    fz2 = dict(fz, protocol_sha256="0" * 64)
    (out / RN.FROZEN_NAME).write_text(json.dumps(fz2))
    assert _main(["run", "--out", str(out), "--fixture", "--families", "confirmation", "--allow-repeat"]) == 6


def test_frozen_selection_is_never_overwritten(full_run):
    fz_path = full_run / RN.FROZEN_NAME
    fz = json.loads(fz_path.read_text())
    tampered = dict(fz)
    tampered["selection"] = json.loads(json.dumps(fz["selection"]))
    for k in tampered["selection"]:
        tampered["selection"][k]["selected"] = "c_over_N_d_over_N2" if fz["selection"][k]["selected"] != "c_over_N_d_over_N2" else "none"
    try:
        fz_path.write_text(json.dumps(tampered))
        # confirmation cells exist -> confirmation phase reduce reads the file and scores it; it cannot re-freeze.
        assert _main(["reduce", "--out", str(full_run)]) == 0
        red = json.loads((full_run / "REDUCED.json").read_text())
        assert red["frozen_choice"] == {k: v["selected"] for k, v in tampered["selection"].items()}
    finally:
        fz_path.write_text(json.dumps(fz))
        assert _main(["reduce", "--out", str(full_run)]) == 0


# ---------------------------------------------------------------------------------------
# README test 7: mutation — rejected by the package's own scientific checks, not the digest
# ---------------------------------------------------------------------------------------
def test_mutation_corrupted_lower_bound_rejected_by_science_check(tmp_path):
    out = tmp_path / "mut"
    _pilot_fixture(out)
    led_path = out / "EXECUTION_LEDGER.jsonl"
    victim = next(p for p in sorted((out / "cells").glob("*.json")) if json.loads(p.read_text())["result"]["kind"] == "fit"
                  and json.loads(p.read_text())["result"]["theta"] != 0.0)
    js = json.loads(victim.read_text())
    js["result"]["a_lower_bound"] = js["result"]["a_best_found"] + 1.0     # a 'valid' lower bound above the best-found value
    js["result"]["a_interval"] = [js["result"]["a_lower_bound"], js["result"]["a_best_found"]]
    digest = R.write_json_atomic(victim, js)
    # re-sync the ledger digest so that ONLY the scientific check can catch it
    lines = led_path.read_text().splitlines()
    new = []
    for l in lines:
        rec = json.loads(l)
        if rec.get("state") == "completed" and rec["cell_id"] == js["cell_id"]:
            rec["output_sha256"] = digest
        new.append(json.dumps(rec, sort_keys=True))
    led_path.write_text("\n".join(new) + "\n")
    rc = _main(["reduce", "--out", str(out), "--allow-partial"])
    assert rc == 4, "a lower bound above the best-found value must be rejected by the science check (exit 4)"
    with pytest.raises(RN.ScienceError, match="not a valid interval|exceeds best-found"):
        RN.check_fit_result(js["result"])
    # direct checks of the other invariants
    good = json.loads(victim.read_text())["result"]
    bad = dict(good, a_lower_bound=good["a_best_found"], a_interval=[good["a_best_found"], good["a_best_found"]], x_U=-1e-3)
    with pytest.raises(RN.ScienceError, match="x_U"):
        RN.check_fit_result(bad)
    alg = next(json.loads(p.read_text())["result"] for p in (out / "cells").glob("*.json") if json.loads(p.read_text())["result"]["kind"] == "algebraic" and json.loads(p.read_text())["result"]["theta"] == 0.0)
    with pytest.raises(RN.ScienceError, match="theta = 0 control"):
        RN.check_algebraic_result(dict(alg, a_best_found=1e-3, a_interval=[0.0, 1e-3]))
    with pytest.raises(RN.ScienceError, match="spectrum"):
        RN.check_algebraic_result(dict(alg, eigenvalues_max_abs_dev=1e-6))


def test_mutation_restart_selected_not_by_training_loss_rejected(tmp_path, monkeypatch):
    out = tmp_path / "mut2"
    # selection by the WORST training loss (a rule other than training loss): the self-check refuses at completion
    monkeypatch.setattr(RN, "select_restart", lambda losses: int(np.argmax(np.asarray(losses, float))))
    rc = _main(["run", "--out", str(out), "--fixture", "--families", "pilot", "algebraic"])
    assert rc == 4, "a ScienceError at completion is a schema-class refusal (exit 4) with a charged failed attempt"
    led = L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=5400, package="t20s", protocol_sha256="x")
    failed = [a for a in led.attempts().values() if a["state"] == "failed"]
    assert len(failed) == 1 and "training-loss minimiser" in failed[0]["error"]
    assert not (out / "cells" / f"{failed[0]['cell_id']}.json").exists()      # the refused cell wrote no output
    # cells completed before the refusal can only be ones where both restarts tie exactly (argmax == argmin)
    for pth in (out / "cells").glob("*.json"):
        r = json.loads(pth.read_text())["result"]
        assert r["restart_train_losses"][0] == r["restart_train_losses"][1]
    monkeypatch.undo()
    # selection by population risk instead of training loss is also refused by the check itself
    r = {"theta": 0.3, "N": 40, "selected_by": "population_risk", "R": 2, "restart_train_losses": [1.0, 2.0], "restart_pop_risks": [2.0, 1.0],
         "selected_restart": 1, "als_selected_restart": 0}
    with pytest.raises(RN.ScienceError, match="training loss"):
        RN.check_fit_result(r)


# ---------------------------------------------------------------------------------------
# analysis primitives
# ---------------------------------------------------------------------------------------
def test_simultaneous_ci_and_profile_and_selection():
    rng = np.random.default_rng(0)
    N = np.array([400, 800, 1000, 1250, 1600, 3200], float)
    truth = 2.0 - 2200.0 / N                        # -3.5 ... -0.2 at 1000, +0.24 at 1250: decided opposite signs
    Y = truth[None, :] + 0.2 * rng.standard_normal((24, N.size))
    ci = AN.simultaneous_ci(Y, seed=1, replicates=500)
    assert ci["method"] == "max_abs_t_bootstrap" and ci["q_max_t"] > 2.0
    assert all(h >= b * 0.5 for h, b in zip(ci["half_width"], ci["bonferroni"]["half_width"]))
    prof = AN.sign_profile(ci)
    assert prof["signs"][0] == "-" and prof["signs"][-1] == "+" and len(prof["adjacent_brackets"]) >= 1
    ci2 = AN.simultaneous_ci(Y[:4], seed=1, replicates=200)
    assert ci2["method"] == "bonferroni_t_fallback"
    # selection: a clean c/N target picks c_over_N; pure noise picks none (or at least never a wrong claim of support)
    seeds = list(range(3000, 3016))
    Yx = {0: 3000.0 / N[None, :] + 0.05 * rng.standard_normal((16, N.size)), 1: 2500.0 / N[None, :] + 0.05 * rng.standard_normal((16, N.size))}
    sel = AN.cv_select(Yx, N.tolist(), seeds)
    assert sel["selected"] in ("c_over_N", "c_over_N_alpha", "c_over_N_d_over_N2")
    assert sel["cv_error"][sel["selected"]] < sel["cv_error"]["none"]
    Yn = {0: 5.0 + rng.standard_normal((16, N.size))}
    sel_n = AN.cv_select(Yn, N.tolist(), seeds)
    assert sel_n["selected"] in AN.CANDIDATES
    A, B = AN.split_halves(seeds)
    assert A == seeds[:8] and B == seeds[8:]
    p = AN.fit_model("c_over_N_d_over_N2", N, 3000 / N + 5e5 / N ** 2)
    assert abs(p["c"] - 3000) < 1e-6 and abs(p["d"] - 5e5) < 1e-3


# ---------------------------------------------------------------------------------------
# production role path (pilot/confirmation roles, non-fixture reduce) on a shrunken protocol copy
# ---------------------------------------------------------------------------------------
def test_production_role_path_with_shrunken_protocol(tmp_path):
    P = json.loads(PROTO.read_text())
    P["design"]["production"].update({"C": 3, "T": 4, "k_c": 2, "k_t": 2, "spectrum": {"top": [6.0, 5.0, 4.0, 3.0], "gap": 1.5, "tail_min": 0.05},
                                      "thetas": [0.0, 0.3], "N_core": [40, 80], "N_anchor": [200], "R": 2, "R_pop": 4, "als_n_iter": 60,
                                      "pop_als_n_iter": 200, "pilot_seeds": [3000, 3009], "confirmation_seeds": [4000, 4009]})
    proto = tmp_path / "PROTO_SHRUNK.json"
    proto.write_text(json.dumps(P))
    out = tmp_path / "prod"
    run = lambda *a: RN.main([a[0], "--protocol", str(proto)] + list(a[1:]))  # noqa: E731
    assert run("run", "--out", str(out), "--families", "confirmation") == 6 and not out.exists()
    assert run("run", "--out", str(out), "--families", "algebraic", "pilot") == 0
    assert run("reduce", "--out", str(out)) == 3                       # confirmation cells missing, no --allow-partial
    assert run("reduce", "--out", str(out), "--allow-partial") == 0     # pilot freeze
    red = json.loads((out / "REDUCED.json").read_text())
    assert red["phase"] == "pilot" and red["partial"] is True and red["pilot_complete"] and red["development_only"] is False
    assert red["pilot"]["per_theta"]["0.3"]["d_simultaneous"]["method"] == "max_abs_t_bootstrap"   # 10 seeds: the bootstrap branch
    fz = json.loads((out / RN.FROZEN_NAME).read_text())
    assert fz["development_only"] is False and fz["pilot_cells"] == 60
    assert run("run", "--out", str(out), "--families", "confirmation") == 0
    assert run("reduce", "--out", str(out)) == 0
    red = json.loads((out / "REDUCED.json").read_text())
    assert red["phase"] == "confirmation" and red["partial"] is False and red["reconciliation"]["complete"]
    assert red["ledger"]["charged"]["fits"] == 120 * 3 and red["ledger"]["charged"]["solves"] == 14
    cells = R.read_cells(out)
    roles = {js["role"] for files in cells.values() for _, js in files}
    assert roles == {"pilot", "confirmation", "algebraic"} and not any(js["development_only"] for files in cells.values() for _, js in files)
    for th, p in red["confirmation"]["per_theta"].items():
        assert p["sign_profile"]["profile"] is not None and len(p["by_N"]) == 3
    for name, sc in red["confirmation"]["scoring"].items():
        assert sc["selected"] == fz["selection"][name]["selected"]


# --- the fit-accounting guard is a real check ------------------------------------------------
def _ledger_records(out):
    return [json.loads(l) for l in (out / "EXECUTION_LEDGER.jsonl").read_text().splitlines() if l.strip()]


def test_fit_accounting_guard_can_fail(tmp_path, monkeypatch):
    """A fit cell reports 1 PCA + the ALS restarts actually returned; declare one
    more in the matrix and the run is refused (exit 4), charged, with no cell file."""
    P = PR.load_protocol(PROTO)
    real = RN.build_matrix
    target = next(c for c in real(P, True).cells
                  if c["n_fits"] > 0 and c["params"].get("phase") == "pilot")

    def misdeclared(P_, fixture):
        m = real(P_, fixture)
        for c in m.cells:
            if c["cell_id"] == target["cell_id"]:
                c["n_fits"] = int(c["n_fits"]) + 1
        return m

    monkeypatch.setattr(RN, "build_matrix", misdeclared)
    out = tmp_path / "misdeclared"
    assert _main(["run", "--out", str(out), "--fixture", "--cells", target["cell_id"]]) == 4
    recs = _ledger_records(out)
    assert recs[-1]["state"] == "failed" and "accounting mismatch" in recs[-1]["error"]
    assert not list((out / "cells").glob("*.json"))


def test_fit_cells_report_the_instrumented_pca_and_als_counts(full_run):
    for files in R.read_cells(full_run).values():
        for _, js in files:
            assert js["fits_observed"]["total"] == js["fits_used"]
            if js["result"]["kind"] == "fit":
                by = js["fits_observed"]["by_label"]
                assert by["pca"] == 1 and by["als_restart"] == js["result"]["fits"]["als_restarts"]
