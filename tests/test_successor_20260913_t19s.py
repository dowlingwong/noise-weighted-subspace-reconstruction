"""Local acceptance tests for the T19-S v2 package (L1), covering the eight
required tests of successor_20260913/README.md with the real reduced runner
on synthetic fixtures, plus the archive-equivalence and regression checks the
design asks for.  Nothing here is a scientific result."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.paper1 import g1 as G  # noqa: E402
from experiments.paper1 import t19_coupling as T19  # noqa: E402
from experiments.mlst2026.successor_20260913.common import ledger as L, protocol as PR, results as RS  # noqa: E402
from experiments.mlst2026.successor_20260913.t19s import observables as O, runner as R, synthetic as S  # noqa: E402

PKG = REPO / "experiments" / "mlst2026" / "successor_20260913" / "t19s"
PROTOCOL = PKG / "PROTOCOL_T19S_v2.json"


def _main(argv: list[str], protocol: Path = PROTOCOL) -> int:
    return R.main([argv[0], "--protocol", str(protocol)] + argv[1:])


def _ledger(out: Path, P: dict) -> L.ExecutionLedger:
    return L.ExecutionLedger(out / "EXECUTION_LEDGER.jsonl", cap=P["budget"]["estimator_fits_cap"], package="t19s", protocol_sha256=P["_sha256"])


@pytest.fixture(scope="module")
def P():
    return PR.load_protocol(PROTOCOL)


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory):
    """One fixture run + reduce shared by the read-only tests; destructive tests copy it."""
    out = tmp_path_factory.mktemp("t19s_fixture") / "run"
    assert _main(["run", "--out", str(out), "--fixture"]) == 0
    assert _main(["reduce", "--out", str(out)]) == 0
    return out


def _copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


# ---------------------------------------------------------------------------
# 1. validate / enumerate compute nothing; counts come from the matrix
# ---------------------------------------------------------------------------
def test_validate_and_enumerate_compute_nothing(tmp_path, P):
    assert _main(["validate"]) == 0
    out = tmp_path / "enum"
    assert _main(["enumerate", "--out", str(out)]) == 0
    assert (out / "run_matrix.jsonl").exists() and not (out / "cells").exists()
    summ = json.loads((out / "run_matrix_summary.json").read_text())
    assert summ["cells"] == 136 and summ["fits"] == 3840 == P["budget"]["fits_planned_from_matrix"]
    assert summ["fits"] <= P["budget"]["estimator_fits_cap"]
    assert summ["by_role"]["pilot"] == {**summ["by_role"]["pilot"], "cells": 20, "fits": 640}
    assert summ["by_role"]["confirmation"]["cells"] == 100 and summ["by_role"]["confirmation"]["fits"] == 3200
    assert summ["by_role"]["algebraic"]["fits"] == 0 and summ["by_role"]["control"]["fits"] == 0
    assert summ["by_role"]["algebraic"]["cells"] == 7 and summ["by_role"]["control"]["cells"] == 9
    # every paired cell plans exactly 1 oracle + 2*8 + 2*6 + 3 fits
    rows = [json.loads(l) for l in (out / "run_matrix.jsonl").read_text().splitlines()]
    assert all(r["n_fits"] == 32 and r["n_solves"] == 29 for r in rows if r["family"] == "paired")
    assert all(r["n_solves"] == len(r["params"]["eps_grid"]) for r in rows if r["family"] != "paired")


# ---------------------------------------------------------------------------
# 2. fixture end to end with the declared schema
# ---------------------------------------------------------------------------
def test_fixture_end_to_end_has_declared_schema(fixture_run):
    red = json.loads((fixture_run / "REDUCED.json").read_text())
    assert red["development_only"] is True and red["reconciliation"]["complete"] and red["partial"] is False
    assert red["scientific_checks"]["passed"] is True
    assert red["stage"] == "development"
    for k in ("order_table", "controls"):
        assert k in red["deterministic"]
    conf = red["confirmation"]
    for k in ("a_theorem_domain_check", "b_first_order_agreement_norm", "c_residual_orders", "d_training_sample_arm", "census_by_arm"):
        assert k in conf
    assert conf["development_only"] is True
    sel = json.loads((fixture_run / R.SELECTION_FILE).read_text())
    assert sel["schema"] == R.SELECTION_SCHEMA and sel["development_only"] is True
    # every cell file carries the result schema and the development label
    for files in RS.read_cells(fixture_run).values():
        for _, js in files:
            RS.check_cell_schema(js, R.RESULT_REQUIRED)
            assert js["development_only"] is True
            if js["result"]["kind"] == "paired":
                for rec in js["result"]["records"]:
                    for key in ("eta_used_for_classification", "eta_used_is", "domain_class", "lambda_min_IplusE", "eigensolver_residual", "bound", "observables", "basis"):
                        assert key in rec
                    assert rec["bound"]["violation"] >= 0.0
                    for obs in O.OBSERVABLES:
                        assert obs in rec["observables"]
            else:
                assert js["result"]["model"] in ("covariance", "additive")
                assert js["result"]["records"][0]["eta_basis"].startswith("population")


# ---------------------------------------------------------------------------
# 3. scientific controls and the predeclared order table
# ---------------------------------------------------------------------------
def test_controls_and_order_table_from_fixture(fixture_run):
    red = json.loads((fixture_run / "REDUCED.json").read_text())
    ctr = red["deterministic"]["controls"]
    assert all(c["max_projector_displacement_fro"] < 1e-12 and c["absolute_branch"] for c in ctr["zero"])
    assert all(c["max_projector_displacement_fro"] < 1e-9 and c["max_graph_displacement_fro"] < 1e-9 for c in ctr["block_diagonal"])
    table = red["deterministic"]["order_table"]
    for model in ("covariance", "additive_control"):
        for fam, entry in table[model].items():
            pred = entry["predeclared"]
            for dd in entry["directions"]:
                assert dd["within_tolerance"], (model, fam, dd["slopes"])
                for obs in O.OBSERVABLES:
                    assert abs(dd["slopes"][obs] - pred[obs]) <= 0.25, (model, fam, obs, dd["slopes"][obs])
                assert dd["max_identity_error"] < 1e-10
    # the order table itself is the predeclared one
    assert table["covariance"]["generic"]["predeclared"] == {"graph": 2, "projector_matrix": 2, "projector_norm": 2, "angle": 2}
    assert table["covariance"]["pure_coupling"]["predeclared"] == {"graph": 3, "projector_matrix": 2, "projector_norm": 3, "angle": 3}
    assert table["additive_control"]["pure_coupling"]["predeclared"]["graph"] == 3
    # 2-D example inside the run
    two_d = table["covariance"]["two_d_example"]["directions"][0]["two_d"]
    for p in two_d:
        assert p["abs_error_t"] < 1e-12
        if p["reference"] is not None:
            assert abs(p["residual_over_eps2_measured"] - p["reference"]) < 1e-5
    # paired-cell invariants held
    for files in RS.read_cells(fixture_run).values():
        for _, js in files:
            assert all(v is not False for v in js["result"]["checks"].values()), js["result"]["checks"]


def test_predeclared_orders_match_protocol(P):
    cov = P["predeclared_orders"]["covariance_model"]
    for fam in ("generic", "pure_coupling", "two_d_example"):
        assert cov[fam] == O.PREDECLARED_ORDERS[fam]
    assert P["predeclared_orders"]["additive_control"]["pure_coupling"] == O.PREDECLARED_ORDERS["pure_coupling"]
    assert P["predeclared_orders"]["additive_control"]["generic"] == O.PREDECLARED_ORDERS["generic"]


def test_two_d_regression_example_direct():
    """A = diag(2,1), E = eps[[0,1],[1,0]], k = 1 (review finding S1)."""
    A, Q, lam = O.TWO_D_A, np.eye(2), np.array([2.0, 1.0])
    for eps, ref in O.TWO_D_REFERENCE.items():
        sol = O.covariance_model_subspace(A, eps * O.TWO_D_E0, 1)
        X = O.graph_of(sol["V"], Q, 1)
        t = -2 * eps / (1 + np.sqrt(1 + 8 * eps ** 2))
        assert abs(X[0, 0] - t) < 1e-12
        X1, ident = O.first_order_graph(lam, eps * O.TWO_D_E0, 1, "covariance")
        assert ident and abs(X1[0, 0] + eps) < 1e-15
        obs = O.observables(sol["V"], Q, lam, 1, X1)
        assert abs(obs["projector_matrix"]["residual_fro"] / eps ** 2 - ref) < 1e-5
        # the graph, norm and angle residuals are cubic; the matrix residual quadratic
        assert obs["graph"]["residual_fro"] < 10 * eps ** 3
        assert obs["projector_norm"]["residual"] < 10 * eps ** 3
        assert obs["angle"]["residual"] < 10 * eps ** 3
        assert obs["identity_abs_error"] < 1e-14
    # exact identity on a random graph
    Xr = np.random.default_rng(3).standard_normal((10, 2)) * 0.05
    V = O.orth(np.vstack([np.eye(2), Xr]))
    obs = O.observables(V, np.eye(12), np.r_[3.0, 2.0, np.ones(10)], 2, np.zeros((10, 2)))
    assert obs["identity_abs_error"] < 1e-12 and obs["absolute_branch"] and obs["graph"]["ratio"] is None


def test_reuses_the_audited_archive_functions():
    """The generic-d code reproduces the frozen G1 and the archived T19 estimator at d = 256."""
    g = S.make_generator(256)
    g0 = G.make_g1()
    assert np.array_equal(g.A_pop, g0.A_pop) and np.array_equal(g.Sigma, g0.Sigma) and np.array_equal(g.lam, g0.lam)
    assert G.check_against_recorded(g)["matches_recorded"]
    rng = np.random.default_rng(11)
    X, _ = S.draw_training(g, rng, 300)
    X0, _ = G.draw_training(g0, np.random.default_rng(11), 300)
    assert np.array_equal(X, X0)
    E = T19.make_E_family(g.Q, "F", 0.05, np.random.default_rng(1))
    Sigma_hat = S.sigma_hat_from_E(g, E)
    P1, Si1, d1 = S.fit_subspace(X, Sigma_hat, 2, floor=False)
    P0, Si0, d0, _ = T19.fit_subspace(X, Sigma_hat, floor=False)
    assert np.allclose(P1, P0, atol=1e-12) and np.allclose(Si1, Si0)
    V = S.whitened_basis(g, P1)
    assert np.allclose(O.graph_of(V, g.Q, 2), T19.displacement(V, g.Q))
    Eq = g.Q.T @ E @ g.Q
    X1, _ = O.first_order_graph(g.lam, Eq, 2, "covariance")
    _, cov_only = T19.first_order(np.zeros_like(Eq), Eq, g.lam)
    assert np.allclose(X1, cov_only)
    # the perturbed-subspace solve agrees with the archive's perturbed_spectrum
    sol = O.covariance_model_subspace(g.A_pop, E, 2)
    spec = T19.perturbed_spectrum(g.A_pop, g.Q, E)
    assert abs(O.sin_theta_max(sol["V"], g.Q[:, :2]) - spec["retained_angle_sin"]) < 1e-10
    assert abs(sol["separation"] - spec["separation"]) < 1e-10
    # the estimator really returns the invariant subspace of A_hat (I+E)^{-1}
    A_hat, lam_hat, Q_hat = S.sample_operator(g, X)
    E_s = 0.05 * O.make_direction("pure_coupling", Q_hat, 2, np.random.default_rng(2), lam_b=lam_hat)
    P2, _, _ = S.fit_subspace(X, S.sigma_hat_from_E(g, E_s), 2, floor=False)
    sol2 = O.covariance_model_subspace(A_hat, E_s, 2)
    # (sin_theta_max resolves no better than sqrt(machine eps) ~ 1.5e-8)
    assert O.sin_theta_max(S.whitened_basis(g, P2), sol2["V"]) < 1e-6
    # streamed calibration equals the archived estimators on the same draw
    gs = S.make_generator(32)
    calib = S.calibration_covariances(gs, np.random.default_rng(3), [64, 300], chunk=128)
    pool = S.draw_noise(gs, np.random.default_rng(3), 300)
    for m in (64, 300):
        assert np.allclose(calib[m]["S_full"], T19.empirical_cov(pool[:m]), atol=1e-13)
        assert np.allclose(calib[m]["S_toeplitz"], T19.toeplitz_cov(pool[:m]), atol=1e-13)


# ---------------------------------------------------------------------------
# 4. namespaces disjoint; fixture refuses pilot/confirmation cells
# ---------------------------------------------------------------------------
def test_namespaces_disjoint_and_fixture_refuses_pilot(tmp_path, P):
    prod = R.build_matrix(P, False)
    pilot = {c["seed"] for c in prod.cells if c["role"] == "pilot"}
    conf = {c["seed"] for c in prod.cells if c["role"] == "confirmation"}
    assert pilot and conf and not (pilot & conf)
    assert all(PR.namespace_of_seed(P, s) == "pilot" for s in pilot)
    assert all(PR.namespace_of_seed(P, s) == "confirmation" for s in conf)
    assert all(c["seed"] is None for c in prod.cells if c["role"] in ("algebraic", "control"))
    fx = R.build_matrix(P, True)
    assert all(PR.namespace_of_seed(P, c["seed"]) == "development" for c in fx.cells if c["seed"] is not None)
    assert not {c["cell_id"] for c in fx.cells if c["role"] == "development"} & {c["cell_id"] for c in prod.cells}
    out = tmp_path / "refuse"
    assert _main(["run", "--out", str(out), "--fixture", "--roles", "pilot"]) == 6
    assert _main(["run", "--out", str(out), "--fixture", "--roles", "confirmation"]) == 6
    assert not (out / "cells").exists()


# ---------------------------------------------------------------------------
# 5. reduce refuses partial, duplicated and tampered runs (exit 3)
# ---------------------------------------------------------------------------
def test_reduce_refuses_partial_duplicate_and_tampered(tmp_path, fixture_run):
    part = _copy(fixture_run, tmp_path / "partial")
    victim = sorted((part / "cells").glob("*.json"))[0]
    victim.unlink()
    assert _main(["reduce", "--out", str(part)]) == 3
    assert _main(["reduce", "--out", str(part), "--allow-partial"]) == 0
    dup = _copy(fixture_run, tmp_path / "dup")
    src = sorted((dup / "cells").glob("*.json"))[0]
    js = json.loads(src.read_text())
    js["cell_id"] = "deadbeefdeadbeef"
    (dup / "cells" / "deadbeefdeadbeef.json").write_text(json.dumps(js))
    assert _main(["reduce", "--out", str(dup)]) == 3
    tam = _copy(fixture_run, tmp_path / "tamper")
    p = [q for q in sorted((tam / "cells").glob("*.json")) if json.loads(q.read_text())["result"]["kind"] == "paired"][0]
    js = json.loads(p.read_text())
    js["result"]["records"][0]["bound"]["violation"] = 0.0
    p.write_text(json.dumps(js))
    assert _main(["reduce", "--out", str(tam)]) == 3


# ---------------------------------------------------------------------------
# 6. injected failure is charged; the retry accumulates
# ---------------------------------------------------------------------------
def test_injected_failure_is_charged_and_retry_accumulates(tmp_path, P):
    out = tmp_path / "fail"
    fx = R.build_matrix(P, True)
    victim = [c for c in fx.cells if c["role"] == "development"][0]
    assert _main(["run", "--out", str(out), "--fixture", "--fail-cell", victim["cell_id"]]) == 1
    led = _ledger(out, P)
    ch = led.charged()
    assert ch["by_state"]["failed"] == 1 and ch["fits"] == victim["n_fits"] == 15
    assert _main(["run", "--out", str(out), "--fixture"]) == 0
    ch = led.charged()
    assert ch["by_state"]["failed"] == 1 and ch["by_state"]["completed"] == len(fx.cells)
    assert ch["fits"] == victim["n_fits"] + sum(c["n_fits"] for c in fx.cells)
    assert _main(["reduce", "--out", str(out)]) == 0


# ---------------------------------------------------------------------------
# 7. mutation: a wrong first-order formula / swapped observables is rejected
#    by the package's scientific checks (exit 4), not by the digest check
# ---------------------------------------------------------------------------
def test_mutation_first_order_without_lambda_weight_is_rejected(tmp_path, monkeypatch):
    real = O.first_order_graph

    def mutated(lam, Eq, k, model="covariance"):
        # drops the -lambda_j weight of the covariance model (uses the additive formula)
        return real(lam, Eq, k, "additive")

    monkeypatch.setattr(O, "first_order_graph", mutated)
    out = tmp_path / "mut1"
    assert _main(["run", "--out", str(out), "--fixture", "--roles", "algebraic", "control"]) == 0
    rc = _main(["reduce", "--out", str(out), "--allow-partial"])
    assert rc == 4
    assert not (out / "REDUCED.json").exists()
    # the cells themselves recorded the failing checks (the reducer did not need the digest to notice)
    failing = [k for files in RS.read_cells(out).values() for _, js in files for k, v in js["result"]["checks"].items() if v is False]
    assert any(k.startswith("order_graph") for k in failing) and any(k.startswith("two_d") for k in failing)
    monkeypatch.undo()
    # an untouched run of the same cells reduces cleanly
    out2 = tmp_path / "clean"
    assert _main(["run", "--out", str(out2), "--fixture", "--roles", "algebraic", "control"]) == 0
    assert _main(["reduce", "--out", str(out2), "--allow-partial"]) == 0


def test_mutation_swapped_observables_is_rejected(tmp_path, monkeypatch):
    real = O.observables

    def swapped(V, Q, lam, k, X1):
        obs = real(V, Q, lam, k, X1)
        obs["graph"], obs["projector_matrix"] = obs["projector_matrix"], obs["graph"]
        obs["graph"]["residual_fro"] = obs["graph"].pop("residual_op")
        obs["projector_matrix"]["residual_op"] = obs["projector_matrix"].pop("residual_fro")
        obs["projector_matrix"]["residual_fro"] = obs["projector_matrix"]["residual_op"]
        return obs

    monkeypatch.setattr(O, "observables", swapped)
    out = tmp_path / "mut2"
    assert _main(["run", "--out", str(out), "--fixture", "--roles", "algebraic"]) == 0
    assert _main(["reduce", "--out", str(out), "--allow-partial"]) == 4


# ---------------------------------------------------------------------------
# 8. resource estimate from the fixture timings
# ---------------------------------------------------------------------------
def test_resource_estimate_from_fixture(fixture_run, P):
    # the common `estimate` prices the matrix of the run root it is pointed at (the fixture matrix here)
    assert _main(["estimate", "--out", str(fixture_run)]) == 0
    est = json.loads((fixture_run / "RESOURCE_ESTIMATE.json").read_text())
    assert est["unmeasured_families"] == [] and est["total_cpu_seconds_point"] > 0
    assert est["families"]["paired"]["scale"] == P["budget"]["fixture_to_production_scale"]["paired"]
    # the package-local `estimate-production` prices the PRODUCTION matrix with the fixture timings
    assert _main(["estimate-production", "--out", str(fixture_run)]) == 0
    prod = json.loads((fixture_run / "RESOURCE_ESTIMATE_PRODUCTION.json").read_text())
    assert prod["unmeasured_families"] == [] and prod["total_cpu_seconds_point"] >= est["total_cpu_seconds_point"]  # common now prices production too
    assert prod["counts"]["fits"] == 3840 and prod["retry_allowance"]["fits_headroom_under_cap"] == 4800 - 3840
    assert prod["stages"] == {"pilot_cells": 20, "confirmation_cells": 100, "deterministic_cells": 16}
    assert prod["families"]["paired"]["production_cpu_seconds"] == pytest.approx(prod["families"]["paired"]["fixture_seconds_per_cell"] * P["budget"]["fixture_to_production_scale"]["paired"] * 120)


# ---------------------------------------------------------------------------
# pilot -> freeze -> confirmation flow on a shrunken TEST copy of the protocol
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_protocol(tmp_path_factory):
    P = json.loads(PROTOCOL.read_text())
    P["version"] = "TEST COPY (shrunken production design; not the frozen protocol)"
    fx = P["design"]["fixture"]
    P["design"]["production"] = {
        "d": fx["d"], "k": fx["k"], "n_train": fx["n_train"], "generator": "TEST: g1_like_d32",
        "deterministic": {**fx["deterministic"], "families": {"pure_coupling": 1, "generic": 1, "block_diagonal": 1, "zero": 1, "two_d_example": 1},
                          "additive_control_families": {"pure_coupling": 1, "generic": 1}},
        "paired": {**fx["paired"], "pilot_seeds": [1000, 1001], "confirmation_seeds": [2000, 2001]},
    }
    P["budget"]["estimator_fits_cap"] = 120
    P["budget"]["fits_planned_from_matrix"] = 60
    p = tmp_path_factory.mktemp("t19s_small") / "PROTOCOL_TEST_COPY.json"
    p.write_text(json.dumps(P, indent=1))
    return p


def test_pilot_freeze_confirmation_flow(tmp_path, small_protocol):
    out = tmp_path / "flow"
    Ps = PR.load_protocol(small_protocol)
    assert _main(["validate"], small_protocol) == 0
    # a confirmation cell refuses to start before the pilot selection is frozen (charged failure, exit 1)
    assert _main(["run", "--out", str(out), "--roles", "confirmation"], small_protocol) == 1
    led = _ledger(out, Ps)
    assert led.charged()["by_state"]["failed"] == 1 and led.charged()["fits"] == 15
    assert not (out / R.SELECTION_FILE).exists()
    # pilot cells, then the pilot-stage reduce (partial by construction) freezes the selection
    assert _main(["run", "--out", str(out), "--roles", "pilot"], small_protocol) == 0
    assert _main(["reduce", "--out", str(out)], small_protocol) == 3
    assert _main(["reduce", "--out", str(out), "--allow-partial"], small_protocol) == 0
    red = json.loads((out / "REDUCED.json").read_text())
    assert red["stage"] == "pilot" and red["partial"] is True and red["selection"]["written_now"] is True
    sel_path = out / R.SELECTION_FILE
    sel = json.loads(sel_path.read_text())
    assert sel["development_only"] is False and sel["source_role"] == "pilot"
    assert set(sel["engineered"]["eps_in_domain"]) <= set(Ps["design"]["production"]["paired"]["eps_grid"])
    sha_before = RS.sha256_file(sel_path)
    # re-reducing the pilot does not rewrite the frozen selection
    assert _main(["reduce", "--out", str(out), "--allow-partial"], small_protocol) == 0
    assert RS.sha256_file(sel_path) == sha_before
    # confirmation + deterministic cells, then the complete reduce
    assert _main(["run", "--out", str(out)], small_protocol) == 0
    assert _main(["reduce", "--out", str(out)], small_protocol) == 0
    red = json.loads((out / "REDUCED.json").read_text())
    assert red["stage"] == "confirmation" and red["partial"] is False and red["development_only"] is False
    conf = red["confirmation"]
    assert conf["n_cells"] == 2 and conf["partial"] is False
    assert conf["a_theorem_domain_check"]["pooled"]["n_scored"] >= 1
    assert "C" in conf["c_residual_orders"]["by_family"] and "F" in conf["c_residual_orders"]["by_family"]
    assert conf["stress_label"].startswith("eta in [0.2, 1.3] is OUTSIDE")
    for files in RS.read_cells(out).values():
        for _, js in files:
            if js["role"] == "confirmation":
                assert js["result"]["selection_sha256"] == RS.sha256_file(sel_path)
    # a tampered frozen selection is refused at the confirmation reduce
    sel["engineered"]["eps_in_domain"] = []
    sel_path.write_text(json.dumps(sel, indent=1, sort_keys=True))
    assert _main(["reduce", "--out", str(out)], small_protocol) == 3
    # the ledger charged everything, including the refused confirmation attempt
    ch = led.charged()
    assert ch["by_state"]["failed"] == 1 and ch["fits"] == 15 + 4 * 15
