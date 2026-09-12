"""T21-S (L3) acceptance tests: the eight README tests plus the package's own
scientific checks, all against the real reduced runner on the d = 24 fixture.

Nothing here is a scientific result: every run uses the development seed
namespace and is marked development_only.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.mlst2026.successor_20260913.common import ledger as L, matrix as M, protocol as PR, results as R  # noqa: E402
from experiments.mlst2026.successor_20260913.t21s import geometry as GEO, predictor as PRED, runner as RUN  # noqa: E402

PROTO = REPO / "experiments/mlst2026/successor_20260913/t21s/PROTOCOL_T21S_v2.json"
PILOT_FAMS = ["pilot-orthogonal-hi", "pilot-orthogonal-lo", "pilot-correlated-hi", "pilot-correlated-lo", "algebraic"]
CONF_FAMS = ["confirmation-orthogonal-hi", "confirmation-orthogonal-lo", "confirmation-correlated-hi", "confirmation-correlated-lo"]


def _main(*argv) -> int:
    return RUN.main(list(argv))


def _P():
    return PR.load_protocol(PROTO)


def _ledger(root: Path) -> L.ExecutionLedger:
    return L.ExecutionLedger(root / "EXECUTION_LEDGER.jsonl", cap=10 ** 6, package="t21s", protocol_sha256="x")


# ----------------------------------------------------------------------------- shared fixture run
@pytest.fixture(scope="module")
def full_fixture_root(tmp_path_factory) -> Path:
    """pilot -> reduce (freeze) -> confirmation -> reduce, once for the module."""
    root = tmp_path_factory.mktemp("t21s_full") / "run"
    assert _main("run", "--protocol", str(PROTO), "--out", str(root), "--fixture", "--families", *PILOT_FAMS) == 0
    assert not (root / RUN.FROZEN_NAME).exists()
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(root), "--allow-partial") == 0
    assert (root / RUN.FROZEN_NAME).exists(), "the pilot reducer must write the frozen predictor"
    assert _main("run", "--protocol", str(PROTO), "--out", str(root), "--fixture", "--families", *CONF_FAMS) == 0
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(root)) == 0
    return root


# ----------------------------------------------------------------------------- README tests 1, 8 and help
def test_validate_and_enumerate_compute_nothing(tmp_path, capsys):
    assert _main("validate", "--protocol", str(PROTO)) == 0
    out = tmp_path / "enum"
    assert _main("enumerate", "--protocol", str(PROTO), "--out", str(out)) == 0
    assert (out / "run_matrix.jsonl").exists() and not (out / "cells").exists()
    summ = json.loads((out / "run_matrix_summary.json").read_text())
    P = _P()
    assert summ["fits"] == 2496 and summ["solves"] == 2498 and summ["cells"] == 2499
    assert summ["fits"] == P["budget"]["planned_fits"] <= P["budget"]["estimator_fits_cap"]
    assert summ["by_role"]["pilot"]["cells"] == 416 and summ["by_role"]["confirmation"]["cells"] == 2080 and summ["by_role"]["algebraic"]["cells"] == 3
    # 104 configurations per phase, mechanically
    cells = [json.loads(l) for l in (out / "run_matrix.jsonl").read_text().splitlines()]
    confs = {RUN.config_key(c["params"]) for c in cells if c["role"] == "confirmation"}
    assert len(confs) == 104


def test_help_never_computes():
    with pytest.raises(SystemExit) as e:
        _main("--help")
    assert e.value.code == 0


def test_run_refuses_archive_roots():
    bad = REPO / "results" / "t21s_should_never_exist"
    assert _main("run", "--protocol", str(PROTO), "--out", str(bad), "--fixture") == 6
    assert not bad.exists()


def test_estimate_from_fixture_timings(full_fixture_root):
    assert _main("estimate", "--protocol", str(PROTO), "--out", str(full_fixture_root)) == 0
    est = json.loads((full_fixture_root / "RESOURCE_ESTIMATE.json").read_text())
    assert est["unmeasured_families"] == []
    proj = est["production_projection"]
    assert proj["counts"]["fits"] == 2496 and proj["unmeasured_families"] == [] and proj["total_cpu_seconds_point"] > 0
    assert proj["retry_headroom"]["fits_headroom_under_cap"] == 254


# ----------------------------------------------------------------------------- README test 2: end-to-end schema
def test_fixture_end_to_end_has_declared_schema(full_fixture_root):
    red = json.loads((full_fixture_root / "REDUCED.json").read_text())
    assert red["development_only"] is True and red["partial"] is False and red["reconciliation"]["complete"]
    for k in ("phase_completeness", "controls", "algebraic", "configurations", "pilot", "confirmation", "frozen_predictor", "historical_verdicts_preserved"):
        assert k in red, k
    assert all(red["phase_completeness"][p]["complete"] for p in ("pilot", "confirmation", "algebraic"))
    assert "NOT met" in red["historical_verdicts_preserved"]
    cells = R.read_cells(full_fixture_root)
    assert len(cells) == 419
    for files in cells.values():
        _, js = files[0]
        assert js["development_only"] is True and js["seed"] is None or 900000 <= js["seed"] <= 900999
        if js["role"] == "algebraic":
            assert set(RUN.ALGEBRAIC_RESULT_REQUIRED) <= set(js["result"]) and js["result"]["ok"] is True
        else:
            assert set(RUN.DATA_RESULT_REQUIRED) <= set(js["result"])
            assert js["fits_used"] == 1 and js["solves_used"] == 1
    conf = red["confirmation"]
    assert conf["complete"] and conf["all_configurations_complete"] and conf["overall"]["n_rows"] == 104 * 4
    assert set(conf["verdict_by_geometry"]) == {"orthogonal", "correlated"}
    # uncertainty is across seeds: every scored row has n_seeds = 2 (the fixture allocation), never 4 or 8
    assert {r["n_seeds"] for r in conf["rows"]} == {2}


# ----------------------------------------------------------------------------- README test 3: scientific controls
def test_gram_invariants_realised_to_1e10():
    for d in (24, 256):
        for geo in GEO.GEOMETRIES:
            inv = GEO.check_invariants(d, geo, theta=np.array([3.0, 3.0, 3.0, 0.0]))
            assert inv["ok"], inv
            assert inv["gram_max_abs_error"] <= 1e-10 and inv["unit_norm_max_abs_error"] <= 1e-10
            assert inv["population_spectrum_max_abs_error"] <= 1e-10
    U = GEO.directions(64, "correlated")
    assert np.allclose(U.T @ U, GEO.gram("correlated"), atol=1e-12)
    assert np.allclose(np.linalg.norm(U, axis=0), 1.0)
    # the correlated arm carries off-diagonals of both signs and an exact zero
    G = GEO.gram("correlated")
    off = G[np.triu_indices(4, 1)]
    assert (off > 0).any() and (off < 0).any() and (off == 0).any()
    # the frozen construction is reproducible
    assert np.array_equal(GEO.directions(40, "orthogonal"), GEO.directions(40, "orthogonal"))


def test_all_zero_control_is_at_chance_direct():
    """K = 0: recovery ~ Beta(k/2,(d-k)/2) with mean k/d; 40 development seeds, tolerance 4 SE."""
    P = _P()
    d, n, k = 24, 12, 4
    params = {"geometry": "correlated", "gamma_arm": "hi", "d": d, "n_train": n, "k": k, "kind": "all_zero", "varied_direction": None, "ratio": None, "phase": "pilot"}
    per_seed = []
    for seed in range(900500, 900540):
        res = RUN.run_data_cell(P, {"cell_id": "direct", "seed": seed, "params": params})
        assert res["predicted_recovery"] == [0.0] * 4
        per_seed.append(float(np.mean(res["recovery"])))
    se = RUN._beta_sd(k, d) / np.sqrt(len(per_seed))
    assert abs(np.mean(per_seed) - k / d) <= 4 * se, (np.mean(per_seed), k / d, se)


def test_all_zero_control_checked_in_reduce(full_fixture_root):
    red = json.loads((full_fixture_root / "REDUCED.json").read_text())
    az = red["controls"]["all_zero"]
    assert len(az) == 4 and all(v["pass"] for v in az.values())
    for v in az.values():
        assert v["chance_level"] == pytest.approx(4 / 24) and v["tolerance_abs"] == pytest.approx(5 * RUN._beta_sd(4, 24) / np.sqrt(v["n_seeds"]))
        assert v["max_predicted_recovery"] == 0.0


def test_orthogonal_unique_equals_ordinary_and_correlated_differs(full_fixture_root):
    red = json.loads((full_fixture_root / "REDUCED.json").read_text())
    assert red["controls"]["orthogonal_unique_equals_ordinary"]["max_abs_diff"] <= 1e-9
    assert red["controls"]["correlated_unique_differs"]["max_abs_diff"] > 0.05
    # per cell, from the raw outputs
    for files in R.read_cells(full_fixture_root).values():
        _, js = files[0]
        if js["role"] == "algebraic":
            continue
        r = js["result"]
        assert all(r["unique_defined"]), "every unique component of both Grams is defined (min norm ratio 0.46)"
        if js["params"]["geometry"] == "orthogonal":
            assert max(abs(a - b) for a, b in zip(r["recovery"], r["unique_recovery"])) <= 1e-9
            assert r["unique_norm_ratio"] == pytest.approx([1.0] * 4)
        else:
            assert r["unique_norm_ratio"] == pytest.approx([0.4639, 0.5632, 0.6604, 0.8804], abs=1e-3)


def test_unique_component_undefined_flag():
    """A direction inside the span of the others yields null unique recovery with the flag."""
    U = GEO.directions(30, "orthogonal")
    U2 = U.copy()
    U2[:, 3] = (U[:, 0] + U[:, 1]) / np.sqrt(2)          # u_4 in span{u_1, u_2}
    Up, ratio = GEO.unique_components(U2)
    assert ratio[3] < GEO.UNIQUE_UNDEFINED_BELOW
    lam, V = PRED.population_eigenpairs(GEO.signal_covariance(U2, np.array([3.0, 3.0, 3.0, 3.0])), 4)
    pred = PRED.predict_all(U2, Up, [r >= GEO.UNIQUE_UNDEFINED_BELOW for r in ratio], GEO.signal_covariance(U2, np.array([3.0, 3.0, 3.0, 3.0])), 0.5)
    # u_1, u_2, u_4 are now linearly dependent (each lies in the span of the others); u_3 is not
    assert [p is None for p in pred["predicted_unique_recovery"]] == [True, True, False, True]
    assert pred["predicted_unique_recovery"][2] == pytest.approx(PRED.c2_isolated_spike(3.0, 0.5))


def test_predictor_sanity_zero_below_threshold_and_tends_to_one():
    for gamma in (0.25, 2.0):
        sg = np.sqrt(gamma)
        assert PRED.c2_isolated_spike(0.0, gamma) == 0.0
        assert PRED.c2_isolated_spike(0.99 * sg, gamma) == 0.0 and PRED.c2_isolated_spike(sg, gamma) == 0.0
        assert 0 < PRED.c2_isolated_spike(1.01 * sg, gamma) < 0.05
        assert PRED.c2_isolated_spike(2 * sg, gamma) == pytest.approx((1 - 0.25) / (1 + sg / 2))
        assert PRED.c2_isolated_spike(1e7 * sg, gamma) == pytest.approx(1.0, abs=1e-6)
        assert PRED.sanity(gamma)["ok"]
    # theta/sqrt(gamma) alone does not fix c2 (the archive lesson): same ratio, different gamma, different c2
    assert PRED.c2_isolated_spike(2 * np.sqrt(2.0), 2.0) != pytest.approx(PRED.c2_isolated_spike(2 * 0.5, 0.25))
    # loadings: in the orthogonal arm the predictor reduces to c2(theta_j); in the correlated arm an unexcited
    # direction is predicted above zero through its overlap with excited ones, while its unique component is 0
    d, gamma = 64, 0.5
    for geo, expect_mix in (("orthogonal", False), ("correlated", True)):
        U = GEO.directions(d, geo)
        Up, ratio = GEO.unique_components(U)
        theta = np.array([0.0, 3.0, 3.0, 3.0]) * np.sqrt(gamma)
        pred = PRED.predict_all(U, Up, [True] * 4, GEO.signal_covariance(U, theta), gamma)
        if expect_mix:
            assert pred["predicted_recovery"][0] > 0.2 and pred["predicted_unique_recovery"][0] == pytest.approx(0.0, abs=1e-12)
        else:
            assert pred["predicted_recovery"][0] == pytest.approx(0.0, abs=1e-12)
            assert pred["predicted_recovery"][1:] == pytest.approx([PRED.c2_isolated_spike(3 * np.sqrt(gamma), gamma)] * 3)


# ----------------------------------------------------------------------------- README test 4: namespaces
def test_namespaces_disjoint_and_fixture_refuses_pilot_and_confirmation(tmp_path):
    P = _P()
    m = RUN.build_matrix(P, False)
    pilot = {c["seed"] for c in m.cells if c["role"] == "pilot"}
    conf = {c["seed"] for c in m.cells if c["role"] == "confirmation"}
    assert pilot and conf and not (pilot & conf)
    assert all(PR.namespace_of_seed(P, s) == "pilot" for s in pilot) and all(PR.namespace_of_seed(P, s) == "confirmation" for s in conf)
    assert M.check_seed_roles(m, P) == []
    fx = RUN.build_matrix(P, True)
    dev = {c["seed"] for c in fx.cells if c["seed"] is not None}
    assert all(PR.namespace_of_seed(P, s) == "development" for s in dev) and not (dev & (pilot | conf))
    assert fx.matrix_hash() != m.matrix_hash()
    out = tmp_path / "fx"
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--roles", "pilot") == 6
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--roles", "confirmation") == 6
    assert not (out / "cells").exists()


# ----------------------------------------------------------------------------- README test 5: reduce refusals
def test_reduce_refuses_partial_duplicate_and_tampered(tmp_path):
    out = tmp_path / "part"
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--families", "pilot-orthogonal-hi", "algebraic") == 0
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(out)) == 3, "partial run must be refused without --allow-partial"
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(out), "--allow-partial") == 0
    red = json.loads((out / "REDUCED.json").read_text())
    # (on a fixture root common clears 'missing' under --allow-partial, so the phase block carries the truth)
    assert red["phase_completeness"]["pilot"]["complete"] is False and red["phase_completeness"]["pilot"]["observed"] == 52
    assert not (out / RUN.FROZEN_NAME).exists(), "an incomplete pilot must not freeze"
    # duplicated cell file
    src = sorted((out / "cells").glob("*.json"))[0]
    js = json.loads(src.read_text())
    js["cell_id"] = "deadbeefdeadbeef"
    dup = out / "cells" / "deadbeefdeadbeef.json"
    dup.write_text(json.dumps(js))
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(out), "--allow-partial") == 3
    dup.unlink()
    # tampered cell (digest differs from the ledger)
    js = json.loads(src.read_text())
    js["result"]["recovery"][0] = 0.5
    src.write_text(json.dumps(js))
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(out), "--allow-partial") == 3


# ----------------------------------------------------------------------------- README test 6: failure accounting
def test_injected_failure_is_charged_and_retry_accumulates(tmp_path):
    out = tmp_path / "fail"
    P = _P()
    fx = RUN.build_matrix(P, True)
    fam = [c for c in fx.cells if c["family"] == "pilot-orthogonal-lo"]
    victim = fam[3]["cell_id"]
    rc = _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--families", "pilot-orthogonal-lo", "--fail-cell", victim)
    assert rc == 1
    ch = _ledger(out).charged()
    assert ch["by_state"]["failed"] == 1 and ch["by_state"]["completed"] == 3 and ch["fits"] == 4
    assert not (out / "cells" / f"{victim}.json").exists()
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--families", "pilot-orthogonal-lo") == 0
    ch = _ledger(out).charged()
    assert ch["by_state"]["failed"] == 1 and ch["by_state"]["completed"] == len(fam) and ch["fits"] == len(fam) + 1
    assert (out / "cells" / f"{victim}.json").exists()


# ----------------------------------------------------------------------------- freeze discipline
def test_confirmation_refused_without_frozen_predictor(tmp_path):
    out = tmp_path / "noconf"
    rc = _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--families", "confirmation-orthogonal-hi")
    assert rc == 6
    assert not (out / "cells").exists() and not (out / "EXECUTION_LEDGER.jsonl").exists(), "nothing may be charged"
    # a pilot run alone does not unlock confirmation either: the pilot must be REDUCED first
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--families", *PILOT_FAMS) == 0
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--families", "confirmation-orthogonal-hi") == 6
    assert _ledger(out).charged()["attempts"] == 211
    # and reducing confirmation cells whose frozen file is missing is refused at reduce (exit 4)
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(out), "--allow-partial") == 0
    fz = out / RUN.FROZEN_NAME
    assert fz.exists()
    content = json.loads(fz.read_text())
    assert content["predictor"]["formula_id"] == PRED.FORMULA_ID and content["predictor"]["coefficients_fitted_on_any_data"] is False
    assert content["n_confirmation_configurations"] == 104 and content["confirmation_permitted"] is True
    assert content["tolerance"]["primary_abs"] == 0.05 and content["development_only"] is True
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--families", "confirmation-orthogonal-hi") == 0
    fz.unlink()
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(out), "--allow-partial") == 4


def test_frozen_predictor_is_deterministic_and_tamper_evident(full_fixture_root, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(full_fixture_root, root)
    fz = root / RUN.FROZEN_NAME
    before = fz.read_bytes()
    red_a = json.loads((root / "REDUCED.json").read_text())
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(root)) == 0
    red_b = json.loads((root / "REDUCED.json").read_text())
    assert fz.read_bytes() == before, "a second reduce must not rewrite the freeze"
    for k in ("configurations", "confirmation", "controls", "pilot"):
        assert red_a[k] == red_b[k], k
    assert red_a["frozen_predictor"]["sha256"] == red_b["frozen_predictor"]["sha256"]   # (path differs: the tree was copied)
    # a tampered frozen prediction is caught by the scientific check, not by the digest (the frozen file has none)
    js = json.loads(fz.read_text())
    key = next(k for k in js["configurations"] if k.startswith("correlated|hi") and "|grid|" in k)
    js["configurations"][key]["predicted_recovery"][0] += 0.2
    fz.write_text(json.dumps(js))
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(root)) == 4


# ----------------------------------------------------------------------------- README test 7: mutations
def _correlated_grid_cells(P, n=3):
    fx = RUN.build_matrix(P, True)
    cs = [c for c in fx.cells if c["family"] == "pilot-correlated-hi" and c["params"]["kind"] == "grid" and c["params"]["ratio"] == 0.0]
    return [c["cell_id"] for c in cs[:n]]


def test_mutation_predictor_without_loadings_is_rejected(tmp_path, monkeypatch):
    """A runner whose predictor drops the loadings (assumes u_j = v_j) writes
    internally consistent files with valid digests; the reducer's independent
    recomputation from the exact K rejects them (exit 4)."""
    P = _P()
    out = tmp_path / "mut_loadings"
    real_predict = PRED.predict_all

    def no_loadings(U, Up, udef, K, gamma):
        # the archive's proxy: the Rayleigh quotient u_j^T K u_j as "the direction's spike", no loadings
        p = real_predict(U, Up, udef, K, gamma)
        rq = [float(U[:, j] @ K @ U[:, j]) for j in range(U.shape[1])]
        p["predicted_recovery"] = [PRED.c2_isolated_spike(t, gamma) for t in rq]
        return p

    monkeypatch.setattr(PRED, "predict_all", no_loadings)
    monkeypatch.setattr(RUN, "check_data_result", lambda *a, **k: None)                    # the corrupted runner does not self-check
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--cells", *_correlated_grid_cells(P)) == 0
    monkeypatch.undo()
    # digests are intact ...
    led = _ledger(out)
    for p, js in ((p, js) for files in R.read_cells(out).values() for p, js in files):
        assert R.sha256_file(p) == led.attempts()[js["attempt_uuid"]]["output_sha256"]
    # ... and the scientific check still refuses
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(out), "--allow-partial") == 4
    # the same mutation is invisible on the orthogonal arm (u_j^T K u_j = theta_j and loadings^2 = I there):
    # only the correlated arm can reject it, which is why that arm exists
    fx = RUN.build_matrix(P, True)
    orth = [c["cell_id"] for c in fx.cells if c["family"] == "pilot-orthogonal-hi" and c["params"]["kind"] == "grid"][:2]
    out2 = tmp_path / "mut_loadings_orth"
    monkeypatch.setattr(PRED, "predict_all", no_loadings)
    monkeypatch.setattr(RUN, "check_data_result", lambda *a, **k: None)
    assert _main("run", "--protocol", str(PROTO), "--out", str(out2), "--fixture", "--cells", *orth) == 0
    monkeypatch.undo()
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(out2), "--allow-partial") == 0


def test_mutation_unnormalised_direction_is_rejected(tmp_path, monkeypatch):
    """Recovery computed as ||P (1.5 u_j)||^2 without dividing by ||u_j||^2 leaves
    [0, 1]; the runner's own check refuses it at run time (charged failure, exit 4)
    and, if that check were disabled, the reducer refuses it."""
    P = _P()
    fx = RUN.build_matrix(P, True)
    strong = [c["cell_id"] for c in fx.cells if c["family"] == "pilot-orthogonal-lo" and c["params"]["kind"] == "all_strong"]
    monkeypatch.setattr(RUN, "_recovery", lambda Pb, w: float(np.sum((Pb.T @ (1.5 * w)) ** 2)))
    out = tmp_path / "mut_norm_run"
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--cells", strong[0]) == 4
    ch = _ledger(out).charged()
    assert ch["by_state"]["failed"] == 1 and ch["fits"] == 1 and not (out / "cells").exists()
    out2 = tmp_path / "mut_norm_reduce"
    monkeypatch.setattr(RUN, "check_data_result", lambda *a, **k: None)
    assert _main("run", "--protocol", str(PROTO), "--out", str(out2), "--fixture", "--cells", *strong) == 0
    monkeypatch.undo()
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(out2), "--allow-partial") == 4


def test_mutation_signal_in_all_zero_cell_is_rejected_by_the_control(tmp_path, monkeypatch):
    """A generator that leaks signal into the all-zero cells is caught by the
    all-zero control at reduce, independently of the digest."""
    P = _P()
    fx = RUN.build_matrix(P, True)
    az = [c["cell_id"] for c in fx.cells if c["family"] == "pilot-orthogonal-lo" and c["params"]["kind"] == "all_zero"]
    real_draw = RUN.draw_data

    def leaky(seed, n, U, theta):
        return real_draw(seed, n, U, np.full_like(theta, 30.0))

    monkeypatch.setattr(RUN, "draw_data", leaky)
    monkeypatch.setattr(RUN, "check_data_result", lambda *a, **k: None)
    out = tmp_path / "leak"
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--cells", *az) == 0
    monkeypatch.undo()
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(out), "--allow-partial") == 4


# ----------------------------------------------------------------------------- protocol / provenance content
def test_protocol_records_archive_facts_and_no_prior_freeze():
    P = _P()
    assert P["schema"] == PR.SCHEMA and P["package"] == "t21s"
    assert P["seed_namespaces"] == {"development": [900000, 900999], "pilot": [5000, 5099], "confirmation": [6000, 6299]}
    prov = P["provenance"]
    assert prov["supersedes"] is None and "none" in prov["prior_protocol"].lower()
    facts = json.dumps(prov["design_inputs_from_archive"])
    for s in ("-0.652", "0.594", "-0.368", "-0.315", "39.6", "15.1", "2.11", "0.54", "800", "3200", "all-zero", "constant-target"):
        assert s in facts, s
    assert "NOT met" in prov["design_inputs_from_archive"]["historical_verdicts"]
    assert "fixed allocation" in P["stopping_rule"] and "no early stop" in P["stopping_rule"]
    for k in ("estimator_fits_cap", "fixture_to_production_scale", "retry_policy"):
        assert k in P["budget"]
    findings = (PROTO.parent / "FINDINGS_T21S.md").read_text()
    assert "NOT met" in findings and "overlap" in findings
    for name in ("SCHEMA.md", "REMOTE_COMMANDS.md", "requirements.txt"):
        assert (PROTO.parent / name).exists()
    rc = (PROTO.parent / "REMOTE_COMMANDS.md").read_text()
    assert "<RUN_ROOT>" in rc and "<WORKERS>" in rc and "<DATA_ROOT>" in rc
