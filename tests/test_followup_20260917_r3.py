"""R3 (T21 finite-size recovery) acceptance tests.

Everything here runs against the real runner on the d = 16/32 development
fixture under the development seed namespace.  No test is a scientific result:
every fixture output is development_only and never citable.
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

from experiments.mlst2026.successor_20260913.common import matrix as M, protocol as PR, results as R  # noqa: E402
from experiments.mlst2026.successor_20260913.t21s import geometry as GEO, runner as T21RUN  # noqa: E402
from experiments.mlst2026.followup_20260917.r3_t21_finite import runner as RUN, scaling as SC  # noqa: E402

PROTO = REPO / "experiments/mlst2026/followup_20260917/r3_t21_finite/PROTOCOL_R3_T21_FINITE_v1.json"
PKG_DIR = PROTO.parent


def _main(*argv) -> int:
    return RUN.main(list(argv))


def _P():
    return PR.load_protocol(PROTO)


def _fixture_families(phase: str) -> list[str]:
    P = _P()
    return sorted({SC.family_name(phase, c) for c in SC.all_configurations(P, True)})


PILOT_FAMS = _fixture_families("pilot")
CONF_FAMS = _fixture_families("confirmation")


# ----------------------------------------------------------------------------- shared fixture run
@pytest.fixture(scope="module")
def full_fixture_root(tmp_path_factory) -> Path:
    """pilot -> reduce (freeze) -> confirmation -> reduce, once for the module."""
    root = tmp_path_factory.mktemp("r3_full") / "run"
    P = str(PROTO)
    assert _main("run", "--protocol", P, "--out", str(root), "--fixture", "--families", *PILOT_FAMS, "algebraic") == 0
    assert not (root / RUN.FROZEN_NAME).exists()
    assert _main("reduce", "--protocol", P, "--out", str(root), "--allow-partial") == 0
    assert (root / RUN.FROZEN_NAME).exists(), "the pilot reducer must write the frozen edge file"
    assert _main("run", "--protocol", P, "--out", str(root), "--fixture", "--families", *CONF_FAMS) == 0
    assert _main("reduce", "--protocol", P, "--out", str(root)) == 0
    return root


# ----------------------------------------------------------------------------- README tests 1: no compute
def test_validate_and_enumerate_compute_nothing(tmp_path):
    assert _main("validate", "--protocol", str(PROTO)) == 0
    out = tmp_path / "enum"
    assert _main("enumerate", "--protocol", str(PROTO), "--out", str(out)) == 0
    assert (out / "run_matrix.jsonl").exists() and not (out / "cells").exists()
    summ = json.loads((out / "run_matrix_summary.json").read_text())
    P = _P()
    assert summ["cells"] == 3460 and summ["fits"] == 3456 and summ["solves"] == 3458
    assert summ["fits"] == P["budget"]["planned_fits"] <= P["budget"]["estimator_fits_cap"]
    assert summ["by_role"]["pilot"]["cells"] == 1152 and summ["by_role"]["confirmation"]["cells"] == 2304
    assert summ["by_role"]["algebraic"]["cells"] == 4
    cells = [json.loads(l) for l in (out / "run_matrix.jsonl").read_text().splitlines()]
    for phase in ("pilot", "confirmation"):
        confs = {SC.config_key(c["params"]) for c in cells
                 if c["role"] != "algebraic" and c["params"]["phase"] == phase}
        assert len(confs) == 96, phase
    # 24 production families + algebraic, each family is one (geometry, arm, rung)
    assert len(summ["by_family"]) == 25


def test_help_never_computes():
    with pytest.raises(SystemExit) as e:
        _main("--help")
    assert e.value.code == 0


def test_run_refuses_archive_roots():
    bad = REPO / "results" / "r3_should_never_exist"
    assert _main("run", "--protocol", str(PROTO), "--out", str(bad), "--fixture") == 6
    assert not bad.exists()


# ----------------------------------------------------------------------------- fixture end-to-end
def test_fixture_end_to_end_has_declared_schema(full_fixture_root):
    red = json.loads((full_fixture_root / "REDUCED.json").read_text())
    assert red["development_only"] is True and red["partial"] is False and red["reconciliation"]["complete"]
    for k in ("phase_completeness", "controls", "algebraic", "configurations", "pilot", "confirmation",
              "frozen_edge", "historical_outcomes_preserved", "ladder_preservation", "finite_size_endpoint",
              "gated_interpretation", "candidate", "overlap_confound"):
        assert k in red, k
    assert all(red["phase_completeness"][p]["complete"] for p in ("pilot", "confirmation", "algebraic"))
    assert red["ladder_preservation"]["ok"] is True
    assert all(red["algebraic"].values())
    # old outcomes are restated verbatim, never edited
    old = red["historical_outcomes_preserved"]
    assert "296/298" in old and "NOT met" in old and "untested" in old
    cells = R.read_cells(full_fixture_root)
    assert len(cells) == 516
    for files in cells.values():
        _, js = files[0]
        assert js["development_only"] is True
        assert js["seed"] is None or 922000 <= js["seed"] <= 922999
        if js["role"] == "algebraic":
            assert set(RUN.ALGEBRAIC_RESULT_REQUIRED) <= set(js["result"]) and js["result"]["ok"] is True
        else:
            assert set(RUN.DATA_RESULT_REQUIRED) <= set(js["result"])
            assert js["fits_used"] == 1 and js["solves_used"] == 1
    conf = red["confirmation"]
    assert conf["complete"] and conf["all_configurations_complete"]
    assert conf["overall"]["n_rows"] == 64 * 4
    # uncertainty is across the 4 independent fixture seeds, never across directions
    assert {r["n_seeds"] for r in conf["rows"]} == {4}
    assert set(red["gated_interpretation"]) >= {"orthogonal_prerequisite_pass", "correlated_largest_pass", "loadings_step_status"}
    assert red["gated_interpretation"]["loadings_step_status"] in ("untested", "supported on synthetic Gaussian spiked data only", "not supported")


def test_fixture_never_citable_and_old_outcome_preserved(full_fixture_root):
    red = json.loads((full_fixture_root / "REDUCED.json").read_text())
    assert "development_only" in red["status_vocabulary"] and "never citable" in red["status_vocabulary"]
    # the five unmet predictions and the 296/298 outcome are stated, not repaired
    assert "five registered T21 predictions" in red["historical_outcomes_preserved"]


# ----------------------------------------------------------------------------- generator invariants
def test_generator_preserves_gram_and_spike_quantities_across_d():
    P = _P()
    pc = SC.preservation_check(P, False)
    assert pc["ok"], json.dumps({g: v["ok"] for g, v in pc["per_geometry"].items()})
    for geo in P["design"]["geometry"]["arms"]:
        per = pc["per_geometry"][geo]
        assert per["ok"] is True
        assert per["schur_unique_norm_ratio"] == pytest.approx([1 / x for x in np.diag(np.linalg.inv(GEO.gram(geo)))], abs=1e-12)
        for key, row in per["configs"].items():
            assert row["gram_max_abs_error"] <= GEO.GRAM_TOLERANCE, key
            assert row["population_spectrum_invariance_max_abs"] <= 1e-8, key
            assert row["c2_invariance_max_abs"] <= 1e-12, key
    # direct: the realised Gram is exactly the declared one at every rung
    for rung in SC.ladder(P, False):
        for geo in ("orthogonal", "correlated"):
            U = GEO.directions(rung["d"], geo)
            assert np.max(np.abs(U.T @ U - GEO.gram(geo))) <= 1e-10
    # the theta vector itself is invariant across rungs at fixed gamma
    for arm in ("hi", "lo"):
        base = SC.ladder(P, False)[0]
        p0 = {"geometry": "orthogonal", "gamma_arm": arm, "size_mult": 1, "d": base["d"],
              "n_train": base["n_train"][arm], "k": 4, "kind": "grid", "config_id": "below_edge",
              "varied_direction": 0, "ratio": 0.5}
        t0 = SC.theta_vector(P, p0)
        for rung in SC.ladder(P, False):
            p = {**p0, "size_mult": rung["mult"], "d": rung["d"], "n_train": rung["n_train"][arm]}
            assert SC.theta_vector(P, p) == pytest.approx(t0)


def test_legacy_classifier_matches_archived_module():
    """The R3 legacy classifier is the archived one, not a widened band."""
    P = _P()
    base = SC.ladder(P, False)[0]
    for geo in ("orthogonal", "correlated"):
        for arm in ("hi", "lo"):
            for cfg in SC.grid_configs(P) + [{"config_id": "all_zero", "varied_direction": None, "ratio": None},
                                              {"config_id": "all_strong", "varied_direction": None, "ratio": None}]:
                kind = cfg["config_id"] if cfg["config_id"] in ("all_zero", "all_strong") else "grid"
                params = {"geometry": geo, "gamma_arm": arm, "size_mult": base["mult"], "d": base["d"],
                          "n_train": base["n_train"][arm], "k": 4, "kind": kind, "config_id": cfg["config_id"],
                          "varied_direction": cfg["varied_direction"], "ratio": cfg["ratio"]}
                pred, _U, _Up, _t = T21RUN.compute_predictions(P, params)
                assert SC.legacy_edge_classes(P, pred) == list(T21RUN.edge_classes(P, pred, pred["sqrt_gamma"]))


def test_wide_band_is_not_the_legacy_band():
    P = _P()
    assert SC.legacy_band(P) == (0.7, 1.8)
    assert all(c["band"] != [0.7, 1.8] for c in SC.candidate_family(P))


# ----------------------------------------------------------------------------- namespaces
OLD_NAMESPACES = [
    (900000, 900999), (5000, 5099), (6000, 6299),           # t21s
    (920000, 920999), (21000, 21299), (22000, 22499),       # R1
    (921000, 921999), (23000, 23299), (24000, 24499),       # R2
]


def test_namespaces_disjoint_and_fixture_refuses_pilot_and_confirmation(tmp_path):
    P = _P()
    ns = P["seed_namespaces"]
    assert ns == {"development": [922000, 922999], "pilot": [25000, 25299], "confirmation": [26000, 26499]}
    mine = [tuple(ns[k]) for k in ("development", "pilot", "confirmation")]
    # pairwise disjoint within R3
    for i in range(len(mine)):
        for j in range(i + 1, len(mine)):
            a, b = mine[i], mine[j]
            assert a[1] < b[0] or b[1] < a[0]
    # disjoint from every earlier namespace
    for a in mine:
        for b in OLD_NAMESPACES:
            assert a[1] < b[0] or b[1] < a[0], (a, b)
    m = RUN.build_matrix(P, False)
    pilot = {c["seed"] for c in m.cells if c["role"] == "pilot"}
    conf = {c["seed"] for c in m.cells if c["role"] == "confirmation"}
    assert pilot and conf and not (pilot & conf)
    assert M.check_seed_roles(m, P) == []
    fx = RUN.build_matrix(P, True)
    dev = {c["seed"] for c in fx.cells if c["seed"] is not None}
    assert all(PR.namespace_of_seed(P, s) == "development" for s in dev)
    assert fx.matrix_hash() != m.matrix_hash()
    out = tmp_path / "fx"
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--roles", "pilot") == 6
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--roles", "confirmation") == 6
    assert not (out / "cells").exists()


# ----------------------------------------------------------------------------- freeze discipline
def test_confirmation_refused_without_frozen_edge(tmp_path):
    out = tmp_path / "noconf"
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--families", *CONF_FAMS) == 6
    assert not (out / "cells").exists() and not (out / "EXECUTION_LEDGER.jsonl").exists(), "nothing may be charged"
    # a pilot run alone does not unlock confirmation: the pilot must be reduced first
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--families", *PILOT_FAMS, "algebraic") == 0
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--families", *CONF_FAMS) == 6
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(out), "--allow-partial") == 0
    fz = out / RUN.FROZEN_NAME
    assert fz.exists()
    content = json.loads(fz.read_text())
    assert content["legacy"]["formula_id"] == T21RUN.PRED.FORMULA_ID
    assert content["legacy"]["coefficients_fitted_on_any_data"] is False
    assert content["legacy"]["tolerance"]["primary_abs"] == 0.05
    assert content["confirmation_permitted"] is True and content["development_only"] is True
    assert content["n_confirmation_cells"] == 256


def test_reduce_refuses_partial_and_tampered_cells(tmp_path, full_fixture_root):
    # partial run refused without --allow-partial
    out = tmp_path / "part"
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture",
                 "--families", PILOT_FAMS[0], "algebraic") == 0
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(out)) == 3
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(out), "--allow-partial") == 0
    # tampered cell bytes (digest differs from the ledger)
    root = tmp_path / "copy"
    shutil.copytree(full_fixture_root, root)
    src = sorted((root / "cells").glob("*.json"))[0]
    js = json.loads(src.read_text())
    if js["role"] == "algebraic":
        js["result"]["ok"] = False
    else:
        js["result"]["recovery"][0] = 0.5
    src.write_text(json.dumps(js))
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(root)) == 3
    # confirmation cells present but the frozen file removed -> refused at reduce (exit 4)
    root2 = tmp_path / "copy2"
    shutil.copytree(full_fixture_root, root2)
    (root2 / RUN.FROZEN_NAME).unlink()
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(root2)) == 4
    # tampered frozen legacy classification -> refused at reduce (exit 4)
    root3 = tmp_path / "copy3"
    shutil.copytree(full_fixture_root, root3)
    fz = json.loads((root3 / RUN.FROZEN_NAME).read_text())
    key = next(iter(fz["configurations"]))
    fz["configurations"][key]["legacy_edge_class"] = ["near_edge"] * len(fz["configurations"][key]["legacy_edge_class"])
    (root3 / RUN.FROZEN_NAME).write_text(json.dumps(fz))
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(root3)) == 4


# ----------------------------------------------------------------------------- scientific mutation
def test_mutation_widened_band_on_legacy_classification_is_refused(tmp_path, monkeypatch):
    """A runner that silently applies a widened near-edge band to the OLD
    classification writes internally consistent files with valid digests; the
    reducer's independent recomputation from the protocol band rejects them."""
    P = _P()
    real = SC.compute_predictions

    def widened(P_, params):
        pred = real(P_, params)
        pred["edge_class"] = SC.edge_classes_band(pred["loadings_sq"], pred["population_excess_eigenvalues"],
                                                  pred["sqrt_gamma"], (0.4, 1.8), SC.legacy_loading_min(P_))
        return pred

    fx = RUN.build_matrix(P, True)
    targets = [c["cell_id"] for c in fx.cells
               if c["role"] == "development" and c["params"]["phase"] == "pilot"
               and c["params"]["geometry"] == "orthogonal"
               and c["params"]["config_id"] in ("below_edge", "old_failure_subthreshold")][:4]
    assert targets
    monkeypatch.setattr(SC, "compute_predictions", widened)
    monkeypatch.setattr(RUN, "check_data_result", lambda *a, **k: None)   # the corrupted runner does not self-check
    out = tmp_path / "mut_band"
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--cells", *targets) == 0
    monkeypatch.undo()
    # the cell files exist with valid digests ...
    assert len(list((out / "cells").glob("*.json"))) == len(targets)
    # ... and the reducer still refuses, from its independent protocol-band recomputation
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(out), "--allow-partial") == 4


def test_mutation_signal_in_all_zero_control_is_refused(tmp_path, monkeypatch):
    """A generator that leaks signal into the all-zero cells is caught by the
    all-zero control at reduce, independently of the digest."""
    P = _P()
    fx = RUN.build_matrix(P, True)
    az = [c["cell_id"] for c in fx.cells
          if c["role"] == "development" and c["params"]["kind"] == "all_zero" and c["params"]["phase"] == "pilot"]
    real_draw = T21RUN.draw_data

    def leaky(seed, n, U, theta):
        return real_draw(seed, n, U, np.full_like(theta, 30.0))

    monkeypatch.setattr(T21RUN, "draw_data", leaky)
    out = tmp_path / "leak"
    assert _main("run", "--protocol", str(PROTO), "--out", str(out), "--fixture", "--cells", *az[:4]) == 0
    monkeypatch.undo()
    assert _main("reduce", "--protocol", str(PROTO), "--out", str(out), "--allow-partial") == 4


# ----------------------------------------------------------------------------- gated interpretation
def test_gated_interpretation_cannot_support_loadings_when_prerequisite_fails():
    P = _P()
    endpoint_bad = {"orthogonal_prerequisite_pass": False, "correlated_largest_pass": True, "endpoint_met": False}
    gi = SC.gated_interpretation(P, endpoint_bad, {"selected": {"id": "none"}})
    assert gi["loadings_step_status"] == "untested"
    assert "isolated-spike overlap" in gi["reason"]
    endpoint_ok = {"orthogonal_prerequisite_pass": True, "correlated_largest_pass": True, "endpoint_met": True}
    assert SC.gated_interpretation(P, endpoint_ok, {"selected": {"id": "none"}})["loadings_step_status"] == "supported on synthetic Gaussian spiked data only"
    endpoint_corr_fail = {"orthogonal_prerequisite_pass": True, "correlated_largest_pass": False, "endpoint_met": False}
    assert SC.gated_interpretation(P, endpoint_corr_fail, {"selected": {"id": "none"}})["loadings_step_status"] == "not supported"


def test_fixture_gated_interpretation_is_consistent(full_fixture_root):
    red = json.loads((full_fixture_root / "REDUCED.json").read_text())
    ep = red["finite_size_endpoint"]
    gi = red["gated_interpretation"]
    assert gi["orthogonal_prerequisite_pass"] == ep["orthogonal_prerequisite_pass"]
    # a correlated pass can never be read as support while the prerequisite fails
    if not ep["orthogonal_prerequisite_pass"]:
        assert gi["loadings_step_status"] == "untested"
    # development-only output cannot support the loadings step under any outcome
    assert "never citable" in gi["development_only_caveat"]


# ----------------------------------------------------------------------------- candidate selection
def test_candidate_selection_uses_pilot_only_and_declares_none(full_fixture_root):
    P = _P()
    cells = R.read_cells(full_fixture_root)
    pilot = [js for files in cells.values() for _p, js in files
             if js["role"] != "algebraic" and js["params"]["phase"] == "pilot"]
    sel = SC.select_edge_candidate(P, pilot)
    assert sel["selected"]["id"] in [c["id"] for c in SC.candidate_family(P)] + ["none"]
    assert {e["id"] for e in sel["evidence"].values()} == {c["id"] for c in SC.candidate_family(P)}
    for e in sel["evidence"].values():
        assert e["n_primary_pass"] <= e["n_primary_scored"]
    # the frozen file records the same selection
    fz = json.loads((full_fixture_root / RUN.FROZEN_NAME).read_text())
    assert fz["selected_candidate"] == sel["selected"]


def test_candidate_selection_rule_is_declared_and_predeclared():
    P = _P()
    rule = P["design"]["candidate_selection_rule"]
    assert "smallest band width" in rule["select"]
    assert rule["if_none"].startswith("freeze 'none'")
    assert "never edits the legacy classification" in rule["purpose"]


# ----------------------------------------------------------------------------- finite-size endpoint mechanics
def test_finite_size_endpoint_declines_only_when_both_rungs_improve():
    P = _P()
    ids = P["design"]["finite_size_endpoint"]["decline_pair_config_ids"]
    cfg = SC.config_id_lookup(P)

    def row(cid, mult, err):
        c = cfg[cid]
        return {"config": f"orthogonal|hi|d{mult}|grid", "geometry": "orthogonal", "gamma_arm": "hi", "kind": "grid",
                "config_id": cid, "size_mult": mult, "d": {1: 256, 2: 512, 4: 1024}[mult],
                "varied_direction": c["varied_direction"], "ratio": c["ratio"], "direction": c["varied_direction"],
                "edge_class": "primary", "n_seeds": 24, "expected_n_seeds": 24, "predicted": 0.0, "observed_mean": err,
                "observed_se": 0.01, "observed_ci95": [-0.02, 0.02], "abs_error_of_mean": err, "tolerance_abs": 0.05,
                "pass": err <= 0.05, "ci95_covers_prediction": False}

    rows = []
    for cid in ids:
        rows += [row(cid, 1, 0.07), row(cid, 2, 0.06), row(cid, 4, 0.04)]   # declines and within tolerance
    ep = SC.finite_size_endpoint(P, rows)
    assert all(v["ok"] for v in ep["decline"].values())
    assert ep["decline"][ids[0]]["base_abs_error"] == pytest.approx(0.07)
    assert ep["decline"][ids[0]]["largest_abs_error"] == pytest.approx(0.04)
    # one pair rises at the largest rung -> endpoint not met
    rows2 = []
    for cid in ids:
        vals = [0.07, 0.06, 0.04] if cid == ids[0] else [0.02, 0.03, 0.08]
        rows2 += [row(cid, m, e) for m, e in zip((1, 2, 4), vals)]
    ep2 = SC.finite_size_endpoint(P, rows2)
    assert ep2["endpoint_met"] is False
    assert ep2["decline"][ids[1]]["decline_ok"] is False


# ----------------------------------------------------------------------------- resource estimate
def test_estimate_from_fixture_timings(full_fixture_root):
    assert _main("estimate", "--protocol", str(PROTO), "--out", str(full_fixture_root)) == 0
    est = json.loads((full_fixture_root / "RESOURCE_ESTIMATE.json").read_text())
    assert est["unmeasured_families"] == []
    assert est["measured_projection_available"] is True
    assert est["cpu_seconds"]["cell_compute_point"] > 0
    assert est["counts"]["fits"] == 3456
    assert est["priced_counts"]["fits"] == 3456
    assert set(est["families"]) >= {"pilot-orthogonal-hi-d256", "confirmation-correlated-lo-d1024", "algebraic"}
    assert "wall_time_hours_by_workers" in est and est["wall_time_hours_by_workers"]["8"]["hours_point"] > 0
    # the flat fixture-scale cross-check exists but is not the estimate
    assert "cross_check_flat_fixture_scale" in est


# ----------------------------------------------------------------------------- protocol / provenance
def test_protocol_records_supersede_provenance_and_preserved_outcome():
    P = _P()
    assert P["schema"] == PR.SCHEMA and P["package"] == "r3_t21_finite"
    assert P["supersedes"]["package"] == "t21s" and "PROTOCOL_T21S_v2.json" in P["supersedes"]["protocol_file"]
    prov = P["provenance"]
    assert prov["designed_after"] == "2026-09-17"
    assert "296/298" in prov["old_outcome_preserved_verbatim"] and "NOT met" in prov["old_outcome_preserved_verbatim"]
    assert "orthogonal|hi|N128" in json.dumps(prov["old_primary_failures"])
    assert "NOT" in P["provenance"]["targeted_not_full_grid"].upper() or "not be read" in P["provenance"]["targeted_not_full_grid"]
    assert "fixed allocation" in P["stopping_rule"] and "no early stop" in P["stopping_rule"]
    for k in ("estimator_fits_cap", "retry_policy"):
        assert k in P["budget"]
    assert P["design"]["base_d"] == 256 and P["design"]["size_ladder"] == [1, 2, 4]
    assert P["design"]["gamma_arms"] == {"hi": 2.0, "lo": 0.25}
    for name in ("README.md", "REMOTE_COMMANDS.md", "BENCHMARK.json", "requirements.txt"):
        assert (PKG_DIR / name).exists(), name
    rc = (PKG_DIR / "REMOTE_COMMANDS.md").read_text()
    assert "<RUN_ROOT>" in rc and "<WORKERS>" in rc and "<PROTOCOL>" in rc


def test_matrix_catalogue_matches_design_and_ladder():
    P = _P()
    matrix = P["matrix"]
    assert matrix["geometries"] == P["design"]["geometry"]["arms"]
    assert matrix["gamma_arms"] == P["design"]["gamma_arms"]
    assert matrix["scaling_ladder"] == [{"mult": s["mult"], "d": s["d"], "n_train": s["n_train"]} for s in P["design"]["sizes"]]
    assert [(c["config_id"], c["varied_direction"], c["ratio"]) for c in matrix["targeted_configurations"]] == [
        (c["config_id"], c["varied_direction"], c["ratio"]) for c in P["design"]["grid"]["configurations"]]
    assert matrix["planned_fits"] == P["budget"]["planned_fits"]
    assert matrix["planned_solves"] == P["budget"]["planned_solves"]
    assert matrix["configurations_per_phase"] == 96


def test_overlap_confound_is_reported_separately(full_fixture_root):
    red = json.loads((full_fixture_root / "REDUCED.json").read_text())
    oc = red["overlap_confound"]
    assert "kept separate from the finite-size endpoint" in oc["note"]
    assert oc["correlated_grid"]
    for ent in oc["correlated_grid"].values():
        assert ent["chance_level"] > 0
        assert any(d["unique_defined"] for d in ent["directions"])
        for d in ent["directions"]:
            assert d["ordinary_mean"] is not None and d["unique_mean"] is not None


def test_algebraic_ladder_preservation_and_legacy_rule_cells(full_fixture_root):
    cells = R.read_cells(full_fixture_root)
    alg = [js["result"] for files in cells.values() for _p, js in files if js["role"] == "algebraic"]
    by_check = {v["check"]: v for v in alg}
    assert set(by_check) == {"ladder_preservation", "predictor_sanity", "legacy_classification_rule"}
    assert all(v["ok"] for v in alg)
    pres = [v for v in alg if v["check"] == "ladder_preservation"]
    assert len(pres) == 2 and all(v["details"]["ok"] for v in pres)
    rule = by_check["legacy_classification_rule"]
    assert rule["details"]["matches_archived"]["ok"] is True


# --- A3: executable ladder-preservation receipt at EVERY rung ------------------------
def test_ladder_preservation_receipt_at_every_rung():
    from experiments.mlst2026.followup_20260917.r3_t21_finite import scaling as SC2
    P = PR.load_protocol(PROTO)
    pc = SC2.preservation_check(P, fixture=True)
    assert pc["ok"] is True
    rungs = SC2.ladder(P, fixture=True)
    receipts = pc["rung_receipts"]
    assert len(receipts) == len(rungs)
    assert {r["mult"] for r in receipts} == {r["mult"] for r in rungs}
    for r in receipts:
        assert r["ok"] is True and r["n"] == r["n_ok"]
        assert {"gram_ok", "spectrum_ok", "unique_norm_ratio_ok", "c2_ok"} <= set(r)


def test_preservation_receipt_subcommand_writes_a_pass_fail_receipt(tmp_path):
    import json
    out = tmp_path / "receipt.json"
    rc = RUN.main(["preservation-receipt", "--protocol", str(PROTO), "--out", str(out)])
    assert rc == 0
    rec = json.loads(out.read_text())
    assert rec["ok"] is True and rec["fixture"] is True
    assert rec["rung_receipts"] and all(r["ok"] for r in rec["rung_receipts"])


def test_committed_production_ladder_receipt_has_all_three_rungs():
    import json
    rec = json.loads((REPO / "experiments/mlst2026/followup_20260917/r3_t21_finite/LADDER_PRESERVATION_RECEIPT.json").read_text())
    assert rec["fixture"] is False and rec["ok"] is True
    assert [r["d"] for r in rec["rung_receipts"]] == [256, 512, 1024]
    for r in rec["rung_receipts"]:
        assert r["ok"] is True and r["n"] == r["n_ok"] and r["n"] > 0
