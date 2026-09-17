"""Round-2 correction regression tests for the follow-up packages.

Each test reproduces one finding from
``output/implementation-review-20260917/REVIEW.md`` (F1--F10) and fails if the
corresponding fix regresses.  No production simulation is launched; every test
uses bounded fixtures or synthetic stored records.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.mlst2026.followup_20260917.r0_t22_h5 import reducer as r0  # noqa: E402
from experiments.mlst2026.followup_20260917.r1_t22_calibration import candidates as C  # noqa: E402
from experiments.mlst2026.followup_20260917.r1_t22_calibration import runner as r1  # noqa: E402
from experiments.mlst2026.followup_20260917.r4_t22_feasibility import r1_gate as gate  # noqa: E402
from experiments.mlst2026.followup_20260917.r4_t22_feasibility import runner as r4  # noqa: E402
from experiments.mlst2026.followup_20260917.r4_t22_feasibility import streams as st  # noqa: E402


# --------------------------------------------------------------------------- F1
def _synthetic_root(tmp_path: Path) -> Path:
    root = tmp_path / "src_root"
    (root / "cells").mkdir(parents=True)
    (root / "REDUCED.json").write_text("{}")
    (root / "protocol_snapshot.json").write_text("{}")
    (root / "marker").write_text("synthetic only")
    return root


def _tree_hashes(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("rel", ["", ".", "sub", "sub/deeper"])
def test_f1_r0_refuses_same_and_overlapping_destinations_without_touching_source(tmp_path, rel):
    root = _synthetic_root(tmp_path)
    before = _tree_hashes(root)
    target = root if rel == "" else root / rel
    with pytest.raises(ValueError):
        r0.reduce_run(root, tmp_path / "out", copy_to=target, allow_in_place=False)
    assert _tree_hashes(root) == before


def test_f1_r0_refuses_symlink_alias_of_the_source(tmp_path):
    root = _synthetic_root(tmp_path)
    link = tmp_path / "alias"
    link.symlink_to(root, target_is_directory=True)
    before = _tree_hashes(root)
    with pytest.raises(ValueError):
        r0.reduce_run(root, tmp_path / "out", copy_to=link)
    assert _tree_hashes(root) == before


def test_f1_r0_refuses_an_existing_destination_and_never_deletes(tmp_path):
    root = _synthetic_root(tmp_path)
    dest = tmp_path / "existing"
    dest.mkdir()
    (dest / "keepme").write_text("unrelated")
    before = _tree_hashes(root)
    with pytest.raises(ValueError):
        r0.reduce_run(root, tmp_path / "out", copy_to=dest)
    assert (dest / "keepme").exists()          # nothing was deleted
    assert _tree_hashes(root) == before


def test_f1_r0_refuses_an_output_inside_the_input_tree(tmp_path):
    root = _synthetic_root(tmp_path)
    with pytest.raises(ValueError):
        r0.reduce_run(root, root / "export", copy_to=tmp_path / "copydest")


# --------------------------------------------------------------------------- F10
def _rec(*, mu, stored_covers, ci_lo, ci_hi, t_stored, t_lo, t_hi, re_lo, re_hi, point, withheld=False,
         contrast="raw-bin|empca"):
    def c_for(mu_c):
        return {"true_mu": mu_c, "covers": stored_covers, "t_covers": t_stored,
                "ci95_lo": ci_lo, "ci95_hi": ci_hi, "t_lo": t_lo, "t_hi": t_hi,
                "point": point, "bootstrap_sd": 0.1, "length": ci_hi - ci_lo, "t_length": t_hi - t_lo,
                "re_estimate": point, "re_lo": re_lo, "re_hi": re_hi, "per_file": {}}
    contrasts = {k: c_for(mu) for k in r0.ALL_CONTRASTS}
    contrasts[contrast] = c_for(mu)
    return {"withheld": withheld, "scenario": "synthetic", "contrasts": contrasts}


def test_f10_r0_recomputes_coverage_from_bounds_not_stored_flags():
    # stored flag says covered; the bounds say it is not -> the bound wins
    recs = [_rec(mu=0.10, stored_covers=True, ci_lo=0.20, ci_hi=0.30,
                 t_stored=True, t_lo=0.20, t_hi=0.30, re_lo=0.20, re_hi=0.30, point=0.25)] * 5
    s = r0.summarise_records("synthetic", recs, {"drift_sd": 0.0})["contrasts"][r0.PRIMARY_CONTRAST]
    assert s["coverage_unconditional"]["two_stage"]["k"] == 0
    assert s["coverage_unconditional"]["t_interval"]["k"] == 0


def test_f10_r0_target_statistic_is_not_an_established_mismatch():
    recs = []
    for _ in range(20):
        r = _rec(mu=0.10, stored_covers=True, ci_lo=0.09, ci_hi=0.11, t_stored=True, t_lo=0.09, t_hi=0.11,
                 re_lo=0.095, re_hi=0.115, point=0.10)
        r["contrasts"]["raw-bin|empca"]["re_estimate"] = 0.108
        recs.append(r)
    t = r0.summarise_records("synthetic", recs, {"drift_sd": 0.0})["contrasts"][r0.PRIMARY_CONTRAST]["target"]
    assert t["displacement_flag"] is True
    assert t["establishes_population_mismatch"] is False
    assert t["establishes_cause_of_undercoverage"] is False
    assert "exploratory" in t["statistic"]
    assert set(t["population_targets"]) == {"two_stage_equal_file", "random_effects"}


def test_f10_r0_inventory_detects_duplicate_cell_ids(tmp_path):
    root = _synthetic_root(tmp_path)
    cell = {"cell_id": "dup", "family": "sim_confirm", "role": "confirmation", "seed": 1,
            "params": {"scenario": {"drift_sd": 0.0, "name": "s"}},
            "result": {"summary": {"scenario_name": "s"}, "records": [{"withheld": False}]}}
    for name in ("a.json", "b.json"):
        (root / "cells" / name).write_text(json.dumps(cell))
    _, _, _, _, _, validation = r0.load_stored(root)
    assert validation["ok"] is False
    assert any("duplicate cell_id" in p for p in validation["problems"])
    assert validation["reducer_source_sha256"]


# --------------------------------------------------------------------------- F2
def test_f2_archived_baseline_endpoints_and_p_are_carried_exactly():
    pf = C.PerFile(y=np.array([1.0, 1.1, 0.9, 1.05]), s=np.array([0.2, 0.3, 0.25, 0.22]), n_bootstrap=40)
    ivs = C.candidate_intervals(pf, 1.0, 0.1, two_stage_lo=0.65, two_stage_hi=1.15, two_stage_p=0.03)
    base = ivs["two_stage"]
    assert (base["lo"], base["hi"]) == (0.65, 1.15)       # asymmetric stored endpoints, exactly
    assert base["p_coherent"] == 0.03                     # stored bootstrap p, not a normal tail
    assert C.coherent_pvalue("two_stage", base) == 0.03
    # the normal approximation is a DIFFERENT, separately named candidate
    nz = ivs["normal_z"]
    assert (nz["lo"], nz["hi"]) == pytest.approx((1.0 - C.Z975 * 0.1, 1.0 + C.Z975 * 0.1))
    assert nz["form"].startswith("point +/- z*SE")


# --------------------------------------------------------------------------- F3
def test_f3_withheld_replicates_are_never_supported():
    keys = {k for ks in r1.NULL_CLAIM_KEYS.values() for k in ks}
    rec = {"withheld": True,
           "candidates": {k: {"two_stage": {"point": 1.0, "lo": 0.5, "hi": 1.5, "p_coherent": 1e-10}}
                          for k in keys}}
    decisions = r1._claim_decisions(rec, "two_stage")
    assert all(v is False for v in decisions.values())


def test_f3_support_never_exceeds_one_minus_withholding():
    keys = {k for ks in r1.NULL_CLAIM_KEYS.values() for k in ks}
    good = {"candidates": {k: {"two_stage": {"point": 1.0, "lo": 0.5, "hi": 1.5, "p_coherent": 1e-10}}
                           for k in keys}}
    records = [{**good, "withheld": i < 8} for i in range(10)]   # 80% withheld
    supported = sum(any(r1._claim_decisions(r, "two_stage").values()) for r in records)
    withheld = sum(r["withheld"] for r in records)
    assert supported <= len(records) - withheld
    # and an empty conditional sample is handled explicitly, not as a pass
    empty = {"withheld": True, "candidates": {k: {"two_stage": {"point": 0.0, "lo": 0.0, "hi": 0.0,
                                                                 "p_coherent": 1.0}} for k in keys}}
    assert all(v is False for v in r1._claim_decisions(empty, "two_stage").values())


# --------------------------------------------------------------------------- F4
def test_f4_primary_eligibility_excludes_target_mismatched_candidates():
    assert "welch_fe_t" not in C.PRIMARY_ELIGIBLE and "re_t" not in C.PRIMARY_ELIGIBLE
    assert set(C.PRIMARY_ELIGIBLE) | set(C.SECONDARY_DIAGNOSTIC) == set(C.CANDIDATES)
    doc = C.selection_rule_doc()
    assert doc["primary_eligible"] == list(C.PRIMARY_ELIGIBLE)


def test_f4_welch_dof_is_the_file_count_not_the_bootstrap_count():
    y = np.array([1.0, 1.1, 0.9, 1.05])
    a = C.candidate_intervals(C.PerFile(y=y, s=np.full(4, 0.1), n_bootstrap=40), 1.0, 0.1)
    b = C.candidate_intervals(C.PerFile(y=y, s=np.full(4, 0.1), n_bootstrap=4000), 1.0, 0.1)
    assert a["welch_fe_t"]["dof"] == b["welch_fe_t"]["dof"] == 3.0
    assert a["welch_fe_t"]["dof"] != 4000 - 1


def test_f4_unified_type1_predicate_rejects_5_of_100():
    w = r1.wilson(5, 100)
    assert C.type1_accepts_upper_bound(w, 0.07) is False        # upper bound 0.11175 > 0.07
    assert C.historical_type1_overlap(w, 0.07) is True          # the archived overlap rule accepts


# --------------------------------------------------------------------------- F5
def _freeze(path):
    path.write_text(json.dumps({
        "schema": "mlst-r1-frozen-candidate/1", "chosen_candidate": "two_stage_t",
        "development_only": False, "protocol_sha256": "p", "matrix_sha256": "m",
        "candidate_family": list(C.CANDIDATES), "primary_eligible": list(C.PRIMARY_ELIGIBLE),
        "secondary_diagnostic": list(C.SECONDARY_DIAGNOSTIC), "adequacy_rule": C.selection_rule_doc(),
        "frozen_from_pilot_cells": ["a"]}))


def test_f5_empty_acceptance_is_blocked_not_passed(tmp_path):
    fz = tmp_path / "fz.json"
    _freeze(fz)
    ac = tmp_path / "ac.json"
    ac.write_text("{}")
    g = gate.evaluate_r1_precondition(fz, ac)
    assert g["status"] == "blocked" and g["cancel"] is False and gate.ready_to_transfer(g) is False


def test_f5_unresolved_gate_removes_large_file_cells(tmp_path, monkeypatch):
    P = r4.PROTOCOL_DEFAULT and __import__(
        "experiments.mlst2026.successor_20260913.common.protocol",
        fromlist=["load_protocol"]).load_protocol(r4.PROTOCOL_DEFAULT)
    monkeypatch.setattr(r4, "evaluate_gate",
                        lambda P_, **kw: {"status": "unresolved", "cancel": False, "blocked": False,
                                          "cancel_large_file_counts": [32, 64], "reasons": [], "r1": {}})
    scs = r4.effective_eval_scenarios(P)
    assert all(s.n_files not in (32, 64) for s in scs)


def test_f5_run_refuses_when_the_gate_is_unresolved(tmp_path):
    argv = ["run", "--protocol", str(r4.PROTOCOL_DEFAULT), "--out", str(tmp_path / "o")]
    assert r4._precheck_gate(argv) is not None


# --------------------------------------------------------------------------- F6
def test_f6_r4_applies_the_frozen_r1_candidate():
    sc = r4._sc("alt-medium", 4, 0.0)
    inf = {"candidate": "two_stage_t", "version": "r1-t22-calibration-v1", "applies_r1_candidate": True,
           "r1_receipt": "x.json", "r1_receipt_sha256": "abc"}
    rec = st.run_replicate_streams(4242, sc, file_replicates=20, hier_replicates=50, curve_replicates=10,
                                   inference=inf)
    assert rec["inference"]["candidate"] == "two_stage_t"
    assert rec["inference"]["applies_r1_candidate"] is True
    assert rec["inference"]["r1_receipt_sha256"] == "abc"
    for key, c in rec["contrasts"].items():
        assert c["selected_candidate"] == "two_stage_t"
        assert "selected_lo" in c and "selected_hi" in c and "selected_p_coherent" in c
    # the archived baseline is retained separately and identifiable
    assert rec["baseline_contrasts"] and rec["baseline_verdicts"]


# --------------------------------------------------------------------------- F7
def test_f7_development_design_does_not_change_the_evaluation_realization():
    sc = r4._sc("alt-medium", 8, 0.6, n_dev=4, n_cal=r4.CAL_EVAL)
    dev64, ev64, info64 = st.simulate_scenario_streams(999, sc, n_cal_dev=64)
    dev256, ev256, info256 = st.simulate_scenario_streams(999, sc, n_cal_dev=256)
    # development calibration length changes; development arrays differ
    assert len(dev64[0].cal_read) != len(dev256[0].cal_read)
    # evaluation arrays and the per-file noise scales are IDENTICAL
    for a, b in zip(ev64, ev256):
        assert a.N == b.N and np.array_equal(a.cal_read, b.cal_read)
        assert np.array_equal(a.eval_read, b.eval_read)
    assert [f.N for f in ev64] == [f.N for f in ev256]
    assert info64["n_cal_eval"] == info256["n_cal_eval"] == r4.CAL_EVAL


def test_f7_development_design_table_preserves_scenario_identity():
    pairs = r4.dev_design_pairs()
    names = [s.name for s, _ in pairs]
    assert len(names) == len(set(names))
    assert any(n.startswith("alt-medium") for n in names) and any(n.startswith("null-partial") for n in names)
    # a key that omitted the effect would collide: both effects must appear at the
    # same (n_dev_files, n_cal_dev) pair
    same_design = [n for n in names if "ndev4-" in n and "ncaldev64" in n]
    assert len(same_design) == 2


def test_f7_feasibility_row_includes_any_null_family_and_completeness():
    row = {"is_drift": False, "true_mu": {"raw-bin|empca": 0.0001171875},
           "claims": {"denoising_benefit": {"is_null": False, "supported": {"rate": 0.9},
                                            "supported_given_not_withheld": {"rate": 0.95}}},
           "coverage_primary": {"raw-bin|empca": {"within_tolerance": True}},
           "withholding": {"rate": 0.2},
           "any_null_claim_supported": {"within_tolerance": False},
           "support_bound": {"plausible_max_unconditional_support": 0.9, "screen_power_0_5_possible": True}}
    rules = {"max_withholding_rate": 0.5, "min_power_medium_effect": 0.5}
    fr = r4.an.feasibility_row([row], rules, {"coverage": [0.925, 0.975]},
                               expected_scenarios=2, complete=True)
    assert fr["any_null_family_within_tolerance"] is False
    assert fr["feasible_under_rules"] is False          # familywise error blocks feasibility
    fr2 = r4.an.feasibility_row([row], rules, {"coverage": [0.925, 0.975]},
                                expected_scenarios=2, complete=False)
    assert fr2["complete"] is False and fr2["feasible_under_rules"] is False


# --------------------------------------------------------------------------- F9
def test_f9_r1_freeze_must_bind_pilot_content(tmp_path):
    P = __import__("experiments.mlst2026.successor_20260913.common.protocol",
                   fromlist=["load_protocol"]).load_protocol(r1.PROTOCOL_DEFAULT)
    fz = tmp_path / r1.FROZEN_NAME
    # a freeze missing the pilot-cell binding and the eligibility split is refused
    fz.write_text(json.dumps({"schema": r1.FROZEN_SCHEMA, "protocol_sha256": P["_sha256"],
                              "development_only": False}))
    with pytest.raises(RuntimeError):
        r1._read_frozen(tmp_path, P, fixture=False)


def test_f9_r1_acceptance_requires_a_complete_confirmation(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "REDUCED.json").write_text(json.dumps({
        "partial": False, "confirmation": {"complete": False, "acceptance": {"frozen_candidate": "none"}}}))
    with pytest.raises(SystemExit):
        r1.acceptance_command(["--run-root", str(root)])


def test_f9_r2_confirmation_precheck_checks_the_pilot_digest(tmp_path):
    import importlib
    r2 = importlib.import_module("experiments.mlst2026.followup_20260917.r2_t20_theta012.runner")
    PR = importlib.import_module("experiments.mlst2026.successor_20260913.common.protocol")
    P = PR.load_protocol(r2.PROTOCOL_DEFAULT)
    m = r2.build_matrix(P, False)
    out = tmp_path / "r2root"
    out.mkdir()
    (out / r2.FROZEN_NAME).write_text(json.dumps({
        "schema": r2.FROZEN_SCHEMA, "package": r2.PACKAGE, "protocol_sha256": P["_sha256"],
        "matrix_sha256": m.matrix_hash(), "pilot_digest": "not-the-pilot"}))
    rc = r2._confirmation_precheck(["run", "--protocol", str(r2.PROTOCOL_DEFAULT), "--out", str(out)])
    assert rc == 6          # refused before any confirmation draw, because no/mismatched pilot cells


# --------------------------------------------------------------------------- R0-v2 export
_EXPORT_V2 = REPO / "experiments/mlst2026/followup_20260917/r0_t22_h5/evidence_export_v2"


@pytest.mark.skipif(not (_EXPORT_V2 / "PROVENANCE.json").exists(),
                    reason="R0-v2 export not built in this checkout")
def test_r0_v2_export_is_validated_and_labels_displacement_exploratory():
    prov = json.loads((_EXPORT_V2 / "PROVENANCE.json").read_text())
    assert prov["amendment_version"] == "R0-v2"
    assert prov["reduced_json_sha256"] == "55c96c4b40e4631c5da16650394f92a1516a3ce95f82fe5629498840424d916c"
    assert prov["input_validation"]["ok"] is True
    assert prov["input_validation"]["n_confirmation_cells"] > 0
    assert prov["reducer_source_sha256"]
    rep = json.loads((_EXPORT_V2 / "H5_RANDOM_EFFECTS_REDUCTION.json").read_text())
    nd = rep["scope_tables"]["non_drift_65"]
    assert nd["random_effects_in_band"] == 24 and abs(nd["random_effects_coverage_median"] - 0.876) <= 0.001
    assert rep["consistency_with_frozen_reduction"]["scenarios_differing"] == 0
    any_scenario = next(iter(rep["scenarios"].values()))
    t = any_scenario["contrasts"]["raw-bin|empca"]["target"]
    assert t["establishes_population_mismatch"] is False
    assert t["establishes_cause_of_undercoverage"] is False
    assert "exploratory" in t["statistic"]


# --------------------------------------------------------------------------- R4 dev-gate screen wiring
def test_f5_withholding_screen_is_wired_and_malformed_is_refused(tmp_path, monkeypatch):
    import json as _json
    screen = tmp_path / "screen.json"
    screen.write_text(_json.dumps({"4": False, "8": False, "16": False}))
    monkeypatch.setattr(r4, "r1_gate_config", lambda P: {"withholding_screen_path": str(screen)})
    assert r4._withholding_screen({}) == {"4": False, "8": False, "16": False}
    screen.write_text(_json.dumps({"4": "no"}))                      # malformed
    from experiments.mlst2026.successor_20260913.common.protocol import ProtocolError
    with pytest.raises(ProtocolError):
        r4._withholding_screen({})
    # a universal-infeasible screen cancels the large-file extension once the
    # precondition otherwise passes (the passing-freeze case is covered in the R4 suite)
    assert gate.evaluate_r1_precondition(
        "/nope", None, r4_withholding_screen={"32": True, "64": True})["status"] == "unresolved"
