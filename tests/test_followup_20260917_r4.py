"""R4 (T22 feasibility) local acceptance tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.mlst2026.successor_20260913.common import protocol as PR  # noqa: E402
from experiments.mlst2026.followup_20260917.r4_t22_feasibility import analysis as an  # noqa: E402
from experiments.mlst2026.followup_20260917.r4_t22_feasibility import runner as RN  # noqa: E402
from experiments.mlst2026.followup_20260917.r4_t22_feasibility import streams as st  # noqa: E402

PROTO = RN.PROTOCOL_DEFAULT


def main(*argv):
    return RN.main([argv[0], "--protocol", str(PROTO)] + list(argv[1:]))


def test_validate_and_enumerate_compute_nothing(tmp_path):
    assert main("validate") == 0
    out = tmp_path / "enum"
    assert main("enumerate", "--out", str(out)) == 0
    assert (out / "run_matrix.jsonl").exists() and not (out / "cells").exists()
    s = json.loads((out / "run_matrix_summary.json").read_text())
    # with no R1 acceptance receipt the R1 gate is unresolved and the large-file
    # cells (3 scenarios each at n=32 and n=64) are omitted from the matrix: 16 - 6.
    assert s["by_family"]["eval_grid"]["cells"] == 10 * 20
    assert s["by_family"]["dev_grid"]["cells"] == 12 * 20


def test_separate_streams_keep_development_fixed():
    a = RN._sc("alt-medium", 8, 0.6)
    b = RN._sc("alt-medium", 64, 0.6)
    dev_a, _, _ = st.simulate_scenario_streams(999, a)
    dev_b, _, _ = st.simulate_scenario_streams(999, b)
    assert all(x.N == y.N and abs(x.lam_emp - y.lam_emp) < 1e-15 for x, y in zip(dev_a, dev_b))


def test_support_bound_arithmetic():
    b = an.support_bound([{"withheld": v} for v in [True] * 80 + [False] * 20])
    assert abs(b["observed_max_unconditional_support"] - 0.2) < 1e-12
    assert b["plausible_max_unconditional_support"] >= 0.2
    assert b["screen_power_0_5_possible"] is False


def test_feasibility_row_screens_when_withholding_cannot_fall_below_half():
    row = {"is_drift": False, "true_mu": {"raw-bin|empca": 0.0001171875},
           "claims": {"denoising_benefit": {"is_null": False, "supported": {"rate": 0.02},
                                            "supported_given_not_withheld": {"rate": 0.7}}},
           "coverage_primary": {"raw-bin|empca": {"within_tolerance": False}},
           "withholding": {"rate": 0.96},
           "support_bound": {"plausible_max_unconditional_support": 0.06, "screen_power_0_5_possible": False}}
    fr = an.feasibility_row([row], {"max_withholding_rate": 0.5, "min_power_medium_effect": 0.5}, {"coverage": [0.925, 0.975]})
    assert fr["feasible_under_rules"] is False
    assert fr["screened_infeasible_by_withholding_bound"] is True


def test_fixture_end_to_end_reports_both_modules(tmp_path):
    out = tmp_path / "fx"
    assert main("run", "--out", str(out), "--fixture", "--families", "eval_grid", "dev_grid", "control") == 0
    assert main("reduce", "--out", str(out)) == 0
    red = json.loads((out / "REDUCED.json").read_text())
    assert red["development_only"] is True
    assert red["module_a_eval_grid"] and red["module_b_dev_grid"]
    assert red["feasibility_by_n_files"]
    for row in red["feasibility_by_n_files"].values():
        assert "screened_infeasible_by_withholding_bound" in row
    assert red["development_design_table"]
    assert "P(supported) <= 1 - P(withheld)" in red["statement"]


def test_declares_no_selection_authority():
    P = PR.load_protocol(PROTO)
    assert "pilot_selection_rule" not in P
    assert P["seed_namespaces"]["confirmation"][0] > P["seed_namespaces"]["pilot"][1]
    assert P["inference"]["feasibility"]["min_independent_files"] == 4


# --- A2: R1-failure -> R4-cancellation gate -------------------------------------------
from experiments.mlst2026.followup_20260917.r1_t22_calibration import candidates as C  # noqa: E402


def _write_freeze(path, candidate, dev=False, protocol="p", matrix="m", family=None):
    path.write_text(json.dumps({
        "schema": "mlst-r1-frozen-candidate/1", "chosen_candidate": candidate,
        "development_only": dev, "protocol_sha256": protocol, "matrix_sha256": matrix,
        "candidate_family": family or list(C.CANDIDATES),
        "primary_eligible": list(C.PRIMARY_ELIGIBLE),
        "secondary_diagnostic": list(C.SECONDARY_DIAGNOSTIC),
        "adequacy_rule": C.selection_rule_doc(),
        "frozen_from_pilot_cells": ["cell-a", "cell-b"]}))


def _write_acceptance(path, *, coverage=True, type1=True, candidate="two_stage_t", dev=False,
                      all_infeasible=False, complete=True, protocol="p", matrix="m",
                      explicit=True, primary_eligible=True):
    path.write_text(json.dumps({
        "schema": "mlst-followup-r1-acceptance/2",
        "coverage_ok": {"4": coverage, "8": coverage, "16": coverage},
        "type1_ok": {"4": type1, "8": type1, "16": type1},
        "frozen_candidate": candidate, "development_only": dev,
        "confirmation_complete": complete, "candidate_primary_eligible": primary_eligible,
        "decisions_explicit": explicit, "protocol_sha256": protocol, "matrix_sha256": matrix,
        "withholding_screen_all_infeasible": all_infeasible}))


def test_r1_gate_cancels_on_none_and_is_not_ready_to_transfer(tmp_path):
    fz = tmp_path / "FROZEN_CANDIDATE.json"
    _write_freeze(fz, "none")
    g = RN.gate.evaluate_r1_precondition(fz)
    assert g["status"] == "cancel" and g["trigger"] == "r1_chose_none"
    assert RN.gate.ready_to_transfer(g) is False


def test_r1_gate_passes_only_with_a_frozen_accepted_candidate(tmp_path):
    fz, ac = tmp_path / "fz.json", tmp_path / "ac.json"
    _write_freeze(fz, "two_stage_t")
    _write_acceptance(ac, coverage=True, type1=True)
    g = RN.gate.evaluate_r1_precondition(fz, ac)
    assert g["status"] == "pass" and RN.gate.ready_to_transfer(g) is True
    # absent acceptance evidence with a present freeze is BLOCKED, never pass
    assert RN.gate.evaluate_r1_precondition(fz)["status"] == "blocked"


def test_r1_gate_refuses_target_mismatched_and_empty_receipts(tmp_path):
    fz, ac = tmp_path / "fz.json", tmp_path / "ac.json"
    _write_freeze(fz, "welch_fe_t")               # secondary, not primary-eligible
    _write_acceptance(ac, candidate="welch_fe_t")
    assert RN.gate.evaluate_r1_precondition(fz, ac)["status"] == "blocked"
    _write_freeze(fz, "two_stage_t")
    ac.write_text("{}")                           # an empty acceptance object must not pass
    g = RN.gate.evaluate_r1_precondition(fz, ac)
    assert g["status"] == "blocked" and g["cancel"] is False
    _write_acceptance(ac, complete=False)          # an incomplete confirmation cannot be accepted
    assert RN.gate.evaluate_r1_precondition(fz, ac)["status"] == "blocked"
    _write_acceptance(ac, explicit=False)
    assert RN.gate.evaluate_r1_precondition(fz, ac)["status"] == "blocked"


def test_r1_gate_mutation_flips_the_decision(tmp_path):
    fz, ac = tmp_path / "fz.json", tmp_path / "ac.json"
    _write_freeze(fz, "two_stage_t")
    _write_acceptance(ac, coverage=False)
    g_bad = RN.gate.evaluate_r1_precondition(fz, ac)
    assert g_bad["status"] == "cancel" and g_bad["trigger"] == "r1_interval_failed"
    _write_acceptance(ac, coverage=True)          # the mutation: repair the acceptance
    assert RN.gate.evaluate_r1_precondition(fz, ac)["status"] == "pass"


def test_r1_gate_cancels_on_universal_withholding_screen(tmp_path):
    fz, ac = tmp_path / "fz.json", tmp_path / "ac.json"
    _write_freeze(fz, "two_stage_t")
    _write_acceptance(ac, all_infeasible=True)
    g = RN.gate.evaluate_r1_precondition(fz, ac)
    assert g["status"] == "cancel" and g["trigger"] == "withholding_screen_infeasible"


def test_r4_matrix_omits_large_file_cells_when_the_gate_cancels(monkeypatch):
    P = PR.load_protocol(PROTO)
    real = RN.evaluate_gate

    def cancelled(P_, **kw):
        return {"status": "cancel", "trigger": "r1_chose_none", "cancel": True,
                "cancel_large_file_counts": [32, 64], "reasons": [], "r1": {}}

    monkeypatch.setattr(RN, "evaluate_gate", cancelled)
    m = RN.build_matrix(P, False)
    sizes = {c["params"]["scenario"]["n_files"] for c in m.cells if c["family"] == "eval_grid"}
    assert 32 not in sizes and 64 not in sizes
    # and when the gate passes, they are present
    monkeypatch.setattr(RN, "evaluate_gate", lambda P_, **kw: {"status": "pass", "cancel": False,
                                                               "cancel_large_file_counts": [32, 64],
                                                               "reasons": [], "r1": {}})
    m2 = RN.build_matrix(P, False)
    sizes2 = {c["params"]["scenario"]["n_files"] for c in m2.cells if c["family"] == "eval_grid"}
    assert 32 in sizes2 and 64 in sizes2


def test_r4_unique_large_file_scenarios_are_enumerated_for_cancellation():
    scs = [s for s in RN.eval_scenarios() if s.n_files in (32, 64)]
    assert len(scs) == 6  # 3 scenarios at each of n=32 and n=64
