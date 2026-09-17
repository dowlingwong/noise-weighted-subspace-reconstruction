"""A1: every compute package carries a machine-readable execution-mode decision.

The rule: serial unless a package's serial wall time exceeds its reviewed budget
and a bounded shard/merge design meets all five conditions.  No CephFS-safe
ledger rewrite may be a silent dependency of a science package.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "experiments" / "mlst2026" / "followup_20260917"
CONDITIONS = ("disjoint_cell_ownership", "immutable_receipts", "duplicate_missing_detection",
              "every_charged_attempt_preserved", "target_filesystem_locking_validated")
PACKAGES = ("r1_t22_calibration", "r2_t20_theta012", "r3_t21_finite", "r4_t22_feasibility", "r5_t19_reach")


def _load(pkg):
    return json.loads((BASE / pkg / "EXECUTION_MODE.json").read_text())


def test_every_compute_package_has_a_valid_execution_mode_decision():
    for pkg in PACKAGES:
        d = _load(pkg)
        assert d["schema"] == "mlst-followup-execution-mode/1"
        assert d["decision"] in ("serial", "sharded")
        assert isinstance(d["serial_wall_hours_estimate"], (int, float))
        assert "planning_anchor_hours" in d and isinstance(d["fits_planning_anchor"], bool)
        assert "lock_rewrite_required" in d and d["lock_rewrite_required"] is False
        if d["decision"] == "serial":
            assert "rationale" in d and d["rationale"]
        else:
            assert all(d["sharding"]["conditions_met"][c] for c in CONDITIONS), \
                f"{pkg}: sharded without all five conditions met"


def test_no_package_silently_depends_on_a_lock_rewrite():
    agg = json.loads((BASE / "EXECUTION_MODES.json").read_text())
    assert agg["lock_rewrite_required_any"] is False
    for pkg, d in agg["packages"].items():
        assert d["lock_rewrite_required"] is False


def test_r1_borderline_case_is_decided_explicitly():
    d = _load("r1_t22_calibration")
    assert d["borderline"] is True
    assert d["decision"] == "serial"
    # the decision is justified, not asserted
    assert "borderline" in d["rationale"]
    assert d["serial_wall_hours_estimate"] >= d["planning_anchor_hours"] * 0.9


def test_r4_declares_the_r1_cancellation_gate():
    d = _load("r4_t22_feasibility")
    assert d["r1_cancellation_gate"]["declared"] is True
    assert "CANCELLED" in d["r1_cancellation_gate"]["effect"]


def test_aggregate_matches_the_per_package_files():
    agg = json.loads((BASE / "EXECUTION_MODES.json").read_text())
    assert set(agg["packages"]) == set(PACKAGES)
    for pkg in PACKAGES:
        assert agg["packages"][pkg] == _load(pkg)
