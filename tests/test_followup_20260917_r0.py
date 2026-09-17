"""R0 reporting-overlay tests.

These are scientific mutation tests, not hash tests: they fail if the H5
random-effects summary is omitted, if the wrong contrast or denominator is used,
or if the interval bounds are perturbed.  A gated integration test checks the
committed evidence export without touching the 758 MB archive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.mlst2026.followup_20260917.r0_t22_h5 import reducer as R  # noqa: E402

EXPORT = REPO / "experiments" / "mlst2026" / "followup_20260917" / "r0_t22_h5" / "evidence_export"


def _rec(*, mu, re_lo, re_hi, point, covered_two=True, covered_t=False, withheld=False,
         contrast="raw-bin|empca"):
    return {
        "withheld": withheld,
        "scenario": "synthetic",
        "contrasts": {
            contrast: {
                "true_mu": mu, "covers": covered_two, "t_covers": covered_t,
                "ci95_lo": point - 1, "ci95_hi": point + 1, "point": point,
                "bootstrap_sd": 0.1, "length": 2.0, "t_length": 2.0,
                "t_lo": point - 1, "t_hi": point + 1,
                "re_estimate": point, "re_lo": re_lo, "re_hi": re_hi,
                "per_file": {"f0": {"mean": point, "bootstrap_mean": point, "bootstrap_sd": 0.1},
                             "f1": {"mean": point, "bootstrap_mean": point, "bootstrap_sd": 0.1}},
            },
            "zero|empca": {"true_mu": mu, "covers": True, "t_covers": True, "point": point,
                           "bootstrap_sd": 0.1, "length": 2.0, "t_length": 2.0, "re_estimate": point,
                           "re_lo": re_lo, "re_hi": re_hi,
                           "ci95_lo": point - 1, "ci95_hi": point + 1, "t_lo": point - 1, "t_hi": point + 1,
                           "per_file": {}},
            "pca|empca": {"true_mu": mu, "covers": True, "t_covers": True, "point": point,
                          "bootstrap_sd": 0.1, "length": 2.0, "t_length": 2.0, "re_estimate": point,
                          "re_lo": re_lo, "re_hi": re_hi,
                          "ci95_lo": point - 1, "ci95_hi": point + 1, "t_lo": point - 1, "t_hi": point + 1,
                          "per_file": {}},
        },
    }


def test_wilson_and_empty_denominator():
    w = R.wilson(0, 0)
    assert w["rate"] is None and w["n"] == 0 and w["hi"] is None
    assert R.within_band(w) is None
    w = R.wilson(950, 1000)
    assert abs(w["rate"] - 0.95) < 1e-12 and w["lo"] < w["rate"] < w["hi"]


def test_coverage_uses_the_stored_estimand_and_re_bounds():
    recs = [_rec(mu=0.10, re_lo=0.05, re_hi=0.15, point=0.10)] * 7 + \
           [_rec(mu=0.10, re_lo=0.20, re_hi=0.30, point=0.25)] * 3
    s = R.summarise_records("synthetic", recs, {"drift_sd": 0.0})
    c = s["contrasts"][R.PRIMARY_CONTRAST]["coverage_unconditional"]["random_effects"]
    assert c["k"] == 7 and c["n"] == 10 and abs(c["rate"] - 0.7) < 1e-12


def test_perturbing_interval_bounds_changes_the_summary():
    recs = [_rec(mu=0.10, re_lo=0.05, re_hi=0.15, point=0.10)] * 10
    before = R.summarise_records("synthetic", recs, {"drift_sd": 0.0})
    perturbed = json.loads(json.dumps(recs))
    for r in perturbed:                       # push every RE interval off the truth
        r["contrasts"][R.PRIMARY_CONTRAST]["re_hi"] = 1e9
        r["contrasts"][R.PRIMARY_CONTRAST]["re_lo"] = 1e8
    after = R.summarise_records("synthetic", perturbed, {"drift_sd": 0.0})
    assert before["contrasts"][R.PRIMARY_CONTRAST]["coverage_unconditional"]["random_effects"]["k"] == 10
    assert after["contrasts"][R.PRIMARY_CONTRAST]["coverage_unconditional"]["random_effects"]["k"] == 0


def test_wrong_contrast_would_be_caught_by_is_primary_marker():
    recs = [_rec(mu=0.10, re_lo=0.05, re_hi=0.15, point=0.10)] * 5
    s = R.summarise_records("synthetic", recs, {"drift_sd": 0.0})
    assert s["contrasts"][R.PRIMARY_CONTRAST]["is_primary"] is True
    assert all(not s["contrasts"][k]["is_primary"] for k in s["contrasts"] if k != R.PRIMARY_CONTRAST)


def test_not_withheld_denominator_is_explicit():
    recs = [_rec(mu=0.10, re_lo=0.05, re_hi=0.15, point=0.10)] * 4 + \
           [_rec(mu=0.10, re_lo=0.05, re_hi=0.15, point=0.10, withheld=True)] * 6
    s = R.summarise_records("synthetic", recs, {"drift_sd": 0.0})
    c = s["contrasts"][R.PRIMARY_CONTRAST]
    assert c["coverage_unconditional"]["random_effects"]["n"] == 10
    assert c["coverage_given_not_withheld"]["random_effects"]["n"] == 4
    assert c["coverage_given_not_withheld"]["random_effects"]["k"] == 4


def test_target_mismatch_is_flagged_when_the_weighting_is_displaced():
    # equal-file mean is 0.10; the RE estimate sits ~0.5 interval-widths above it
    recs = []
    for i in range(20):
        r = _rec(mu=0.10, re_lo=0.095, re_hi=0.115, point=0.10)
        r["contrasts"][R.PRIMARY_CONTRAST]["re_estimate"] = 0.108
        r["contrasts"][R.PRIMARY_CONTRAST]["re_lo"] = 0.095
        r["contrasts"][R.PRIMARY_CONTRAST]["re_hi"] = 0.115
        recs.append(r)
    s = R.summarise_records("synthetic", recs, {"drift_sd": 0.0})
    assert s["contrasts"][R.PRIMARY_CONTRAST]["target"]["mismatch_flag"] is True


def test_manuscript_table_has_the_three_declared_scopes():
    recs = [_rec(mu=0.10, re_lo=0.05, re_hi=0.15, point=0.10)]
    report = {
        "scope_tables": {
            "non_drift_65": {"n_scenarios": 1, "random_effects_in_band": 1, "random_effects_coverage_median": 0.9,
                             "random_effects_coverage_range": [0.9, 0.9], "two_stage_coverage_median": 0.95,
                             "t_interval_coverage_median": 0.93, "median_length_ratio_random_effects_over_two_stage": 0.6,
                             "mismatch_flagged_scenarios": []},
            "drift_stress_4": {"n_scenarios": 1, "random_effects_in_band": 0, "random_effects_coverage_median": 0.8,
                               "random_effects_coverage_range": [0.8, 0.8], "two_stage_coverage_median": 0.9,
                               "t_interval_coverage_median": 0.9, "median_length_ratio_random_effects_over_two_stage": 0.7,
                               "mismatch_flagged_scenarios": []},
            "all_69_descriptive": {"n_scenarios": 2, "random_effects_in_band": 1, "random_effects_coverage_median": 0.85,
                                   "random_effects_coverage_range": [0.8, 0.9], "two_stage_coverage_median": 0.9,
                                   "t_interval_coverage_median": 0.9, "median_length_ratio_random_effects_over_two_stage": 0.6,
                                   "mismatch_flagged_scenarios": []},
        }
    }
    md = R.manuscript_table(report)
    assert "non-drift (H1/H5 scope)" in md and "drift stress (separate)" in md and "descriptive" in md


def test_verify_fails_if_the_summary_is_omitted():
    nd = {"n_scenarios": 65, "random_effects_in_band": 24, "random_effects_coverage_median": 0.876}
    report = {"scope_tables": {"non_drift_65": nd},
              "consistency_with_frozen_reduction": {"scenarios_differing": 0}}
    assert R.verify(report)["all_passed"] is True
    del report["scope_tables"]["non_drift_65"]
    with pytest.raises(KeyError):
        R.verify(report)


@pytest.mark.skipif(not (EXPORT / "H5_RANDOM_EFFECTS_REDUCTION.json").exists(),
                    reason="R0 evidence export not built in this checkout")
def test_committed_evidence_export_matches_the_frozen_reduction():
    prov = json.loads((EXPORT / "PROVENANCE.json").read_text())
    assert prov["reduced_json_sha256"] == "55c96c4b40e4631c5da16650394f92a1516a3ce95f82fe5629498840424d916c"
    assert prov["verification"]["all_passed"] is True
    # the export's own copy of the frozen reduction is byte-identical (same hash)
    import hashlib
    frozen = (EXPORT / "frozen" / "REDUCED.json").read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == prov["reduced_json_sha256"]
    rep = json.loads((EXPORT / "H5_RANDOM_EFFECTS_REDUCTION.json").read_text())
    nd = rep["scope_tables"]["non_drift_65"]
    assert nd["random_effects_in_band"] == 24 and abs(nd["random_effects_coverage_median"] - 0.876) <= 0.001
    assert rep["consistency_with_frozen_reduction"]["scenarios_differing"] == 0
