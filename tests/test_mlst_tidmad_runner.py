"""Tests for the TIDMAD stage runner allocation, roles and freeze gate."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.mlst2026 import t10_tidmad_runner as R


def test_stretch_length_matches_the_protocol_arithmetic():
    assert R.STRETCH_SAMPLES == 36_995_072
    assert R.STRETCH_SAMPLES / 10_000_000 == pytest.approx(3.6995072)
    assert R.GUARD_SAMPLES == 16384
    assert R.GUARD_SAMPLES / 10_000_000 == pytest.approx(0.0016384)


def test_leading_use_and_leftover_match_the_protocol_numbers():
    assert R.allocation(4096)["used_samples"] == 2_342_912
    assert R.allocation(16384)["used_samples"] == 9_273_344
    assert R.allocation(65536)["used_samples"] == R.STRETCH_SAMPLES
    assert R.allocation(65536)["leftover_samples"] == 0
    assert R.allocation(4096)["leftover_samples"] == 36_995_072 - 2_342_912


def test_every_window_length_keeps_150_evaluation_windows():
    for w in R.WINDOW_GRID:
        a = R.allocation(w)
        assert a["n_eval"] == 150 and a["n_cal"] == 64 and a["n_fit"] == 350


def test_blocks_abut_with_exactly_one_guard_between_them():
    for w in R.WINDOW_GRID:
        a = R.allocation(w)
        assert a["fit_offset"] == a["cal_offset"] + 64 * w + R.GUARD_SAMPLES
        assert a["eval_offset"] == a["fit_offset"] + 350 * w + R.GUARD_SAMPLES
        assert a["eval_offset"] + 150 * w == a["used_samples"]


def test_development_start_rule():
    assert R.DEV_START == 2_097_152 + 5_000_000


def test_file_roles_are_disjoint_and_complete():
    assert set(R.DEV_FILES).isdisjoint(R.POOL_FILES)
    assert len(R.POOL_FILES) == 17
    assert len(set(R.DEV_FILES) | set(R.POOL_FILES)) == 20


def test_fit_cap_is_respected_by_the_declared_stages():
    dev = len(R.DEV_FILES) * len(R.WINDOW_GRID) * 2   # PCA + EMPCA per stretch
    val = 3 * 2
    test = 14 * 2
    assert dev + val + test <= 200


def test_seed_offsets_are_process_stable():
    """A salted hash here would silently break reproducibility across runs."""
    assert R.stable_offset("abra_validation_0001.h5|w=4096") == \
           R.stable_offset("abra_validation_0001.h5|w=4096")
    assert R.stable_offset("a") != R.stable_offset("b")
    assert R.stable_offset("abra_validation_0001.h5|w=4096") == 68
    keys = [f"{n}|w={w}" for n in list(R.DEV_FILES) + list(R.POOL_FILES)
            for w in R.WINDOW_GRID]
    offsets = [R.stable_offset(k) for k in keys]
    assert len(set(offsets)) == len(keys)  # no two stretches share a seed


def _ch(outcomes):
    return {f"{n}|w=4096": {"outcome": o} for n, o in outcomes.items()}


def test_validation_is_the_first_three_eligible_pool_files_in_index_order():
    o = {n: "uncoupled" for n in R.POOL_FILES}
    for n in ("abra_validation_0002.h5", "abra_validation_0004.h5",
              "abra_validation_0009.h5", "abra_validation_0013.h5"):
        o[n] = "eligible-coupled"
    roles = R.assign_pool_roles(_ch(o), 4096)
    assert roles["validation_files"] == ["abra_validation_0002.h5",
                                         "abra_validation_0004.h5",
                                         "abra_validation_0009.h5"]
    assert "abra_validation_0013.h5" in roles["final_test_files"]
    assert roles["validation_status"] == "complete"
    assert len(roles["final_test_files"]) == 14


def test_fewer_than_three_eligible_files_is_resolved_deterministically():
    o = {n: "uncoupled" for n in R.POOL_FILES}
    o["abra_validation_0006.h5"] = "eligible-coupled"
    roles = R.assign_pool_roles(_ch(o), 4096)
    assert roles["validation_files"] == ["abra_validation_0006.h5"]
    assert roles["validation_status"] == "underpowered"
    assert roles["useful_verdict_permitted"] is True
    assert len(roles["final_test_files"]) == 16


def test_no_eligible_pool_file_forbids_a_useful_verdict():
    o = {n: "uncoupled" for n in R.POOL_FILES}
    roles = R.assign_pool_roles(_ch(o), 4096)
    assert roles["validation_files"] == []
    assert roles["validation_status"] == "empty"
    assert roles["useful_verdict_permitted"] is False
    assert roles["final_test_files"] == list(R.POOL_FILES)


def test_every_pool_file_is_evaluated_exactly_once_across_the_two_stages():
    o = {n: "uncoupled" for n in R.POOL_FILES}
    for n in list(R.POOL_FILES)[:3]:
        o[n] = "eligible-coupled"
    roles = R.assign_pool_roles(_ch(o), 4096)
    seen = roles["validation_files"] + roles["final_test_files"]
    assert sorted(seen) == sorted(R.POOL_FILES)
    assert len(seen) == len(set(seen))


def test_selection_reports_no_operating_point_when_nothing_is_eligible():
    dev = {"f|w=4096": {"window_samples": 4096,
                        "characterisation": {"outcome": "uncoupled"},
                        "contrasts": {"zero|empca": {"mean_delta_squared": 1.0,
                                                     "ci95_lo": 0.5, "ci95_hi": 1.5}}}}
    out = R.choose_operating_point(dev)
    assert out["selected_window"] is None
    assert out["verdict"] == "no-eligible-development-stretch"


def test_selection_breaks_ties_towards_the_smaller_window():
    def row(w, d):
        return {"window_samples": w, "characterisation": {"outcome": "eligible-coupled"},
                "contrasts": {"zero|empca": {"mean_delta_squared": d,
                                             "ci95_lo": d - 0.196, "ci95_hi": d + 0.196}}}
    dev = {"a|w=4096": row(4096, 1.0), "b|w=16384": row(16384, 1.0)}
    assert R.choose_operating_point(dev)["selected_window"] == 4096


def test_windows_helper_is_exact_and_non_overlapping():
    x = np.arange(100, dtype=float)
    W = R._windows(x, 10, 3, 20)
    assert W.shape == (3, 20)
    np.testing.assert_array_equal(W[0], np.arange(10, 30))
    np.testing.assert_array_equal(W[2], np.arange(50, 70))
