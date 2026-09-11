"""Tests for the T14 structural-prior runner: budget derivation, fast path,
and the crossover rules declared before confirmation."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.mlst2026 import t14_structural_prior as t14
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY
from experiments.synthetic.nfpa_confirmatory.covariance import _sqrt_inv, whiten_separable


# --- budget derived, not trusted ---------------------------------------------
def test_fits_per_cell_is_derived_from_seeds_methods_and_restarts():
    assert t14.fits_per_cell(10, 3) == 40
    assert t14.fits_per_cell(10, 10) == 110
    assert t14.fits_per_cell(5, 3) == 20


def test_derived_budget_does_not_simply_echo_the_config_value():
    """The config's 40 only holds because the protocol overrides the
    repository's PRIMARY.n_restarts = 10 with 3. Both must be recorded."""
    b = t14.derived_budget()
    assert b["repository_PRIMARY_n_restarts"] == PRIMARY.n_restarts == 10
    assert b["protocol_nfpa_restarts"] == 3
    assert b["fits_per_cell_derived"] == t14.fits_per_cell(10, 3) == 40
    assert t14.fits_per_cell(10, PRIMARY.n_restarts) != b["fits_per_cell_derived"]


def test_total_planned_is_within_both_caps():
    b = t14.derived_budget()
    assert b["stage1_cells"] == 15 and b["stage1_fits"] == 600
    assert b["stage2_cells"] == 10 and b["stage2_fits"] == 400
    assert b["stage3_cells_max"] == 6 and b["stage3_fits_max"] == 240
    assert b["total_planned"] == 1240 <= t14.FIT_CAP_PLANNED
    assert b["total_planned"] < t14.FIT_CAP_HARD == 1500
    assert b["within_planned_cap"] and b["within_hard_cap"]


def test_stage_cells_are_unique_and_match_the_declared_grids():
    c1, c2 = t14.stage_cells(1), t14.stage_cells(2)
    assert len(c1) == 15 and len(set(c1)) == 15
    assert len(c2) == 10 and len(set(c2)) == 10
    assert {c[1] for c in c1} == {0.0, 0.4, 0.8}
    assert {c[2] for c in c1} == {10}
    assert {c[1] for c in c2} == {0.0, 0.8}
    assert {c[2] for c in c2} == {3}
    assert set(c1).isdisjoint(c2)


def test_seed_sets_are_pairwise_disjoint():
    a, b, c = set(t14.SEEDS_EXPLORATION), set(t14.SEEDS_CONFIRMATION), set(t14.SEEDS_SMOKE)
    assert not (a & b) and not (a & c) and not (b & c)
    t14._check_seed_disjointness()


# --- fast path equals the archival algebra -----------------------------------
def test_fast_whiten_matches_the_archival_whitener_to_machine_precision():
    rng = np.random.default_rng(0)
    C, T, n = 8, 64, 40
    A = rng.standard_normal((C, C)); Sc = A @ A.T + C * np.eye(C)
    B = rng.standard_normal((T, T)); St = B @ B.T + T * np.eye(T)
    X = rng.standard_normal((n, C, T))
    ref = whiten_separable(X, Sc, St)
    got = t14.fast_whiten(X, _sqrt_inv(Sc), _sqrt_inv(St))
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_cw_pca_projector_is_the_thin_svd_comparator():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((50, 8, 16))
    P = t14.cw_pca_projector(Z, 6)
    assert P.shape == (8 * 16, 6)
    np.testing.assert_allclose(P.T @ P, np.eye(6), atol=1e-10)
    F = Z.reshape(50, -1, order="F")
    # the rank-6 projection must be the best rank-6 approximation of F
    err = np.linalg.norm(F - F @ (P @ P.T))
    s = np.linalg.svd(F, compute_uv=False)
    assert err == pytest.approx(np.sqrt(np.sum(s[6:] ** 2)), rel=1e-8)


def test_whitened_chi2_is_the_mean_squared_whitened_residual():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[1.0, 1.0], [3.0, 2.0]])
    assert t14.whitened_chi2(a, b) == pytest.approx((0 + 1 + 0 + 4) / 4)


# --- crossover rules ----------------------------------------------------------
def _cell(N, d, resolved=True, alpha=0.4, snr=10, cov="oracle"):
    w = abs(d) / 4 if d else 0.001
    return {"N": N, "alpha": alpha, "snr": snr, "cov_arm": cov,
            "d_bar": d, "mean": d, "ci95_lo": d - w, "ci95_hi": d + w,
            "excludes_zero": resolved, "n_seeds": 10}


def test_predicted_direction_crossing_is_detected_and_selected():
    s = {f"c{N}": _cell(N, d) for N, d in
         zip(t14.N_GRID, (-0.01, -0.005, +0.004, +0.01, +0.02))}
    out = t14.find_crossovers(s)
    assert out["triggers_confirmation"]
    assert out["selected"]["N_lower"] == 400 and out["selected"]["N_upper"] == 800
    assert not out["reverse_crossings"]


def test_reverse_direction_crossing_never_triggers_confirmation():
    s = {f"c{N}": _cell(N, d) for N, d in
         zip(t14.N_GRID, (+0.01, +0.005, -0.004, -0.01, -0.02))}
    out = t14.find_crossovers(s)
    assert not out["triggers_confirmation"]
    assert out["selected"] is None
    assert len(out["reverse_crossings"]) == 1


def test_unresolved_endpoints_make_a_crossing_ambiguous_not_confirmatory():
    s = {f"c{N}": _cell(N, d, resolved=False) for N, d in
         zip(t14.N_GRID, (-0.01, -0.005, +0.004, +0.01, +0.02))}
    out = t14.find_crossovers(s)
    assert not out["triggers_confirmation"]
    assert len(out["ambiguous_crossings"]) == 1
    assert not out["resolved_crossings"]


def test_multiple_crossings_select_the_smallest_N():
    s = {f"c{N}": _cell(N, d) for N, d in
         zip(t14.N_GRID, (-0.01, +0.005, -0.004, +0.01, +0.02))}
    out = t14.find_crossovers(s)
    assert len(out["resolved_crossings"]) == 2
    assert out["selected"]["N_lower"] == 200
    assert len(out["reverse_crossings"]) == 1


def test_no_crossing_means_no_confirmation_and_a_reportable_null():
    s = {f"c{N}": _cell(N, -0.01) for N in t14.N_GRID}
    out = t14.find_crossovers(s)
    assert not out["triggers_confirmation"] and out["selected"] is None
    assert not out["resolved_crossings"] and not out["reverse_crossings"]


def test_boundary_neighbours_still_give_three_N_values_and_six_cells():
    for N, expect in ((200, [200, 400, 800]),
                      (3200, [800, 1600, 3200]),
                      (800, [400, 800, 1600])):
        cells = t14.confirmation_cells({"N_lower": N, "alpha": 0.4, "snr": 10})
        assert sorted({c[0] for c in cells}) == expect
        assert len(cells) == 6
        assert {c[3] for c in cells} == set(t14.COV_ARMS)


def test_confirmation_cells_never_exceed_the_declared_maximum():
    b = t14.derived_budget()
    for N in t14.N_GRID:
        cells = t14.confirmation_cells({"N_lower": N, "alpha": 0.0, "snr": 3})
        assert len(cells) <= b["stage3_cells_max"] == 6


def test_paired_interval_is_deterministic_and_brackets_the_mean():
    d = np.array([-0.01, -0.02, 0.0, -0.015, -0.005, -0.01, -0.02, -0.01, -0.03, -0.01])
    a = t14.paired_interval(d, reps=2000)
    b = t14.paired_interval(d, reps=2000)
    assert a == b
    assert a["ci95_lo"] <= a["mean"] <= a["ci95_hi"]
    assert a["excludes_zero"] is True


def test_estimated_covariance_arm_uses_no_ground_truth_scale():
    """The flip-flop estimate must be identified from the calibration traces
    alone: Sigma_c normalised to trace/C = 1, Sigma_t carrying the variance."""
    from dataclasses import replace
    from experiments.synthetic.nfpa_confirmatory.generator import make_dataset
    cfg = replace(PRIMARY, N_train=50, N_validation=10, N_test=10, T=32,
                  nonseparability_alpha=0.4, n_restarts=1)
    ds = make_dataset(0, "colored", cfg)
    cov = t14.estimated_covariances(ds, 128, 0)
    assert np.trace(cov["Sigma_c"]) / cfg.C == pytest.approx(1.0, rel=1e-6)
    assert np.linalg.eigvalsh(cov["Sigma_c"]).min() > 0
    assert np.linalg.eigvalsh(cov["Sigma_t"]).min() > 0
