"""Multi-seed acceptance gates for the synthetic suite, via the harness.

Each test encodes a gate's acceptance as a seed-swept bootstrap/paired interval
(item 5: every passed equivalence becomes a regression test). Held-out evaluation
is used where the gate fits a subspace (S4/S5/S6).
"""

import numpy as np

from src.noise_geometry.experiments.synthetic import (
    run_s1_of_crb,
    run_s4_white_control,
    run_s6_timing_rank_sweep,
    run_s7_covariance_robustness,
    run_s8_residual_calibration,
    run_s9_multichannel,
)
from src.noise_geometry.validation import paired_difference, run_seed_sweep, summarize

SEEDS = list(range(8))


def test_s1_sigma_over_crb_contains_one_and_unbiased():
    rows = run_seed_sweep(run_s1_of_crb, {"n_traces": 2048, "n_samples": 512}, SEEDS)
    s = summarize(rows, keys=["sigma_over_crb", "bias"])
    lo, hi = s["sigma_over_crb"]["ci95"]
    assert lo <= 1.0 <= hi, f"sigma/CRB 95% CI {(lo, hi)} excludes 1"
    blo, bhi = s["bias"]["ci95"]
    assert blo <= 0.0 <= bhi, f"amplitude bias 95% CI {(blo, bhi)} excludes 0"


def test_s4_pca_and_empca_agree_in_white_noise():
    rows = run_seed_sweep(run_s4_white_control, {"test_frac": 0.5}, SEEDS)
    angle = summarize(rows, keys=["max_principal_angle_deg"])["max_principal_angle_deg"]
    assert angle["mean"] < 1e-3, f"PCA/EMPCA disagree in white noise: {angle['mean']:.2e} deg"
    # negative control: weighted residuals are equal -> paired CI contains 0
    pd = paired_difference(rows, "pca_weighted_residual", "empca_weighted_residual")
    assert not pd["ci95_excludes_zero"], "white-noise control should not show a difference"


def test_s6_higher_rank_helps_under_timing_jitter():
    rows = run_seed_sweep(run_s6_timing_rank_sweep, {"test_frac": 0.5}, SEEDS)
    # rank 1 (col .0) vs rank 6 (col .5): extra ranks lower the held-out residual
    pd = paired_difference(rows, "weighted_residual_by_rank.0", "weighted_residual_by_rank.5")
    assert pd["mean_difference"] > 0 and pd["ci95_excludes_zero"]
    best = summarize(rows, keys=["best_rank_by_clean_mse"])["best_rank_by_clean_mse"]
    assert best["mean"] > 1.0, "timing jitter should favour rank > 1"


def test_s7_estimated_covariance_converges_to_oracle():
    rows = run_seed_sweep(run_s7_covariance_robustness, {"n_eval_traces": 600}, SEEDS)
    s = summarize(rows, keys=["sigma_over_oracle.0", "sigma_over_oracle.4"])
    small_n, large_n = s["sigma_over_oracle.0"]["mean"], s["sigma_over_oracle.4"]["mean"]
    assert large_n < small_n, "more calibration data should approach the oracle"
    assert abs(large_n - 1.0) < 0.1, f"largest-sample sigma/oracle {large_n:.3f} not near 1"


def test_s8_residuals_are_calibrated():
    rows = run_seed_sweep(run_s8_residual_calibration, {}, SEEDS)
    chi2 = summarize(rows, keys=["mean_chi2_per_dof"])["mean_chi2_per_dof"]
    lo, hi = chi2["ci95"]
    assert lo <= 1.0 <= hi or abs(chi2["mean"] - 1.0) < 0.05, f"chi2/dof {chi2['mean']:.3f} not ~1"


def test_s9_full_covariance_beats_diagonal_under_correlation():
    rows = run_seed_sweep(run_s9_multichannel, {"n_traces": 600}, SEEDS)
    # full covariance gives a smaller true-metric weighted residual than diagonal
    pd = paired_difference(rows, "diagonal_true_weighted_residual", "full_true_weighted_residual")
    assert pd["mean_difference"] > 0 and pd["ci95_excludes_zero"]
