"""Unit tests for the Stage C3 CRESST Validation Protocol components.

Every test runs on synthetic data only (no real download required), per the
CRESST_REMOTE_AGENT_RUNBOOK requirement that each new component gets a test
runnable without real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.noise_geometry.cresst.detector_metrics import (
    amplitude_error_summary,
    background_thresholds,
    bootstrap_ci,
    detection_score,
    efficiency_amplitude_thresholds,
    efficiency_curve,
    evaluate_trigger_efficiency,
    paired_bootstrap_stats,
    residual_whitening_ratio,
)
from src.noise_geometry.cresst.injections import inject_direct, inject_measured_psd
from src.noise_geometry.cresst.negative_controls import (
    CONTROL_NOMINAL,
    compute_negative_control_residuals,
    random_weighted_orthonormal_basis,
)
from src.noise_geometry.cresst.splits import (
    count_duplicate_traces,
    select_by_groups,
    time_disjoint_split,
    unique_groups,
)
from src.noise_geometry.cresst.templates import build_multi_template_bank, score_multi_template_of
from src.noise_geometry.metrics import weighted_residual
from src.noise_geometry.noise import estimate_covariance, inverse_covariance, regularize_covariance
from src.noise_geometry.subspace import fit_weighted_pca, project_onto_basis


# --------------------------------------------------------------------------- #
# splits.py
# --------------------------------------------------------------------------- #
def _grouped_features(group_sizes: dict[tuple, int]) -> pd.DataFrame:
    rows = []
    for (run, channel), n in group_sizes.items():
        for _ in range(n):
            rows.append({"run": run, "channel": channel})
    return pd.DataFrame(rows)


def test_unique_groups_and_select_by_groups():
    features = _grouped_features({(1, 1): 3, (1, 2): 2})
    assert unique_groups(features, ("run", "channel")) == [(1, 1), (1, 2)]
    mask = select_by_groups(features, ("run", "channel"), {(1, 1)})
    assert mask.tolist() == [True, True, True, False, False]


def test_time_disjoint_split_is_chronological_and_disjoint():
    rng = np.random.default_rng(0)
    hours = rng.uniform(0.0, 100.0, size=200)
    calib_mask, eval_mask = time_disjoint_split(hours, calibration_fraction=0.7)
    assert np.array_equal(eval_mask, ~calib_mask)
    assert calib_mask.sum() == pytest.approx(140, abs=1)
    # every calibration hour must be <= every evaluation hour (chronological, not random)
    assert hours[calib_mask].max() <= hours[eval_mask].min()


def test_time_disjoint_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        time_disjoint_split(np.arange(10.0), calibration_fraction=1.5)


def test_count_duplicate_traces():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [5.0, 6.0], [1.0, 2.0]], dtype=np.float32)
    assert count_duplicate_traces(X) == 2  # two extra copies of [1,2]
    assert count_duplicate_traces(X[:2]) == 0


# --------------------------------------------------------------------------- #
# templates.py
# --------------------------------------------------------------------------- #
def _synthetic_pulse(n_features: int, rise: float, decay: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_features)
    return np.where(
        t < 0.4,
        np.exp(-((t - 0.4) ** 2) / (2 * rise**2)),
        np.exp(-(t - 0.4) / decay),
    )


def test_build_multi_template_bank_cells_and_scoring():
    rng = np.random.default_rng(1)
    n_features = 128
    n_pulses = 200
    rise_time = rng.uniform(0.02, 0.1, size=n_pulses)
    decay_time = rng.uniform(0.1, 0.4, size=n_pulses)
    pulses = np.array([_synthetic_pulse(n_features, r, d) for r, d in zip(rise_time, decay_time)])
    pulses += rng.normal(scale=0.01, size=pulses.shape)

    bank = build_multi_template_bank(
        pulses, rise_time, decay_time,
        rise_time_quantiles=(1 / 3, 2 / 3),
        decay_time_quantiles=(1 / 3, 2 / 3),
        t0_grid_samples=(-2, 0, 2),
        min_cell_pulses=2,
    )
    assert bank.templates.shape[0] <= 9
    assert bank.templates.shape[1] == n_features

    metric = np.eye(n_features)
    scored = score_multi_template_of(pulses, bank, metric)
    assert scored["weighted_residual"].shape == (n_pulses,)
    assert np.all(np.isfinite(scored["weighted_residual"]))
    # the bank should fit these pulses (drawn from the same family) far better
    # than a badly mismatched single flat template.
    flat_template = np.ones(n_features) / np.sqrt(n_features)
    denom = float(flat_template @ metric @ flat_template)
    amp = (pulses @ metric @ flat_template) / denom
    flat_recon = amp[:, None] * flat_template[None, :]
    from src.noise_geometry.metrics import weighted_residual

    flat_residual = np.mean(weighted_residual(pulses, flat_recon, metric))
    assert np.mean(scored["weighted_residual"]) < flat_residual


def test_score_multi_template_of_recovers_best_shift():
    n_features = 64
    template = _synthetic_pulse(n_features, 0.03, 0.15)
    rise_time = np.array([0.03])
    decay_time = np.array([0.15])
    bank = build_multi_template_bank(
        template[None, :], rise_time, decay_time,
        rise_time_quantiles=(), decay_time_quantiles=(),
        t0_grid_samples=(-4, -2, 0, 2, 4), min_cell_pulses=1,
    )
    shifted = np.zeros_like(template)
    shifted[2:] = template[:-2]  # a +2 sample shift
    metric = np.eye(n_features)
    scored = score_multi_template_of(shifted[None, :], bank, metric)
    assert int(scored["shift"][0]) == 2


# --------------------------------------------------------------------------- #
# injections.py
# --------------------------------------------------------------------------- #
def test_inject_direct_adds_template_to_real_noise():
    rng = np.random.default_rng(2)
    noise = rng.normal(scale=0.1, size=(20, 16))
    template = np.zeros(16)
    template[4:8] = 1.0
    amplitudes = np.array([0.0, 1.0, 2.0])
    injected, idx = inject_direct(noise, template, amplitudes, rng)
    for i, amp in enumerate(amplitudes):
        np.testing.assert_allclose(injected[i], noise[idx[i]] + amp * template)


def test_inject_measured_psd_matches_integrated_variance():
    rng = np.random.default_rng(3)
    n_features = 32
    true_cov = np.eye(n_features) * 0.05
    noise = rng.multivariate_normal(np.zeros(n_features), true_cov, size=2000)
    template = np.zeros(n_features)
    amplitudes = np.zeros(500)
    gaussian_injected = inject_measured_psd(noise, template, amplitudes, rng, shrinkage=0.0)
    real_variance = float(np.mean(np.var(noise, axis=0)))
    gaussian_variance = float(np.mean(np.var(gaussian_injected, axis=0)))
    assert gaussian_variance == pytest.approx(real_variance, rel=0.25)


# --------------------------------------------------------------------------- #
# detector_metrics.py
# --------------------------------------------------------------------------- #
def test_detection_score_zero_reconstruction_matches_chi2_null():
    rng = np.random.default_rng(4)
    test = rng.normal(size=(10, 6))
    metric = np.eye(6)
    score = detection_score(test, np.zeros_like(test), metric)
    np.testing.assert_allclose(score, np.zeros(10))


def test_detection_score_positive_for_good_reconstruction():
    rng = np.random.default_rng(5)
    test = rng.normal(size=(10, 6)) + 5.0
    metric = np.eye(6)
    recon = np.full_like(test, 5.0)
    score = detection_score(test, recon, metric)
    assert np.all(score > 0)


def test_bootstrap_ci_contains_true_mean():
    rng = np.random.default_rng(6)
    values = rng.normal(loc=2.0, scale=1.0, size=500)
    lo, hi = bootstrap_ci(values, level=0.95, n_boot=300, seed=1)
    assert lo < 2.0 < hi


def test_background_thresholds_are_upper_quantiles():
    scores = np.arange(1000, dtype=float)
    thresholds = background_thresholds(scores, [0.01, 0.05])
    assert thresholds[0.01] == pytest.approx(np.quantile(scores, 0.99))
    assert thresholds[0.01] > thresholds[0.05]


def test_efficiency_curve_and_amplitude_thresholds():
    scores_by_amplitude = [np.full(100, v) for v in [0.0, 5.0, 10.0, 15.0]]
    eff = efficiency_curve(scores_by_amplitude, threshold=7.0)
    np.testing.assert_allclose(eff, [0.0, 0.0, 1.0, 1.0])
    thresholds = efficiency_amplitude_thresholds(
        np.array([0.0, 5.0, 10.0, 15.0]), np.array([0.0, 0.0, 1.0, 1.0]), levels=(0.5, 0.9)
    )
    assert 5.0 <= thresholds[0.5] <= 10.0


def test_residual_whitening_ratio_near_one_for_matched_noise():
    rng = np.random.default_rng(7)
    n_features = 64
    noise = rng.normal(scale=1.0, size=(4000, n_features))
    residuals = rng.normal(scale=1.0, size=(4000, n_features))
    out = residual_whitening_ratio(residuals, noise)
    ratio = np.asarray(out["whitening_ratio"])
    assert np.median(ratio) == pytest.approx(1.0, abs=0.3)


def test_paired_bootstrap_stats_excludes_zero_for_clear_difference():
    rng = np.random.default_rng(12)
    a = rng.normal(loc=5.0, scale=1.0, size=300)
    b = rng.normal(loc=2.0, scale=1.0, size=300)
    stats = paired_bootstrap_stats(a, b, bootstrap_reps=300, seed=1)
    assert stats["mean_difference"] == pytest.approx(3.0, abs=0.5)
    assert stats["exceeds_zero"] is True
    assert stats["ci95_difference"][0] > 0


def test_paired_bootstrap_stats_does_not_exclude_zero_for_noise():
    rng = np.random.default_rng(13)
    a = rng.normal(loc=0.0, scale=1.0, size=20)
    b = rng.normal(loc=0.0, scale=1.0, size=20)
    stats = paired_bootstrap_stats(a, b, bootstrap_reps=300, seed=2)
    assert stats["exceeds_zero"] is False


def test_amplitude_error_summary_bias_and_rmse():
    truth = np.full(50, 3.0)
    estimate = truth + 0.5
    out = amplitude_error_summary(estimate, truth)
    assert out["bias"] == pytest.approx(0.5)
    assert out["rmse"] == pytest.approx(0.5)


def test_evaluate_trigger_efficiency_end_to_end():
    rng = np.random.default_rng(8)
    n_features = 32
    background = rng.normal(scale=0.1, size=(400, n_features))
    template = np.zeros(n_features)
    template[10:20] = 1.0
    metric = np.eye(n_features)
    components = template[None, :] / np.linalg.norm(template)
    mean = np.zeros(n_features)

    out = evaluate_trigger_efficiency(
        background, template, components, mean, metric,
        fpr_points=[0.01, 0.05], amplitude_grid=np.array([0.0, 0.5, 1.0, 2.0]),
        n_injections=100, bootstrap_reps=50, seed=8,
    )
    assert set(out.keys()) >= {
        "amplitude_grid", "fpr_points", "background_thresholds",
        "direct_injection", "measured_psd_injection",
        "measured_psd_minus_direct_50pct_amplitude_threshold",
    }
    eff_001 = out["direct_injection"]["efficiency"][0.01]
    assert eff_001[0] <= eff_001[-1]  # efficiency should not decrease with amplitude
    assert eff_001[-1] > eff_001[0]  # and should clearly rise by the top of the grid


# --------------------------------------------------------------------------- #
# negative_controls.py
# --------------------------------------------------------------------------- #
def test_random_weighted_orthonormal_basis_is_orthonormal():
    rng = np.random.default_rng(9)
    n_features = 20
    metric = np.diag(rng.uniform(0.5, 2.0, size=n_features))
    basis = random_weighted_orthonormal_basis(4, n_features, metric, rng)
    gram = basis @ metric @ basis.T
    np.testing.assert_allclose(gram, np.eye(4), atol=1e-8)


def _structured_pulse_dataset(rng, n_train=300, n_test=150, n_features=64, n_noise=200):
    t = np.linspace(0.0, 1.0, n_features)
    base = np.exp(-((t - 0.4) ** 2) / 0.004)
    cov = np.eye(n_features) * 0.02
    for i in range(n_features - 1):
        cov[i, i + 1] = cov[i + 1, i] = 0.01
    noise_train = rng.multivariate_normal(np.zeros(n_features), cov, size=n_noise)

    def make(n):
        amp = rng.uniform(0.5, 1.5, size=n)
        pulses = amp[:, None] * base[None, :]
        pulses += rng.multivariate_normal(np.zeros(n_features), cov, size=n)
        return pulses

    return make(n_train), make(n_test), noise_train, base


def test_compute_negative_control_residuals_all_degrade_on_structured_data():
    rng = np.random.default_rng(10)
    train, test, noise_train, base = _structured_pulse_dataset(rng)
    covariance = regularize_covariance(estimate_covariance(noise_train), shrinkage=0.1)
    metric = inverse_covariance(covariance)
    rank = 3
    empca = fit_weighted_pca(train, metric, rank)
    empca_recon = project_onto_basis(test, empca.components, weights=metric, mean=empca.mean)
    nominal_empca = weighted_residual(test, empca_recon, metric)

    template = base / np.max(np.abs(base))
    denom = float(template @ metric @ template)
    amp = (test @ metric @ template) / denom
    of_recon = amp[:, None] * template[None, :]
    nominal_of = weighted_residual(test, of_recon, metric)

    control_residuals = compute_negative_control_residuals(
        train, test, noise_train, metric, rank, template, seed=11,
    )
    assert set(control_residuals.keys()) == {
        "wrong_psd", "permuted_time_bins", "random_weighted_basis",
        "noise_only_train", "mismatched_of_template",
    }
    for name, control_residual in control_residuals.items():
        nominal = nominal_empca if CONTROL_NOMINAL[name] == "empca" else nominal_of
        stats = paired_bootstrap_stats(control_residual, nominal, bootstrap_reps=300, seed=1)
        assert stats["exceeds_zero"], f"{name} did not degrade: {stats}"


def test_compute_negative_control_residuals_respects_controls_filter():
    rng = np.random.default_rng(14)
    train, test, noise_train, base = _structured_pulse_dataset(rng, n_train=100, n_test=50, n_noise=80)
    covariance = regularize_covariance(estimate_covariance(noise_train), shrinkage=0.1)
    metric = inverse_covariance(covariance)
    template = base / np.max(np.abs(base))
    out = compute_negative_control_residuals(
        train, test, noise_train, metric, 3, template, seed=1, controls=["wrong_psd"],
    )
    assert set(out.keys()) == {"wrong_psd"}
