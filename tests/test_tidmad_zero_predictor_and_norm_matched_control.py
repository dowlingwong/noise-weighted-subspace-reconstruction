"""Tests for the two P1-E13 additions to the TIDMAD stage.

1. ``run_metric_reversal`` now reports the zero predictor (amplitude 0 for every
   window) next to PCA and EMPCA, with the same paired bootstrap. Review
   Major 2: the archived real-data headline compares two estimators that are
   both *below* that floor, so the floor has to travel with the comparison.
2. ``run_negative_controls`` gains ``random_weighted_basis_norm_matched``: a
   random Sigma-orthonormal basis whose reconstruction is rescaled so its
   whitened energy matches the fitted basis's mean whitened reconstruction
   energy on the fit windows. The plain control confounds direction with
   amplitude; this one isolates the direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.noise_geometry.noise import (
    generate_colored_noise,
    inverse_psd_weights,
    make_powerlaw_psd,
)
from src.noise_geometry.tidmad import (
    inject_sinusoid,
    nearest_bin_frequency,
    quadrature_amplitude_variance,
    quadrature_templates_rfft,
)
from src.noise_geometry.tidmad.analysis import build_window_set, run_metric_reversal
from src.noise_geometry.tidmad.negative_controls import CONTROLS, run_negative_controls

SEED = 20260906
N = 1024
FS = 1.0e6


def _prepared(snr_multiple: float):
    rng = np.random.default_rng(SEED)
    _, psd_truth = make_powerlaw_psd(N, FS, kind="pink")
    noise = generate_colored_noise(rng, psd_truth, N, FS, 256)
    calibration = generate_colored_noise(rng, psd_truth, N, FS, 256)
    frequency = nearest_bin_frequency(FS / 8.0, N, FS)
    weights = inverse_psd_weights(psd_truth, N)
    basis_f = quadrature_templates_rfft(N, FS, frequency)
    sigma = float(np.sqrt(quadrature_amplitude_variance(
        basis_f, psd_truth, weights, trace_len=N, sampling_frequency=FS)))
    observed, _ = inject_sinusoid(noise, FS, frequency, amplitude=snr_multiple * sigma, rng=rng)
    clean = observed - noise
    windows = build_window_set(observed, clean, psd_truth, weights, FS, frequency,
                              fit_fraction=0.7, guard_windows=2)
    return windows, calibration


@pytest.fixture(scope="module")
def high_snr():
    return _prepared(12.0)


def test_zero_predictor_is_reported_with_the_two_estimators(high_snr):
    windows, _ = high_snr
    result = run_metric_reversal(windows, rank=2, bootstrap_reps=100, seed=SEED)
    zp = result["zero_predictor"]
    for key in ("zero_predictor_amplitude_rmse", "pca_amplitude_rmse", "empca_amplitude_rmse"):
        assert np.isfinite(zp[key]) and zp[key] > 0.0
    # Well above the floor, EMPCA must beat the zero predictor, and the paired
    # interval must say so rather than the point estimate alone.
    assert zp["empca_beats_zero_predictor"]
    contrast = zp["paired_contrasts"]["zero_predictor_paired_difference_zero_minus_empca"]
    assert contrast["exceeds_zero"]
    assert contrast["mean_difference"] > 0.0


def test_zero_predictor_rmse_equals_the_rms_of_the_truth(high_snr):
    """The floor is a definition, not a fit: |error| of a zero estimate is |truth|."""
    from src.noise_geometry.tidmad.analysis import _truth_amplitude

    windows, _ = high_snr
    result = run_metric_reversal(windows, rank=2, bootstrap_reps=50, seed=SEED)
    truth = _truth_amplitude(windows.evaluation_clean, windows.basis_rfft(), windows.weights)
    assert result["zero_predictor"]["zero_predictor_amplitude_rmse"] == pytest.approx(
        float(np.sqrt(np.mean(np.asarray(truth) ** 2))), rel=1e-12)


def test_norm_matched_control_is_registered_and_degrades(high_snr):
    windows, calibration = high_snr
    result = run_negative_controls(windows, calibration, rank=2, bootstrap_reps=100, seed=SEED)
    assert "random_weighted_basis_norm_matched" in CONTROLS
    assert "random_weighted_basis_norm_matched" in result["degraded"]
    assert result["degraded"]["random_weighted_basis_norm_matched"]


def test_norm_matching_actually_rescales_and_is_recorded(high_snr):
    """The scale must be a real, reported number, not silently 1."""
    windows, calibration = high_snr
    result = run_negative_controls(windows, calibration, rank=2, bootstrap_reps=50, seed=SEED)
    scale = result["norm_match_scale"]
    assert np.isfinite(scale) and scale > 0.0
    assert result["norm_match_energy_fitted"] > 0.0
    assert result["norm_match_energy_random"] > 0.0
    assert scale == pytest.approx(
        np.sqrt(result["norm_match_energy_fitted"] / result["norm_match_energy_random"]), rel=1e-12)
    # The unmatched control is left untouched by the addition.
    assert "random_weighted_basis" in result["control_amplitude_rmse"]
