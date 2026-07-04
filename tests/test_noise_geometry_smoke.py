import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.noise import (
    covariance_from_psd,
    estimate_covariance,
    estimate_psd_ensemble,
    inverse_psd_weights,
    regularize_covariance,
)
from noise_geometry.noise.whitening import unwhiten_with_covariance, whiten_with_covariance
from noise_geometry.synthetic import make_rank1_pulse_dataset, run_of_empca_equivalence


def test_covariance_whitening_round_trip():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(64, 8))
    cov = regularize_covariance(estimate_covariance(X), shrinkage=0.05)
    Xw = whiten_with_covariance(X, cov)
    Xrt = unwhiten_with_covariance(Xw, cov)
    np.testing.assert_allclose(Xrt, X, atol=1e-10)


def test_inverse_psd_weights_are_finite_and_zero_dc():
    psd = np.linspace(0.0, 2.0, 9)
    weights = inverse_psd_weights(psd, trace_len=16)
    assert weights[0] == 0.0
    assert np.all(np.isfinite(weights))
    assert np.all(weights[1:] > 0.0)


def test_covariance_from_psd_round_trips_a_known_circulant_covariance():
    """Deterministic check: for a circulant covariance, extracting its DFT
    eigenvalues, re-expressing them in the one-sided-PSD convention (interior
    bins doubled), and calling covariance_from_psd must recover the original
    matrix -- i.e. the doubling/undoubling and irfft steps are exact inverses.
    """
    n = 16
    rng = np.random.default_rng(0)
    # A simple positive first column (decaying) makes a valid PSD circulant.
    first_column = np.exp(-np.arange(n) * 0.3) + 0.01 * rng.standard_normal(n)
    first_column = 0.5 * (first_column + first_column[(-np.arange(n)) % n])  # enforce even symmetry
    idx = (np.arange(n)[:, None] - np.arange(n)[None, :]) % n
    true_cov = first_column[idx]

    eigs = np.fft.fft(true_cov[:, 0]).real
    n_bins = n // 2 + 1
    psd_one_sided = eigs[:n_bins].copy()
    stop = n // 2 + 1 - (n + 1) % 2
    psd_one_sided[1:stop] *= 2.0  # matches covariance_from_psd's "undouble" convention

    recovered = covariance_from_psd(psd_one_sided, n)
    np.testing.assert_allclose(recovered, true_cov, atol=1e-8)


def test_covariance_from_psd_recovers_white_noise_variance():
    """Statistical check: averaging estimate_psd_ensemble over many white-noise
    draws and feeding it through covariance_from_psd should recover
    approximately sigma^2 * I (diagonal dominant, off-diagonal near zero).
    """
    rng = np.random.default_rng(1)
    n_samples = 32
    sigma2 = 0.5
    noise = rng.normal(scale=np.sqrt(sigma2), size=(4000, n_samples))
    _, psd = estimate_psd_ensemble(noise, average="median")
    cov = covariance_from_psd(psd, n_samples)
    np.testing.assert_allclose(np.diag(cov), sigma2, rtol=0.15)
    off_diag_rms = np.sqrt(np.mean((cov - np.diag(np.diag(cov))) ** 2))
    assert off_diag_rms < 0.1 * sigma2


def test_rank1_synthetic_smoke():
    dataset = make_rank1_pulse_dataset(n_traces=64, n_samples=256, seed=12)
    summary = run_of_empca_equivalence(dataset)
    assert summary["of_truth_corr"] > 0.9
    assert summary["weighted_angle_deg"] < 30.0
