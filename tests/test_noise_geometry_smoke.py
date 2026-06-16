import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.noise import estimate_covariance, inverse_psd_weights, regularize_covariance
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


def test_rank1_synthetic_smoke():
    dataset = make_rank1_pulse_dataset(n_traces=64, n_samples=256, seed=12)
    summary = run_of_empca_equivalence(dataset)
    assert summary["of_truth_corr"] > 0.9
    assert summary["weighted_angle_deg"] < 30.0
