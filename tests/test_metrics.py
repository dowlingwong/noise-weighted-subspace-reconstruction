import numpy as np

from src.noise_geometry.metrics import mse, weighted_residual
from src.noise_geometry.experiments.synthetic import run_s9_multichannel


def test_identity_weighting_reduces_to_mse():
    x = np.array([[1.0, 2.0], [3.0, -1.0]])
    xhat = np.array([[0.5, 1.5], [2.0, 0.0]])
    expected = mse(x, xhat, axis=-1)
    np.testing.assert_allclose(weighted_residual(x, xhat, np.ones(2)), expected)
    np.testing.assert_allclose(weighted_residual(x, xhat, np.eye(2)), expected)


def test_colored_metric_changes_residual_geometry():
    x = np.array([[1.0, 1.0]])
    xhat = np.zeros_like(x)
    raw = mse(x, xhat, axis=-1)
    weighted = weighted_residual(x, xhat, np.diag([0.1, 10.0]))
    assert not np.allclose(raw, weighted)


def test_multichannel_full_covariance_reduces_to_diagonal_when_uncorrelated():
    metrics = run_s9_multichannel(
        {
            "seed": 12,
            "n_channels": 3,
            "n_features": 12,
            "n_traces": 500,
            "correlation_strength": 0.0,
        }
    )
    assert abs(metrics["diagonal_over_full_sigma"] - 1.0) < 1e-10
