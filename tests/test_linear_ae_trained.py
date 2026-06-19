"""S3 bridge via *trained* tied linear autoencoders.

These tests verify that an independently gradient-trained weighted tied linear
AE converges to the EMPCA subspace, rather than delegating to
``fit_weighted_pca``. Evidence is the optimality gap against the EMPCA global
optimum, the principal angle in the M-metric, and M-orthonormality of the
learned basis. Both the direct weighted-loss path and the whitened MSE
cross-check are covered, on diagonal and full-covariance noise.
"""

import numpy as np
import pytest

from src.noise_geometry.subspace import principal_angles
from src.noise_geometry.autoencoders.trained import (
    _as_metric,
    _loss_and_grad,
    empca_optimal_loss,
    train_weighted_linear_ae,
    train_whitened_linear_ae,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _make_diagonal_problem(seed=31, n=400, d=12, k=3):
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(d, k)))[0].T
    coeffs = rng.normal(scale=[3.0, 2.0, 1.0][:k], size=(n, k))
    samples = coeffs @ basis + rng.normal(scale=0.05, size=(n, d))
    weights = np.linspace(0.5, 3.0, d)
    return samples, weights, k


def _make_full_cov_problem(seed=32, n=600, d=8, k=3):
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(d, k)))[0].T
    coeffs = rng.normal(scale=[3.0, 2.0, 1.0][:k], size=(n, k))
    mixing = rng.normal(size=(d, d))
    covariance = mixing @ mixing.T + 0.5 * np.eye(d)
    noise = rng.multivariate_normal(np.zeros(d), 0.01 * covariance, size=n)
    samples = coeffs @ basis + noise
    metric = np.linalg.inv(covariance)
    return samples, metric, k


# --------------------------------------------------------------------------- #
# Gradient correctness (guards the analytic gradient used by L-BFGS)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("problem", [_make_diagonal_problem, _make_full_cov_problem])
def test_analytic_gradient_matches_numerical(problem):
    """Central-difference check, relative to the gradient norm.

    ``scipy.optimize.check_grad`` reports *absolute* forward-difference error,
    which is misleading when the gradient norm is large (here ~7e3). A central
    difference with a relative tolerance is the correct test.
    """
    samples, weights, k = problem()
    w = _as_metric(weights)
    X = samples - samples.mean(axis=0)
    d = X.shape[1]
    rng = np.random.default_rng(7)
    v = rng.standard_normal(d * k)

    analytic = _loss_and_grad(v.reshape(d, k), X, w)[1].ravel()
    h = 1e-6
    numerical = np.empty_like(v)
    for i in range(v.size):
        e = np.zeros_like(v)
        e[i] = h
        fp = _loss_and_grad((v + e).reshape(d, k), X, w)[0]
        fm = _loss_and_grad((v - e).reshape(d, k), X, w)[0]
        numerical[i] = (fp - fm) / (2 * h)

    rel_err = np.max(np.abs(analytic - numerical)) / np.linalg.norm(analytic)
    assert rel_err < 1e-6, f"relative gradient mismatch {rel_err:e}"


# --------------------------------------------------------------------------- #
# L-BFGS: high-precision bridge to EMPCA
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "trainer", [train_weighted_linear_ae, train_whitened_linear_ae]
)
@pytest.mark.parametrize("problem", [_make_diagonal_problem, _make_full_cov_problem])
def test_lbfgs_trained_ae_matches_empca(trainer, problem):
    samples, weights, k = problem()
    res = trainer(samples, weights, k, optimizer="lbfgs", seed=0)

    # Converged to the EMPCA global optimum of the same objective.
    assert res.relative_gap >= -1e-9            # never beats the optimum
    assert res.relative_gap < 1e-6, res.relative_gap
    # Same subspace as fit_weighted_pca, in the noise metric.
    assert res.max_principal_angle_deg < 1e-2, res.max_principal_angle_deg
    # Learned basis is M-orthonormal at the optimum (the tied-AE property).
    assert res.m_orthonormality_error < 1e-4, res.m_orthonormality_error


def test_direct_and_whitened_agree():
    samples, weights, k = _make_full_cov_problem()
    direct = train_weighted_linear_ae(samples, weights, k, optimizer="lbfgs", seed=0)
    whitened = train_whitened_linear_ae(samples, weights, k, optimizer="lbfgs", seed=0)
    angle = float(
        np.max(principal_angles(direct.components, whitened.components, weights=_as_metric(weights)))
    )
    assert angle < 1e-2, angle


# --------------------------------------------------------------------------- #
# Adam: reported, looser realistic-regime convergence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("problem", [_make_diagonal_problem, _make_full_cov_problem])
def test_adam_pass_converges_and_reports(problem):
    samples, weights, k = problem()
    res = train_weighted_linear_ae(samples, weights, k, optimizer="adam", seed=0, max_iter=8000)

    # Adam is reported, not held to the L-BFGS precision bar.
    assert res.loss_history is not None and res.loss_history.size > 1
    assert res.loss_history[-1] <= res.loss_history[0]      # loss decreased
    assert res.relative_gap < 1e-2, res.relative_gap
    assert res.max_principal_angle_deg < 1.0, res.max_principal_angle_deg


# --------------------------------------------------------------------------- #
# Sanity: L* is a genuine lower bound the trained AE approaches from above.
# --------------------------------------------------------------------------- #
def test_optimal_loss_is_lower_bound():
    samples, weights, k = _make_diagonal_problem()
    lstar, _ = empca_optimal_loss(samples, weights, k)
    res = train_weighted_linear_ae(samples, weights, k, optimizer="lbfgs", seed=0)
    assert res.final_loss >= lstar - 1e-9
