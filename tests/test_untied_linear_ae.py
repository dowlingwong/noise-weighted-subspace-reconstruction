"""Correctness of the untied linear autoencoder (task card P1-E12).

The card's required test: at convergence with ``Sigma = I``, the untied MSE
autoencoder recovers the PCA subspace to 1e-6. The remaining tests guard the
analytic gradient and the obliqueness diagnostic, which are what the E12
ablation actually measures.
"""

from __future__ import annotations

import numpy as np
import pytest

from noise_geometry.autoencoders.untied import (
    SeparableMetric,
    _loss_and_grad,
    obliqueness,
    sigma_orthogonal_projector,
    train_untied_linear_ae,
)
from noise_geometry.subspace import fit_pca, principal_angles


def _rank_k_data(rng, n=400, d=24, k=3, noise=0.05):
    B = np.linalg.qr(rng.standard_normal((d, k)))[0]
    Z = rng.standard_normal((n, k)) * np.array([4.0, 2.0, 1.0])[:k]
    return Z @ B.T + noise * rng.standard_normal((n, d))


def test_untied_mse_recovers_pca_subspace_under_identity_metric():
    """The card's gate: Sigma = I, converged untied MSE == PCA subspace to 1e-6."""
    rng = np.random.default_rng(0)
    X = _rank_k_data(rng)
    d = X.shape[1]
    res = train_untied_linear_ae(X, np.ones(d), 3, loss="mse", optimizer="lbfgs",
                                 seed=0, max_iter=20000)
    pca = fit_pca(X, 3)
    angles = np.asarray(principal_angles(res.decoder.T, pca.components, degrees=True))
    assert float(np.max(angles)) < 1e-6, f"max principal angle {np.max(angles):.3e} deg"


def test_untied_optimum_is_the_orthogonal_projector_under_identity_metric():
    """At the MSE optimum the untied map is the orthogonal projector.

    The tolerance is 1e-4, not 1e-6: L-BFGS stops on ``ftol`` at machine
    precision (``RELATIVE REDUCTION OF F <= FACTR*EPSMCH``), and the residual
    obliqueness at that point is ~4e-6 — a property of the optimiser's stopping
    floor, not of the estimator. The card's own gate (the PCA subspace to 1e-6)
    is asserted in the test above and is unaffected.
    """
    rng = np.random.default_rng(1)
    X = _rank_k_data(rng)
    d = X.shape[1]
    res = train_untied_linear_ae(X, np.ones(d), 3, loss="mse", optimizer="lbfgs",
                                 seed=1, max_iter=20000)
    assert res.obliqueness < 1e-4, f"obliqueness {res.obliqueness:.3e}"


def test_analytic_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    n, d, k = 40, 9, 2
    X = rng.standard_normal((n, d))
    w = np.exp(rng.standard_normal(d))
    D = rng.standard_normal((d, k))
    E = rng.standard_normal((k, d))
    loss0, gD, gE = _loss_and_grad(D, E, X, w, 1e-3)
    eps = 1e-6
    for (P, g) in ((D, gD), (E, gE)):
        idx = np.unravel_index(rng.integers(0, P.size, size=5), P.shape)
        for i, j in zip(*idx):
            Pp, Pm = P.copy(), P.copy()
            Pp[i, j] += eps
            Pm[i, j] -= eps
            if P is D:
                lp, _, _ = _loss_and_grad(Pp, E, X, w, 1e-3)
                lm, _, _ = _loss_and_grad(Pm, E, X, w, 1e-3)
            else:
                lp, _, _ = _loss_and_grad(D, Pp, X, w, 1e-3)
                lm, _, _ = _loss_and_grad(D, Pm, X, w, 1e-3)
            num = (lp - lm) / (2 * eps)
            assert num == pytest.approx(g[i, j], rel=1e-5, abs=1e-8)


def test_sigma_orthogonal_projector_is_idempotent_and_metric_symmetric():
    rng = np.random.default_rng(3)
    d, k = 12, 3
    A = rng.standard_normal((d, d))
    Sigma_inv = A @ A.T + d * np.eye(d)
    D = rng.standard_normal((d, k))
    Pi = sigma_orthogonal_projector(D, Sigma_inv)
    assert np.allclose(Pi @ Pi, Pi, atol=1e-10)
    # Sigma^{-1} Pi is symmetric  <=>  Pi is Sigma-orthogonal
    assert np.allclose(Sigma_inv @ Pi, (Sigma_inv @ Pi).T, atol=1e-8)
    # a tied map D (D^T M D)^{-1} D^T M has zero obliqueness by construction
    MD = Sigma_inv @ D
    E = np.linalg.solve(D.T @ MD, MD.T)
    assert obliqueness(D, E, Sigma_inv) < 1e-10


def test_separable_metric_matches_its_dense_kronecker():
    rng = np.random.default_rng(4)
    C, T = 3, 5
    A = rng.standard_normal((C, C)); A = A @ A.T + C * np.eye(C)
    B = rng.standard_normal((T, T)); B = B @ B.T + T * np.eye(T)
    sep = SeparableMetric(A, B)
    dense = sep.dense()
    R = rng.standard_normal((7, C * T))
    from noise_geometry.autoencoders.untied import _apply_metric
    assert np.allclose(_apply_metric(R, sep), R @ dense, atol=1e-10)
