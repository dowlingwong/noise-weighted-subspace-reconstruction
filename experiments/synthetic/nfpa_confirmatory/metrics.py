"""Per-seed evaluation metrics for every fitted method (protocol sections 8-9).

Metrics are evaluated in the oracle generating covariance: raw MSE in
measurement space and the per-coordinate normalized whitened residual
(protocol equation for ``normalized_chi2``). Subspace comparisons use the
common oracle-whitened coordinate system.
"""

from __future__ import annotations

import numpy as np

from noise_geometry.metrics import mse
from noise_geometry.subspace import principal_angles

from .covariance import whiten_separable
from .generator import Dataset
from .methods import MethodFit, basis_whitened, reconstruct


def raw_mse(X: np.ndarray, Xhat: np.ndarray) -> float:
    return float(mse(X, Xhat))


def normalized_chi2(ds: Dataset, X: np.ndarray, Xhat: np.ndarray) -> float:
    Z = whiten_separable(X - ds.train_mean, ds.covariances["Sigma_c"], ds.covariances["Sigma_t"])
    Zhat = whiten_separable(Xhat - ds.train_mean, ds.covariances["Sigma_c"], ds.covariances["Sigma_t"])
    return float(np.mean((Z - Zhat) ** 2))


def subspace_angles_to(fit: MethodFit, reference: MethodFit, ds: Dataset) -> np.ndarray:
    A = basis_whitened(fit, ds)
    B = basis_whitened(reference, ds)
    return np.asarray(principal_angles(A.T, B.T, degrees=True))


def evaluate_fit(ds: Dataset, fit: MethodFit, cw_pca: MethodFit, X: np.ndarray) -> dict[str, float]:
    """Return the full per-seed metric vector for a method on split ``X``."""
    Xhat = reconstruct(fit, ds, X)
    angles = subspace_angles_to(fit, cw_pca, ds)
    return {
        "raw_mse": raw_mse(X, Xhat),
        "normalized_chi2": normalized_chi2(ds, X, Xhat),
        "angle_to_cw_pca_mean_deg": float(np.mean(angles)),
        "angle_to_cw_pca_max_deg": float(np.max(angles)),
    }
