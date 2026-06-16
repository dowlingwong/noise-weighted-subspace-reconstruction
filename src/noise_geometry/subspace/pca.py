"""Small deterministic PCA and weighted-PCA helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SubspaceFit:
    """Container for a fitted linear subspace."""

    components: np.ndarray
    mean: np.ndarray
    explained_variance: np.ndarray
    weights: np.ndarray | None = None


def fit_pca(samples: np.ndarray, rank: int, *, center: bool = True) -> SubspaceFit:
    """Fit an ordinary Euclidean PCA basis with row-wise observations."""
    X = np.asarray(samples, dtype=np.float64)
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1], dtype=np.float64)
    Xc = X - mean[None, :]
    _, s, vh = np.linalg.svd(Xc, full_matrices=False)
    r = int(rank)
    return SubspaceFit(vh[:r].copy(), mean, (s[:r] ** 2) / max(X.shape[0] - 1, 1), None)


def fit_weighted_pca(samples: np.ndarray, weights: np.ndarray, rank: int, *, center: bool = True) -> SubspaceFit:
    """Fit PCA in whitened coordinates and map components back to data units.

    This is the diagonal-covariance version of weighted PCA/EMPCA and is the
    closed-form target for the tied linear AE bridge experiment.
    """
    X = np.asarray(samples, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if X.shape[1] != w.shape[0]:
        raise ValueError("weights length must match feature dimension")
    sqrt_w = np.sqrt(np.clip(w, 0.0, None))
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1], dtype=np.float64)
    Xw = (X - mean[None, :]) * sqrt_w[None, :]
    _, s, vh = np.linalg.svd(Xw, full_matrices=False)
    r = int(rank)
    components = np.zeros((r, X.shape[1]), dtype=np.float64)
    mask = sqrt_w > 0
    components[:, mask] = vh[:r, mask] / sqrt_w[mask]
    return SubspaceFit(components, mean, (s[:r] ** 2) / max(X.shape[0] - 1, 1), w)


def project_onto_basis(samples: np.ndarray, basis: np.ndarray, *, weights: np.ndarray | None = None, mean=None) -> np.ndarray:
    """Reconstruct row-wise samples from a basis using LS or weighted LS."""
    X = np.asarray(samples, dtype=np.float64)
    B = np.asarray(basis, dtype=np.float64)
    mu = np.zeros(X.shape[1], dtype=np.float64) if mean is None else np.asarray(mean, dtype=np.float64)
    Xc = X - mu[None, :]
    if weights is None:
        gram = B @ B.T
        rhs = Xc @ B.T
    else:
        w = np.asarray(weights, dtype=np.float64)
        gram = (B * w[None, :]) @ B.T
        rhs = (Xc * w[None, :]) @ B.T
    coeff = np.linalg.solve(gram, rhs.T).T
    return coeff @ B + mu[None, :]
