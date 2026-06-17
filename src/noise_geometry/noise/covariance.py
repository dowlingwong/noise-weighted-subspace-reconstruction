"""Covariance estimation and regularization."""

from __future__ import annotations

import numpy as np


def estimate_covariance(samples: np.ndarray, *, center: bool = True) -> np.ndarray:
    """Estimate an observation-space covariance from row-wise samples.

    Parameters
    ----------
    samples:
        Array with shape ``(n_observations, n_features)``.
    center:
        If true, subtract the sample mean before estimating covariance.
    """
    X = np.asarray(samples, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("samples must have shape (n_observations, n_features)")
    if X.shape[0] < 2:
        raise ValueError("at least two observations are required")
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    return (X.T @ X) / (X.shape[0] - 1)


def regularize_covariance(cov: np.ndarray, *, floor: float | None = None, shrinkage: float = 0.0) -> np.ndarray:
    """Return a positive-definite covariance estimate with eigenvalue flooring.

    ``shrinkage`` mixes the covariance with its average-variance diagonal target.
    ``floor`` is an absolute lower bound on eigenvalues; if omitted, a small
    data-scaled floor is used.
    """
    C = np.asarray(cov, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("cov must be a square matrix")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be between 0 and 1")

    C = 0.5 * (C + C.T)
    if shrinkage:
        target = np.eye(C.shape[0]) * float(np.trace(C) / C.shape[0])
        C = (1.0 - shrinkage) * C + shrinkage * target

    vals, vecs = np.linalg.eigh(C)
    if floor is None:
        floor = max(float(np.max(vals)) * 1e-10, np.finfo(float).eps)
    vals = np.clip(vals, float(floor), None)
    return (vecs * vals[None, :]) @ vecs.T


def inverse_covariance(cov: np.ndarray, *, floor: float | None = None, shrinkage: float = 0.0) -> np.ndarray:
    """Return a stable inverse covariance matrix."""
    C = regularize_covariance(cov, floor=floor, shrinkage=shrinkage)
    vals, vecs = np.linalg.eigh(C)
    return (vecs * (1.0 / vals)[None, :]) @ vecs.T


def block_covariance(channel_cov: np.ndarray, time_cov: np.ndarray) -> np.ndarray:
    """Build a separable multichannel covariance ``channel_cov kron time_cov``."""
    Cc = np.asarray(channel_cov, dtype=np.float64)
    Ct = np.asarray(time_cov, dtype=np.float64)
    if Cc.ndim != 2 or Cc.shape[0] != Cc.shape[1]:
        raise ValueError("channel_cov must be square")
    if Ct.ndim != 2 or Ct.shape[0] != Ct.shape[1]:
        raise ValueError("time_cov must be square")
    return np.kron(Cc, Ct)
