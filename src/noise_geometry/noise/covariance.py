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


def covariance_from_psd(psd_one_sided: np.ndarray, n_samples: int, *, sampling_frequency: float = 1.0) -> np.ndarray:
    """Stationary (circulant) covariance implied by a one-sided PSD estimate.

    Uses the same one-sided convention as ``estimate_psd_rfft``/
    ``estimate_psd_ensemble`` (interior rFFT bins carry twice the power of
    DC/Nyquist). This is the periodic/circular-boundary approximation the
    rest of this codebase's rFFT-based OF/whitening machinery already
    assumes, so it is not a new modeling choice, just an explicit one.

    Where the observation-space sample covariance (``estimate_covariance``)
    has ``n_samples * (n_samples + 1) / 2`` free parameters, this has only
    ``n_samples // 2 + 1`` (one per PSD bin) -- the right regime when the
    calibration sample size is comparable to or smaller than the trace
    length (N ~ p), where the raw sample covariance is ill-conditioned.
    """
    psd = np.asarray(psd_one_sided, dtype=np.float64)
    n = int(n_samples)
    n_bins = n // 2 + 1
    if psd.shape[0] != n_bins:
        raise ValueError(f"psd length {psd.shape[0]} does not match rfft bin count {n_bins} for n_samples={n}")
    undoubled = psd.copy()
    if n > 2:
        stop = n // 2 + 1 - (n + 1) % 2  # interior bins, excluding DC and (if even) Nyquist
        undoubled[1:stop] = undoubled[1:stop] / 2.0
    autocovariance = np.fft.irfft(undoubled * float(sampling_frequency), n=n)
    return _circulant(autocovariance)


def _circulant(first_column: np.ndarray) -> np.ndarray:
    c = np.asarray(first_column, dtype=np.float64)
    n = c.shape[0]
    idx = (np.arange(n)[:, None] - np.arange(n)[None, :]) % n
    return c[idx]


def block_covariance(channel_cov: np.ndarray, time_cov: np.ndarray) -> np.ndarray:
    """Build a separable multichannel covariance ``channel_cov kron time_cov``."""
    Cc = np.asarray(channel_cov, dtype=np.float64)
    Ct = np.asarray(time_cov, dtype=np.float64)
    if Cc.ndim != 2 or Cc.shape[0] != Cc.shape[1]:
        raise ValueError("channel_cov must be square")
    if Ct.ndim != 2 or Ct.shape[0] != Ct.shape[1]:
        raise ValueError("time_cov must be square")
    return np.kron(Cc, Ct)
