"""Whitening transforms for covariance and PSD representations."""

from __future__ import annotations

import numpy as np

from .covariance import regularize_covariance
from .psd import regularize_psd


def _covariance_factors(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    C = regularize_covariance(cov)
    vals, vecs = np.linalg.eigh(C)
    return vals, vecs


def whiten_with_covariance(samples: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Apply ``Sigma^{-1/2}`` to row-wise samples."""
    X = np.asarray(samples, dtype=np.float64)
    vals, vecs = _covariance_factors(cov)
    return (X @ vecs) / np.sqrt(vals)[None, :]


def unwhiten_with_covariance(samples: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Apply ``Sigma^{1/2}`` to row-wise samples whitened by this module."""
    X = np.asarray(samples, dtype=np.float64)
    vals, vecs = _covariance_factors(cov)
    return (X * np.sqrt(vals)[None, :]) @ vecs.T


def whiten_rfft(X_f: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Whiten rFFT-domain traces by dividing by ``sqrt(PSD)`` bin-wise."""
    J = regularize_psd(psd)
    return np.asarray(X_f) / np.sqrt(J)[None, :]
