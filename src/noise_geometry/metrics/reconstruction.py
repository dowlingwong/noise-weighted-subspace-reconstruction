"""Scalar reconstruction metrics."""

from __future__ import annotations

import numpy as np


def mse(x: np.ndarray, xhat: np.ndarray, *, axis=None):
    """Mean squared residual."""
    return np.mean(np.abs(np.asarray(x) - np.asarray(xhat)) ** 2, axis=axis)


def weighted_residual(x: np.ndarray, xhat: np.ndarray, weights: np.ndarray, *, axis=-1):
    """Mean inverse-noise weighted residual along ``axis``."""
    r2 = np.abs(np.asarray(x) - np.asarray(xhat)) ** 2
    w = np.asarray(weights, dtype=np.float64)
    return np.sum(r2 * w, axis=axis) / np.maximum(np.sum(w), np.finfo(float).eps)


def whitened_mse(x: np.ndarray, xhat: np.ndarray, weights: np.ndarray, *, axis=-1):
    """MSE after diagonal whitening by ``sqrt(weights)``."""
    return weighted_residual(x, xhat, weights, axis=axis)


def amplitude_bias(estimate: np.ndarray, truth: np.ndarray):
    """Signed mean amplitude bias."""
    return float(np.mean(np.asarray(estimate) - np.asarray(truth)))
