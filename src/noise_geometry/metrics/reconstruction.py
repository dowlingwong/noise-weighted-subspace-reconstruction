"""Scalar reconstruction and likelihood-geometry metrics."""

from __future__ import annotations

import numpy as np


def mse(x: np.ndarray, xhat: np.ndarray, *, axis=None):
    """Mean squared residual."""
    return np.mean(np.abs(np.asarray(x) - np.asarray(xhat)) ** 2, axis=axis)


def weighted_inner(u: np.ndarray, v: np.ndarray, metric: np.ndarray) -> complex:
    """Inner product ``u^H Sigma^{-1} v`` for diagonal or full metrics."""
    a = np.asarray(u)
    b = np.asarray(v)
    m = np.asarray(metric, dtype=np.float64)
    if m.ndim == 1:
        return np.sum(np.conj(a) * b * m)
    if m.ndim == 2:
        return np.conj(a) @ m @ b
    raise ValueError("metric must be a 1D weight vector or square matrix")


def weighted_norm(x: np.ndarray, metric: np.ndarray) -> float:
    """Return ``x^H Sigma^{-1} x``."""
    return float(np.real(weighted_inner(x, x, metric)))


def weighted_residual(
    x: np.ndarray,
    xhat: np.ndarray,
    weights: np.ndarray,
    *,
    axis: int = -1,
    normalize: bool = True,
):
    """Inverse-noise weighted residual for diagonal weights or full covariance.

    With unit diagonal weights or an identity matrix and ``normalize=True``,
    this reduces to ordinary MSE along the feature axis.
    """
    residual = np.asarray(x) - np.asarray(xhat)
    metric = np.asarray(weights, dtype=np.float64)
    if metric.ndim == 1:
        r2 = np.abs(residual) ** 2
        value = np.sum(r2 * metric, axis=axis)
        if normalize:
            value = value / np.maximum(np.sum(metric), np.finfo(float).eps)
        return value
    if metric.ndim == 2:
        if residual.shape[-1] != metric.shape[0] or metric.shape[0] != metric.shape[1]:
            raise ValueError("full metric shape must match the last residual dimension")
        value = np.einsum("...i,ij,...j->...", np.conj(residual), metric, residual).real
        if normalize:
            value = value / residual.shape[-1]
        return value
    raise ValueError("weights must be a 1D vector or 2D square matrix")


def whitened_mse(x: np.ndarray, xhat: np.ndarray, weights: np.ndarray, *, axis=-1):
    """MSE after diagonal whitening by ``sqrt(weights)``."""
    return weighted_residual(x, xhat, weights, axis=axis)


def amplitude_bias(estimate: np.ndarray, truth: np.ndarray):
    """Signed mean amplitude bias."""
    return float(np.mean(np.asarray(estimate) - np.asarray(truth)))


def gaussian_nll(
    x: np.ndarray,
    xhat: np.ndarray,
    sigma_inv: np.ndarray,
    *,
    logdet_cov: float | None = None,
    include_constant: bool = False,
) -> np.ndarray:
    """Gaussian negative log likelihood up to optional constants."""
    chi2 = weighted_residual(x, xhat, sigma_inv, normalize=False)
    nll = 0.5 * chi2
    if logdet_cov is not None:
        nll = nll + 0.5 * float(logdet_cov)
    if include_constant:
        n_features = np.asarray(x).shape[-1]
        nll = nll + 0.5 * n_features * np.log(2.0 * np.pi)
    return nll


def amplitude_resolution(estimate: np.ndarray, truth: np.ndarray | None = None) -> dict[str, float]:
    """Return amplitude bias, standard deviation, RMSE, and correlation."""
    est = np.asarray(estimate, dtype=np.float64)
    if truth is None:
        return {"std": float(np.std(est, ddof=1))}
    true = np.asarray(truth, dtype=np.float64)
    err = est - true
    corr = float(np.corrcoef(est, true)[0, 1]) if est.size > 1 else float("nan")
    return {
        "bias": float(np.mean(err)),
        "std": float(np.std(err, ddof=1)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "corr": corr,
    }


def residual_autocorrelation(residuals: np.ndarray, *, max_lag: int = 32) -> np.ndarray:
    """Mean normalized autocorrelation of row-wise residuals."""
    R = np.atleast_2d(np.asarray(residuals, dtype=np.float64))
    R = R - R.mean(axis=1, keepdims=True)
    denom = np.sum(R**2, axis=1)
    out = []
    for lag in range(max_lag + 1):
        if lag == 0:
            num = denom
        else:
            num = np.sum(R[:, :-lag] * R[:, lag:], axis=1)
        out.append(float(np.mean(num / np.maximum(denom, np.finfo(float).eps))))
    return np.asarray(out)


def residual_psd(residuals: np.ndarray, sampling_frequency: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning the one-sided residual PSD."""
    from ..noise import estimate_psd_rfft

    return estimate_psd_rfft(residuals, sampling_frequency)
