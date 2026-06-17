"""Covariance and PSD-domain optimal-filter helpers."""

from __future__ import annotations

import numpy as np


def weighted_inner(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> complex:
    """Return ``sum(conj(a) * b * w)``."""
    return np.sum(np.conj(a) * b * w)


def gls_amplitude(X_f: np.ndarray, template_f: np.ndarray, weights: np.ndarray, *, return_complex: bool = False):
    """Estimate OF/GLS amplitudes in the weighted frequency-domain inner product."""
    X_arr = np.asarray(X_f)
    single = X_arr.ndim == 1
    X2 = np.atleast_2d(X_arr)
    s = np.asarray(template_f)
    w = np.asarray(weights, dtype=np.float64)
    denom = np.real(weighted_inner(s, s, w))
    if denom <= 0:
        raise ValueError("template has zero weighted norm")
    amps = np.sum(np.conj(s)[None, :] * X2 * w[None, :], axis=1) / denom
    if not return_complex:
        amps = np.real(amps)
    return amps[0] if single else amps


def project_rank1(X_f: np.ndarray, template_f: np.ndarray, weights: np.ndarray, *, return_amp: bool = False):
    """Project traces onto a fixed rank-1 template under the inverse-noise metric."""
    X_arr = np.asarray(X_f)
    single = X_arr.ndim == 1
    X2 = np.atleast_2d(X_arr)
    amps = np.asarray(gls_amplitude(X2, template_f, weights))
    recon = amps[:, None] * np.asarray(template_f)[None, :]
    if single:
        recon = recon[0]
        amps = float(amps[0])
    return (recon, amps) if return_amp else recon


def matched_filter_score(X_f: np.ndarray, template_f: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return normalized matched-filter scores for one or more traces."""
    X_arr = np.atleast_2d(np.asarray(X_f))
    s = np.asarray(template_f)
    w = np.asarray(weights, dtype=np.float64)
    denom = np.sqrt(np.real(weighted_inner(s, s, w)))
    if denom <= 0:
        raise ValueError("template has zero weighted norm")
    score = np.real(np.sum(np.conj(s)[None, :] * X_arr * w[None, :], axis=1)) / denom
    return score[0] if np.asarray(X_f).ndim == 1 else score


def psd_amplitude_variance(
    template_f: np.ndarray,
    psd: np.ndarray,
    weights: np.ndarray,
    *,
    trace_len: int,
    sampling_frequency: float,
) -> float:
    """Predict OF amplitude variance for this repository's rFFT convention.

    This includes the FFT and one-sided-PSD normalization used by
    ``generate_colored_noise`` rather than assuming dimensionless unit noise.
    """
    s = np.asarray(template_f)
    J = np.asarray(psd, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if s.shape != J.shape or s.shape != w.shape:
        raise ValueError("template_f, psd, and weights must have matching shapes")
    den = np.real(weighted_inner(s, s, w))
    if den <= 0:
        raise ValueError("template has zero weighted norm")

    scale = float(sampling_frequency) * int(trace_len)
    variance_num = 0.0
    if s.size > 1:
        stop = -1 if trace_len % 2 == 0 else None
        variance_num += float(np.sum((w[1:stop] ** 2) * J[1:stop] * scale * np.abs(s[1:stop]) ** 2 / 4.0))
    if trace_len % 2 == 0 and s.size > 1:
        variance_num += float(w[-1] ** 2 * J[-1] * scale * np.real(s[-1]) ** 2)
    if w[0] != 0:
        variance_num += float(w[0] ** 2 * J[0] * scale / 2.0 * np.real(s[0]) ** 2)
    return variance_num / den**2
