"""One-sided PSD estimation and inverse-PSD weights."""

from __future__ import annotations

import numpy as np


def estimate_psd_rfft(traces: np.ndarray, sampling_frequency: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a one-sided PSD from time-domain traces.

    The last axis is interpreted as time. Multiple traces are averaged.
    """
    X = np.asarray(traces, dtype=np.float64)
    if X.ndim < 1:
        raise ValueError("traces must have at least one dimension")
    n = X.shape[-1]
    norm = float(sampling_frequency) * n
    spec = np.fft.rfft(X, axis=-1)
    psd = np.abs(spec) ** 2 / norm
    if X.ndim > 1:
        psd = psd.mean(axis=tuple(range(X.ndim - 1)))
    if n > 2:
        psd[1 : n // 2 + 1 - (n + 1) % 2] *= 2.0
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sampling_frequency))
    return freqs, psd


def regularize_psd(psd: np.ndarray, *, floor_fraction: float = 1e-8, floor: float | None = None) -> np.ndarray:
    """Clip non-positive PSD bins to a stable positive floor."""
    J = np.asarray(psd, dtype=np.float64)
    if np.any(~np.isfinite(J)):
        raise ValueError("PSD contains non-finite values")
    positive = J[J > 0]
    if positive.size == 0:
        raise ValueError("PSD has no positive bins")
    if floor is None:
        floor = max(float(np.median(positive)) * float(floor_fraction), np.finfo(float).tiny)
    return np.clip(J, float(floor), None)


def inverse_psd_weights(psd: np.ndarray, trace_len: int, *, zero_dc: bool = True) -> np.ndarray:
    """Build one-sided inverse-PSD weights for rFFT-domain inner products.

    This matches the OF convention used in the legacy tests: interior positive
    frequency bins receive ``2 / PSD`` and the Nyquist bin receives ``1 / (2 PSD)``
    for even-length traces.
    """
    J = regularize_psd(psd)
    n_bins = int(trace_len) // 2 + 1
    if J.shape[0] != n_bins:
        raise ValueError(f"PSD length {J.shape[0]} does not match rfft bins {n_bins}")
    w = np.zeros_like(J)
    if trace_len % 2 == 0:
        w[1:-1] = 2.0 / J[1:-1]
        w[-1] = 1.0 / (2.0 * J[-1])
    else:
        w[1:] = 2.0 / J[1:]
    if not zero_dc:
        w[0] = 1.0 / J[0]
    return w
