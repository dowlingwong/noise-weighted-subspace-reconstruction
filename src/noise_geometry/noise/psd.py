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


def estimate_psd_welch(
    traces: np.ndarray,
    sampling_frequency: float = 1.0,
    *,
    segment_length: int | None = None,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a one-sided PSD using Welch averaging over trace segments."""
    X = np.asarray(traces, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError("traces must have shape (n_traces, n_samples)")
    n = X.shape[1]
    if segment_length is None:
        segment_length = min(n, 1024)
    segment_length = int(segment_length)
    if segment_length <= 1 or segment_length > n:
        raise ValueError("segment_length must be in [2, n_samples]")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    step = max(1, int(segment_length * (1.0 - overlap)))
    starts = list(range(0, n - segment_length + 1, step))
    if not starts:
        starts = [0]
    window = np.hanning(segment_length)
    norm = np.mean(window**2)
    segments = np.concatenate([X[:, s : s + segment_length] * window[None, :] for s in starts], axis=0)
    freqs, psd = estimate_psd_rfft(segments, sampling_frequency)
    return freqs, psd / max(norm, np.finfo(float).eps)


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

    The OF convention (interior bins ``2 / PSD``, Nyquist ``1 / (2 PSD)`` for
    even length, DC zeroed) is defined once in
    :func:`canonical.make_weights.build_of_one_sided_weights`; this wrapper
    regularizes the PSD, delegates to it, and applies the ``zero_dc`` override so
    the convention lives in exactly one place.
    """
    try:  # repo-root on path (pytest)
        from ...canonical.make_weights import build_of_one_sided_weights
    except ImportError:  # src on path (scripts/run_experiment.py)
        from canonical.make_weights import build_of_one_sided_weights

    J = regularize_psd(psd)
    n_bins = int(trace_len) // 2 + 1
    if J.shape[0] != n_bins:
        raise ValueError(f"PSD length {J.shape[0]} does not match rfft bins {n_bins}")
    w = build_of_one_sided_weights(J, trace_len)
    if not zero_dc:
        w = w.copy()
        w[0] = 1.0 / J[0]
    return w
