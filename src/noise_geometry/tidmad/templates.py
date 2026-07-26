"""Sinusoidal signal templates and the analytic quadrature subspace.

Why this module exists
----------------------
The TIDMAD signal (and the dark-matter signal the experiment searches for) is a
sinusoidal oscillation of unknown amplitude *and unknown phase*. That makes the
admissible signal manifold

    M = {A cos(2 pi f t + phi) : A >= 0, phi in [0, 2 pi)}
      = span{cos(2 pi f t), sin(2 pi f t)}                     (exactly rank 2)

This is the sharpest available test of Paper 1's rank claim, because unlike
CRESST -- where the rank-6 subspace was an empirical choice -- here the correct
rank is known analytically:

* rank 1 with a fixed (wrong) phase is suboptimal by exactly cos^2(phi_err);
* rank 2 is sufficient and cannot be improved on;
* rank >= 3 can only add noise directions.

See ``docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`` Phase 3.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "quadrature_templates",
    "quadrature_templates_rfft",
    "sinusoid",
    "nearest_bin_frequency",
]


def nearest_bin_frequency(frequency_hz: float, n_samples: int, sampling_frequency: float) -> float:
    """Snap a frequency onto the rFFT grid of an ``n_samples`` window.

    On-grid frequencies make the cosine and sine templates exactly orthogonal
    and confine the signal to a single rFFT bin, which removes spectral leakage
    as a confounder. Off-grid behaviour is a separate (and real) effect; it
    should be studied deliberately rather than entering by accident, so callers
    that want it must pass an off-grid frequency explicitly.
    """
    n = int(n_samples)
    fs = float(sampling_frequency)
    if n <= 1:
        raise ValueError("n_samples must be greater than one")
    if fs <= 0.0:
        raise ValueError("sampling_frequency must be positive")
    df = fs / n
    k = int(round(float(frequency_hz) / df))
    k = min(max(k, 0), n // 2)
    return k * df


def sinusoid(
    n_samples: int,
    sampling_frequency: float,
    frequency_hz: float,
    *,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Return ``amplitude * cos(2 pi f t + phase)`` on the sample grid."""
    n = int(n_samples)
    t = np.arange(n, dtype=np.float64) / float(sampling_frequency)
    return float(amplitude) * np.cos(2.0 * np.pi * float(frequency_hz) * t + float(phase))


def quadrature_templates(
    n_samples: int,
    sampling_frequency: float,
    frequency_hz: float,
) -> np.ndarray:
    """Return the ``(2, n_samples)`` analytic quadrature basis ``[cos; sin]``.

    The basis is *not* normalized here: normalization is a property of the
    weighted geometry (it depends on the PSD), so it is applied where the metric
    is known, in :mod:`noise_geometry.tidmad.analysis`.
    """
    n = int(n_samples)
    t = np.arange(n, dtype=np.float64) / float(sampling_frequency)
    omega = 2.0 * np.pi * float(frequency_hz)
    return np.vstack([np.cos(omega * t), np.sin(omega * t)])


def quadrature_templates_rfft(
    n_samples: int,
    sampling_frequency: float,
    frequency_hz: float,
) -> np.ndarray:
    """Return the quadrature basis as ``(2, n_rfft)`` complex rFFT rows."""
    basis = quadrature_templates(n_samples, sampling_frequency, frequency_hz)
    return np.fft.rfft(basis, axis=1)
