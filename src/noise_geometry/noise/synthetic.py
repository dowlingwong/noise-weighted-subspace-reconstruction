"""Synthetic PSD and Gaussian colored-noise generation."""

from __future__ import annotations

import numpy as np


def make_powerlaw_psd(
    n_samples: int,
    sampling_frequency: float,
    *,
    kind: str = "pink",
    level: float = 1.0,
    knee_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple one-sided PSD for white, pink, red, blue, or violet noise."""
    freqs = np.fft.rfftfreq(int(n_samples), d=1.0 / float(sampling_frequency))
    f0 = freqs[1] if freqs.size > 1 else 1.0
    f = np.maximum(freqs, f0)
    exponents = {
        "white": 0.0,
        "pink": -1.0,
        "red": -2.0,
        "brownian": -2.0,
        "brown": -2.0,
        "blue": 1.0,
        "violet": 2.0,
    }
    if kind not in exponents:
        raise ValueError(f"unknown noise kind {kind!r}")
    if knee_hz is not None:
        f = np.sqrt(f**2 + float(knee_hz) ** 2)
    psd = float(level) * (f / f0) ** exponents[kind]
    psd[0] = psd[1] if psd.size > 1 else psd[0]
    return freqs, psd


def generate_colored_noise(
    rng: np.random.Generator,
    psd: np.ndarray,
    n_samples: int,
    sampling_frequency: float,
    n_traces: int,
) -> np.ndarray:
    """Generate stationary Gaussian noise with a one-sided PSD."""
    J = np.asarray(psd, dtype=np.float64)
    n_bins = int(n_samples) // 2 + 1
    if J.shape[0] != n_bins:
        raise ValueError(f"PSD length {J.shape[0]} does not match rfft bins {n_bins}")
    scale = np.sqrt(J * float(sampling_frequency) * int(n_samples) / 2.0)
    re = rng.standard_normal((int(n_traces), n_bins))
    im = rng.standard_normal((int(n_traces), n_bins))
    X = (re + 1j * im) / np.sqrt(2.0) * scale[None, :]
    X[:, 0] = re[:, 0] * scale[0]
    if n_samples % 2 == 0:
        X[:, -1] = re[:, -1] * scale[-1] * np.sqrt(2.0)
    return np.fft.irfft(X, n=int(n_samples), axis=1)
