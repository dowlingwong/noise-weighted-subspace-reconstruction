"""Signal injection for CRESST detector-facing evaluation.

Two conditions, named per the honesty rule in the CRESST validation protocol:

- ``inject_direct``: direct injection of a template into true noise-flagged
  baseline traces (preferred whenever an event-level noise bank exists, as it
  does in this release).
- ``inject_measured_psd``: a measured-PSD Gaussian condition, drawn from the
  noise covariance estimated from the same noise traces. Never call this
  "measured-noise-trace injection" — it replaces the real noise realization
  with a Gaussian surrogate matching its first/second moments only.
"""

from __future__ import annotations

import numpy as np

from ..noise import estimate_covariance, regularize_covariance


def inject_direct(
    noise_traces: np.ndarray,
    template: np.ndarray,
    amplitudes: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject ``amplitude * template`` into randomly drawn real noise traces.

    Returns ``(injected_traces, noise_indices)``.
    """
    noise_traces = np.asarray(noise_traces, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    idx = rng.integers(0, noise_traces.shape[0], size=amplitudes.shape[0])
    injected = noise_traces[idx] + amplitudes[:, None] * template[None, :]
    return injected, idx


def inject_measured_psd(
    noise_traces: np.ndarray,
    template: np.ndarray,
    amplitudes: np.ndarray,
    rng: np.random.Generator,
    *,
    shrinkage: float = 0.1,
) -> np.ndarray:
    """"Measured-PSD injection": Gaussian noise from the estimated covariance.

    Matches integrated variance to the real noise bank by construction (the
    Gaussian is drawn from the covariance estimated from ``noise_traces``
    itself), but discards real higher-order structure.
    """
    noise_traces = np.asarray(noise_traces, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    covariance = regularize_covariance(estimate_covariance(noise_traces), shrinkage=shrinkage)
    gaussian = rng.multivariate_normal(np.zeros(covariance.shape[0]), covariance, size=amplitudes.shape[0])
    return gaussian + amplitudes[:, None] * template[None, :]
