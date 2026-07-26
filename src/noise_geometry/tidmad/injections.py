"""Synthetic-injection harness: the Phase 1 verification instrument.

Purpose. Before any conclusion is drawn from TIDMAD's own injected signals, the
estimator and its predicted variance must be shown correct on data where the
answer is known by construction. That is what this module provides: inject a
sinusoid of known amplitude and phase into noise (synthetic *or real*
signal-free stretches), then check that

* the recovered amplitude is unbiased,
* the empirical spread matches the PSD-predicted bound,
* a no-signal null returns amplitude consistent with zero.

Injecting into *real* science-file noise is the important case, because it tests
the estimator against the actual noise distribution -- including its
non-Gaussian tails -- rather than against the Gaussian model. Both paths use the
same code, so the difference between them is a measurement, not a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .estimators import fit_quadrature, quadrature_amplitude_variance
from .templates import quadrature_templates, quadrature_templates_rfft

__all__ = ["InjectionResult", "inject_sinusoid", "run_injection_check"]


@dataclass(frozen=True)
class InjectionResult:
    """Outcome of an injection-recovery experiment."""

    n_windows: int
    true_amplitude: float
    mean_recovered_amplitude: float
    amplitude_bias_fraction: float
    empirical_sigma: float
    predicted_sigma: float
    sigma_ratio: float
    cos_coefficients: np.ndarray
    sin_coefficients: np.ndarray
    recovered_amplitudes: np.ndarray
    recovered_phases: np.ndarray

    def as_dict(self) -> dict:
        return {
            "n_windows": int(self.n_windows),
            "true_amplitude": float(self.true_amplitude),
            "mean_recovered_amplitude": float(self.mean_recovered_amplitude),
            "amplitude_bias_fraction": float(self.amplitude_bias_fraction),
            "empirical_sigma": float(self.empirical_sigma),
            "predicted_sigma": float(self.predicted_sigma),
            "sigma_ratio": float(self.sigma_ratio),
        }


def inject_sinusoid(
    noise_windows: np.ndarray,
    sampling_frequency: float,
    frequency_hz: float,
    *,
    amplitude: float,
    phases: Sequence[float] | np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add ``amplitude * cos(2 pi f t + phi)`` to each noise window.

    Phases are randomized by default. That is deliberate: a fixed phase would
    let a rank-1 estimator look artificially good and would defeat the whole
    point of the quadrature test.

    Returns ``(windows_with_signal, phases_used)``.
    """
    X = np.atleast_2d(np.asarray(noise_windows, dtype=np.float64))
    n_windows, n_samples = X.shape
    if phases is None:
        generator = rng if rng is not None else np.random.default_rng()
        phi = generator.uniform(0.0, 2.0 * np.pi, size=n_windows)
    else:
        phi = np.asarray(phases, dtype=np.float64)
        if phi.size == 1:
            phi = np.full(n_windows, float(phi))
        if phi.size != n_windows:
            raise ValueError("phases must be scalar or match the number of windows")

    basis = quadrature_templates(n_samples, sampling_frequency, frequency_hz)
    signal = float(amplitude) * (
        np.cos(phi)[:, None] * basis[0][None, :] - np.sin(phi)[:, None] * basis[1][None, :]
    )
    return X + signal, phi


def run_injection_check(
    noise_windows: np.ndarray,
    psd: np.ndarray,
    weights: np.ndarray,
    sampling_frequency: float,
    frequency_hz: float,
    *,
    amplitude: float,
    rng: np.random.Generator | None = None,
) -> InjectionResult:
    """Inject, recover, and compare against the predicted variance (gate G1).

    ``psd`` and ``weights`` must be estimated from data *disjoint* from
    ``noise_windows``; passing the same windows would make the check circular
    and it would pass for the wrong reason.
    """
    X = np.atleast_2d(np.asarray(noise_windows, dtype=np.float64))
    n_samples = X.shape[1]

    injected, phases = inject_sinusoid(
        X, sampling_frequency, frequency_hz, amplitude=amplitude, rng=rng
    )
    basis_f = quadrature_templates_rfft(n_samples, sampling_frequency, frequency_hz)
    fit = fit_quadrature(np.fft.rfft(injected, axis=1), basis_f, weights)

    # Project the estimated coefficients onto the true quadrature direction so
    # the spread is measured along the direction the amplitude actually lives
    # in. This avoids the Rice bias that contaminates sqrt(a^2+b^2) at low SNR
    # and makes the comparison to the predicted coefficient variance exact.
    projected = fit.cos_coefficient * np.cos(phases) - fit.sin_coefficient * np.sin(phases)

    predicted_variance = quadrature_amplitude_variance(
        basis_f, psd, weights, trace_len=n_samples, sampling_frequency=sampling_frequency
    )
    predicted_sigma = float(np.sqrt(predicted_variance))
    empirical_sigma = float(np.std(projected - float(amplitude), ddof=1)) if projected.size > 1 else float("nan")
    mean_amp = float(np.mean(projected))

    return InjectionResult(
        n_windows=int(X.shape[0]),
        true_amplitude=float(amplitude),
        mean_recovered_amplitude=mean_amp,
        amplitude_bias_fraction=(
            float((mean_amp - amplitude) / amplitude) if amplitude != 0.0 else float("nan")
        ),
        empirical_sigma=empirical_sigma,
        predicted_sigma=predicted_sigma,
        sigma_ratio=float(empirical_sigma / predicted_sigma) if predicted_sigma > 0 else float("nan"),
        cos_coefficients=np.asarray(fit.cos_coefficient),
        sin_coefficients=np.asarray(fit.sin_coefficient),
        recovered_amplitudes=np.asarray(fit.amplitude),
        recovered_phases=np.asarray(fit.phase),
    )
