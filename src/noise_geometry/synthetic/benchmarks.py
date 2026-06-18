"""Controlled synthetic benchmarks for the Paper 1 claims."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..filters import gls_amplitude
from ..noise import generate_colored_noise, inverse_psd_weights, make_powerlaw_psd
from .pulses import exponential_pulse


@dataclass
class Rank1PulseDataset:
    traces: np.ndarray
    clean: np.ndarray
    template: np.ndarray
    amplitudes: np.ndarray
    psd: np.ndarray
    freqs: np.ndarray
    weights: np.ndarray
    sampling_frequency: float


def make_rank1_pulse_dataset(
    *,
    n_traces: int = 256,
    n_samples: int = 1024,
    sampling_frequency: float = 1.0e5,
    noise_kind: str = "pink",
    noise_level: float = 1e-5,
    seed: int = 7,
    amplitude_model: str = "real",
) -> Rank1PulseDataset:
    """Generate ``x_i = c_i s0 + n_i`` with a known PSD.

    ``amplitude_model`` selects the coefficient field, which sets the dimension
    of the signal subspace and therefore the right S2 acceptance metric:

    - ``"real"``: ``c_i`` is a real amplitude (fixed-shape pulse, no timing
      jitter). The signal subspace is one real dimension (the template); rank-1
      EMPCA collapses onto the OF template in the large-sample limit.
    - ``"complex"``: ``c_i`` is a complex coefficient (amplitude and phase /
      sub-sample time-shift). The signal subspace is two real dimensions
      (``span{s, i s}``); the template lies inside the rank-1 EMPCA span but a
      single-template cosine is capped below 1.

    ``amplitudes`` stores the coefficient magnitude in both cases.
    """
    rng = np.random.default_rng(seed)
    template = exponential_pulse(n_samples, sampling_frequency)
    freqs, psd = make_powerlaw_psd(n_samples, sampling_frequency, kind=noise_kind, level=noise_level)
    magnitudes = rng.uniform(0.5, 1.5, size=int(n_traces))
    if amplitude_model == "real":
        clean = magnitudes[:, None] * template[None, :]
    elif amplitude_model == "complex":
        s_f = np.fft.rfft(template)
        phases = rng.uniform(-np.pi, np.pi, size=int(n_traces))
        coeff = magnitudes * np.exp(1j * phases)
        clean = np.fft.irfft(coeff[:, None] * s_f[None, :], n=n_samples, axis=1)
    else:
        raise ValueError("amplitude_model must be 'real' or 'complex'")
    noise = generate_colored_noise(rng, psd, n_samples, sampling_frequency, n_traces)
    weights = inverse_psd_weights(psd, n_samples)
    return Rank1PulseDataset(clean + noise, clean, template, magnitudes, psd, freqs, weights, sampling_frequency)


def run_of_empca_equivalence(
    dataset: Rank1PulseDataset, *, rank: int = 1, amplitude_model: str = "real"
) -> dict[str, float]:
    """OF amplitudes plus a weighted rank-1 subspace check.

    Two acceptance metrics are always reported; ``amplitude_model`` selects which
    is the headline (``acceptance_metric``/``acceptance_angle_deg``):

    - ``weighted_angle_deg`` (real model): angle between the rank-1 EMPCA
      eigenvector and the OF template. Finite-sample, shrinks at the 1/sqrt(N)
      rate; the headline when coefficients are real.
    - ``template_in_span_angle_deg`` (complex model): angle of the template into
      the EMPCA's two-real-dimensional span ``{u, i u}``. Goes to zero even when
      a single-template cosine is structurally capped below 1.
    """
    try:  # repo-root on path (pytest)
        from src.canonical.empca_equivalence_utils import (
            complex_to_real_whitened,
            fit_empca_no_smoothing,
            phase_align_basis,
            weighted_cosine,
        )
    except ImportError:  # src on path (scripts/run_experiment.py)
        from canonical.empca_equivalence_utils import (
            complex_to_real_whitened,
            fit_empca_no_smoothing,
            phase_align_basis,
            weighted_cosine,
        )
    from ..subspace import principal_angles

    X_f = np.fft.rfft(dataset.traces, axis=1)
    s_f = np.fft.rfft(dataset.template)
    w = dataset.weights
    amp_of = gls_amplitude(X_f, s_f, w)

    eigvec, coeff, chi2s = fit_empca_no_smoothing(X_f, w, n_comp=rank, n_iter=100)
    u = eigvec[0]
    aligned = phase_align_basis(u, s_f, w)
    cosine = weighted_cosine(aligned, s_f, w)
    angle = float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))

    # Angle of the template into the EMPCA's 2-real-dim span {u, i u}.
    s_feat = complex_to_real_whitened(s_f, w)
    span_basis = np.vstack([complex_to_real_whitened(u, w), complex_to_real_whitened(1j * u, w)])
    span_angle = float(np.max(principal_angles(s_feat[None, :], span_basis)))

    headline = span_angle if amplitude_model == "complex" else angle
    coeff_corr = float(np.corrcoef(np.abs(coeff[:, 0]), amp_of)[0, 1])
    corr = float(np.corrcoef(amp_of, dataset.amplitudes)[0, 1])
    return {
        "n_traces": float(dataset.traces.shape[0]),
        "amplitude_model": amplitude_model,
        "acceptance_metric": "template_in_span_angle_deg" if amplitude_model == "complex" else "weighted_angle_deg",
        "acceptance_angle_deg": headline,
        "of_truth_corr": corr,
        "empca_of_coeff_corr": coeff_corr,
        "weighted_cosine": cosine,
        "weighted_angle_deg": angle,
        "template_in_span_angle_deg": span_angle,
        "empca_objective_decrease": float(chi2s[0] - chi2s[-1]),
        "amp_rmse": float(np.sqrt(np.mean((amp_of - dataset.amplitudes) ** 2))),
    }
