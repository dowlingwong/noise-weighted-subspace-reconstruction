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
) -> Rank1PulseDataset:
    """Generate ``x_i = a_i s0 + n_i`` with a known PSD."""
    rng = np.random.default_rng(seed)
    template = exponential_pulse(n_samples, sampling_frequency)
    freqs, psd = make_powerlaw_psd(n_samples, sampling_frequency, kind=noise_kind, level=noise_level)
    amplitudes = rng.uniform(0.5, 1.5, size=int(n_traces))
    noise = generate_colored_noise(rng, psd, n_samples, sampling_frequency, n_traces)
    clean = amplitudes[:, None] * template[None, :]
    weights = inverse_psd_weights(psd, n_samples)
    return Rank1PulseDataset(clean + noise, clean, template, amplitudes, psd, freqs, weights, sampling_frequency)


def run_of_empca_equivalence(dataset: Rank1PulseDataset, *, rank: int = 1) -> dict[str, float]:
    """Compute OF amplitudes and a weighted rank-1 subspace check."""
    from EMPCA.empca_equivalence_utils import (
        fit_empca_no_smoothing,
        phase_align_basis,
        weighted_cosine,
    )

    X_f = np.fft.rfft(dataset.traces, axis=1)
    s_f = np.fft.rfft(dataset.template)
    amp_of = gls_amplitude(X_f, s_f, dataset.weights)

    eigvec, coeff, chi2s = fit_empca_no_smoothing(X_f, dataset.weights, n_comp=rank, n_iter=100)
    aligned = phase_align_basis(eigvec[0], s_f, dataset.weights)
    cosine = weighted_cosine(aligned, s_f, dataset.weights)
    angle = float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))
    coeff_corr = float(np.corrcoef(np.abs(coeff[:, 0]), amp_of)[0, 1])
    corr = float(np.corrcoef(amp_of, dataset.amplitudes)[0, 1])
    return {
        "n_traces": float(dataset.traces.shape[0]),
        "of_truth_corr": corr,
        "empca_of_coeff_corr": coeff_corr,
        "weighted_cosine": cosine,
        "weighted_angle_deg": angle,
        "empca_objective_decrease": float(chi2s[0] - chi2s[-1]),
        "amp_rmse": float(np.sqrt(np.mean((amp_of - dataset.amplitudes) ** 2))),
    }
