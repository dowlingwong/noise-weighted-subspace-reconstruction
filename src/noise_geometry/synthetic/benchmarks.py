"""Controlled synthetic benchmarks for the Paper 1 claims."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..filters import gls_amplitude
from ..noise import generate_colored_noise, inverse_psd_weights, make_powerlaw_psd
from ..subspace import fit_weighted_pca, principal_angles
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
    X_f = np.fft.rfft(dataset.traces, axis=1)
    s_f = np.fft.rfft(dataset.template)
    amp_of = gls_amplitude(X_f, s_f, dataset.weights)

    fit = fit_weighted_pca(X_f.real, dataset.weights, rank=rank, center=False)
    angles = principal_angles(fit.components[:1], s_f.real[None, :], weights=dataset.weights)
    corr = float(np.corrcoef(amp_of, dataset.amplitudes)[0, 1])
    return {
        "n_traces": float(dataset.traces.shape[0]),
        "of_truth_corr": corr,
        "weighted_angle_deg": float(angles[0]),
        "amp_rmse": float(np.sqrt(np.mean((amp_of - dataset.amplitudes) ** 2))),
    }
