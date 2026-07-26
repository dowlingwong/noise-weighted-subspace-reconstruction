"""Contract tests for the TIDMAD quadrature estimators.

These pin the mathematical facts the TIDMAD stage relies on, so a convention
change elsewhere in the repository cannot silently invalidate Phase 2 and Phase
3. They are fast and deterministic; the full Phase 1 gate lives in
``scripts/tidmad_phase1_estimator_identity.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.noise_geometry.noise import (
    generate_colored_noise,
    inverse_psd_weights,
    make_powerlaw_psd,
)
from src.noise_geometry.subspace import fit_weighted_pca, principal_angles
from src.noise_geometry.tidmad import (
    chronological_window_split,
    fit_fixed_phase,
    fit_quadrature,
    fixed_phase_snr_loss,
    inject_sinusoid,
    leakage_audit,
    nearest_bin_frequency,
    quadrature_amplitude_variance,
    quadrature_templates,
    quadrature_templates_rfft,
    run_injection_check,
    welch_psd_from_windows,
)

from src.canonical.empca_equivalence_utils import complex_to_real_whitened

FS = 1_000_000.0
N = 2048
SEED = 20260727


@pytest.fixture(scope="module")
def noise_setup():
    rng = np.random.default_rng(SEED)
    _, psd = make_powerlaw_psd(N, FS, kind="pink")
    calibration = generate_colored_noise(rng, psd, N, FS, 256)
    evaluation = generate_colored_noise(rng, psd, N, FS, 256)
    weights = inverse_psd_weights(psd, N)
    frequency = nearest_bin_frequency(FS / 10.0, N, FS)
    return psd, weights, calibration, evaluation, frequency


def test_on_grid_frequency_makes_quadrature_orthogonal(noise_setup):
    _, weights, _, _, frequency = noise_setup
    basis = quadrature_templates(N, FS, frequency)
    # Exactly orthogonal in time domain when the frequency sits on the rFFT grid.
    assert abs(float(basis[0] @ basis[1])) < 1e-8 * N
    # And equal-norm, which is what makes the two quadratures carry equal
    # information and the amplitude variance direction-independent.
    assert float(basis[0] @ basis[0]) == pytest.approx(float(basis[1] @ basis[1]), rel=1e-9)


def test_quadrature_fit_is_exact_without_noise(noise_setup):
    _, weights, _, _, frequency = noise_setup
    basis_f = quadrature_templates_rfft(N, FS, frequency)
    truth_a, truth_b = 3.0, -1.5
    clean = truth_a * np.cos(2 * np.pi * frequency * np.arange(N) / FS) + truth_b * np.sin(
        2 * np.pi * frequency * np.arange(N) / FS
    )
    fit = fit_quadrature(np.fft.rfft(clean[None, :], axis=1), basis_f, weights)
    assert float(fit.cos_coefficient[0]) == pytest.approx(truth_a, rel=1e-8)
    assert float(fit.sin_coefficient[0]) == pytest.approx(truth_b, rel=1e-8)
    assert float(fit.weighted_residual[0]) == pytest.approx(0.0, abs=1e-6)


def test_quadrature_fit_recovers_amplitude_and_phase(noise_setup):
    _, weights, _, _, frequency = noise_setup
    basis_f = quadrature_templates_rfft(N, FS, frequency)
    amplitude, phase = 2.0, 0.7
    t = np.arange(N) / FS
    clean = amplitude * np.cos(2 * np.pi * frequency * t + phase)
    fit = fit_quadrature(np.fft.rfft(clean[None, :], axis=1), basis_f, weights)
    assert float(fit.amplitude[0]) == pytest.approx(amplitude, rel=1e-8)
    assert float(fit.phase[0]) == pytest.approx(phase, abs=1e-8)


def test_injection_recovery_is_unbiased_and_matches_predicted_sigma(noise_setup):
    psd, weights, _, evaluation, frequency = noise_setup
    basis_f = quadrature_templates_rfft(N, FS, frequency)
    sigma = np.sqrt(
        quadrature_amplitude_variance(
            basis_f, psd, weights, trace_len=N, sampling_frequency=FS
        )
    )
    result = run_injection_check(
        evaluation,
        psd,
        weights,
        FS,
        frequency,
        amplitude=50.0 * float(sigma),
        rng=np.random.default_rng(SEED + 1),
    )
    z = (result.mean_recovered_amplitude - result.true_amplitude) / (
        result.empirical_sigma / np.sqrt(result.n_windows)
    )
    assert abs(z) < 4.0, "amplitude estimate is biased"
    assert result.sigma_ratio == pytest.approx(1.0, abs=0.15)


def test_fixed_phase_filter_loses_amplitude_as_cosine(noise_setup):
    _, weights, _, evaluation, frequency = noise_setup
    basis_f = quadrature_templates_rfft(N, FS, frequency)
    injected, _ = inject_sinusoid(evaluation, FS, frequency, amplitude=200.0, phases=0.0)
    X_f = np.fft.rfft(injected, axis=1)
    reference = np.mean(np.asarray(fit_fixed_phase(X_f, basis_f, weights, phase=0.0)))
    for phase_error in (0.3, 0.6, 1.0):
        amps = np.mean(np.asarray(fit_fixed_phase(X_f, basis_f, weights, phase=phase_error)))
        # Amplitude falls as cos; the SNR *power* falls as cos^2 (Paper 1 Eq. 16).
        assert float(amps / reference) == pytest.approx(np.cos(phase_error), abs=0.02)
        assert fixed_phase_snr_loss(phase_error) == pytest.approx(np.cos(phase_error) ** 2)


def test_learned_rank2_subspace_recovers_analytic_quadrature(noise_setup):
    psd, weights, _, evaluation, frequency = noise_setup
    basis_f = quadrature_templates_rfft(N, FS, frequency)
    sigma = np.sqrt(
        quadrature_amplitude_variance(
            basis_f, psd, weights, trace_len=N, sampling_frequency=FS
        )
    )
    injected, _ = inject_sinusoid(
        evaluation, FS, frequency, amplitude=300.0 * float(sigma), rng=np.random.default_rng(SEED + 2)
    )
    features = complex_to_real_whitened(np.fft.rfft(injected, axis=1), weights)
    learned = fit_weighted_pca(features, np.ones(features.shape[1]), 2)
    analytic = complex_to_real_whitened(basis_f, weights)
    assert float(np.max(principal_angles(learned.components, analytic))) < 5.0


def test_psd_estimate_recovers_the_generating_psd(noise_setup):
    psd, _, calibration, _, _ = noise_setup
    estimate = welch_psd_from_windows(list(calibration), FS, window="boxcar", detrend=True)
    ratio = estimate.psd[1:-1] / psd[1:-1]
    assert float(np.median(ratio)) == pytest.approx(1.0, abs=0.05)


def test_chronological_split_is_disjoint_with_a_guard_gap():
    split = chronological_window_split(100, fit_fraction=0.7, guard_windows=5)
    assert split.fit_indices.max() == 69
    assert split.evaluation_indices.min() == 75
    audit = leakage_audit(split, window_samples=1024, stride=1024, sampling_frequency=FS)
    assert audit["index_overlap_count"] == 0
    assert audit["guard_gap_windows_realized"] == 5
    assert audit["windows_overlap_in_samples"] is False


def test_guard_gap_that_consumes_all_evaluation_windows_is_rejected():
    with pytest.raises(ValueError):
        chronological_window_split(10, fit_fraction=0.9, guard_windows=50)
