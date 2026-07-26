"""Contract tests for TIDMAD Phases 2-5.

These pin the properties the phases depend on, so that a change elsewhere in
the repository cannot silently invalidate a gate:

* Phase 2 -- the three metric families disagree in the predicted directions,
  and the truth metric is the one that adjudicates;
* Phase 3 -- the rank curve has its knee at 2, as the closed-form prediction
  requires;
* Phase 4 -- the chi^2 statistic really has expectation one under a correct
  model, and the local-PSD estimator excludes the window it is scoring;
* Phase 5 -- the overlap-add denoiser reconstructs exactly when the filter is
  the identity, and the Wiener transfer function stays in [0, 1].

They run in seconds on small windows; the full gates live in
``scripts/tidmad_phases.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.canonical.empca_equivalence_utils import real_weight_vector
from src.noise_geometry.noise import (
    generate_colored_noise,
    inverse_psd_weights,
    make_powerlaw_psd,
)
from src.noise_geometry.tidmad import (
    inject_sinusoid,
    nearest_bin_frequency,
    quadrature_amplitude_variance,
    quadrature_templates_rfft,
)
from src.noise_geometry.tidmad.analysis import (
    build_window_set,
    run_metric_reversal,
    run_rank_sweep,
)
from src.noise_geometry.tidmad.calibration import (
    _effective_dof,
    assess_global_model,
    assess_local_model,
    whitened_chi2_per_dof,
)
from src.noise_geometry.tidmad.denoise import (
    WienerFilter,
    build_wiener_filter,
    denoise_stream,
    estimate_signal_psd,
    wiener_transfer_function,
)
from src.noise_geometry.tidmad.score import DenoisedFileSpec

FS = 1_000_000.0
N = 1024
SEED = 20260727


@pytest.fixture(scope="module")
def prepared():
    rng = np.random.default_rng(SEED)
    _, psd_truth = make_powerlaw_psd(N, FS, kind="pink")
    noise = generate_colored_noise(rng, psd_truth, N, FS, 256)
    calibration = generate_colored_noise(rng, psd_truth, N, FS, 256)

    frequency = nearest_bin_frequency(FS / 8.0, N, FS)
    weights = inverse_psd_weights(psd_truth, N)
    basis_f = quadrature_templates_rfft(N, FS, frequency)
    sigma = float(
        np.sqrt(
            quadrature_amplitude_variance(
                basis_f, psd_truth, weights, trace_len=N, sampling_frequency=FS
            )
        )
    )
    observed, _ = inject_sinusoid(noise, FS, frequency, amplitude=12.0 * sigma, rng=rng)
    clean = observed - noise
    windows = build_window_set(
        observed, clean, psd_truth, weights, FS, frequency, fit_fraction=0.7, guard_windows=2
    )
    return windows, calibration, psd_truth, weights, frequency


def test_metric_reversal_shows_the_predicted_disagreement(prepared):
    windows, *_ = prepared
    result = run_metric_reversal(windows, rank=2, bootstrap_reps=100, seed=SEED)

    direction = result["direction_check"]
    assert direction["raw_mse_favours_pca"], "isotropic PCA should win the metric it optimizes"
    assert direction["weighted_residual_favours_empca"], "EMPCA should win the likelihood metric"
    # The claim: the truth metric sides with the likelihood, not with raw MSE.
    assert direction["truth_favours_empca"]

    truth_contrast = result["paired_contrasts_window_level"][
        "amplitude_abs_error_pca_minus_empca"
    ]
    assert truth_contrast["exceeds_zero"], "the truth-metric advantage must exclude zero"


def test_metric_reversal_reports_a_clean_leakage_audit(prepared):
    windows, *_ = prepared
    result = run_metric_reversal(windows, rank=2, bootstrap_reps=50, seed=SEED)
    audit = result["leakage_audit"]
    assert audit["index_overlap_count"] == 0
    assert audit["hash_overlap_count"] == 0
    assert audit["guard_gap_windows_realized"] >= 2


def test_rank_sweep_knee_is_at_the_analytic_rank(prepared):
    windows, *_ = prepared
    result = run_rank_sweep(
        windows, ranks=(1, 2, 3, 4), bootstrap_reps=100, saturation_tolerance=0.05, seed=SEED
    )
    rmse = result["amplitude_rmse_by_rank"]
    assert rmse[2] < rmse[1], "rank 2 must beat rank 1: the phase needs a second direction"
    # Rank 3 cannot help: the manifold is exactly two-dimensional.
    assert rmse[3] >= rmse[2] * (1.0 - 0.05)
    assert result["smallest_rank_within_tolerance"] == 2


def test_rank2_subspace_is_close_to_the_analytic_quadrature(prepared):
    windows, *_ = prepared
    result = run_rank_sweep(windows, ranks=(2,), bootstrap_reps=50, seed=SEED)
    angle = result["per_rank"]["2"]["max_principal_angle_to_analytic_deg"]
    # Finite-sample, so not tiny; the scaling law is checked in Phase 1.
    assert 0.0 <= angle < 60.0


def test_chi2_per_dof_has_unit_expectation_under_a_correct_model(prepared):
    _, calibration, psd_truth, weights, _ = prepared
    chi2 = whitened_chi2_per_dof(calibration, weights, FS)
    # 256 windows x ~1000 dof: the mean is pinned tightly.
    assert float(np.mean(chi2)) == pytest.approx(1.0, abs=0.02)


def test_effective_dof_counts_real_degrees_of_freedom(prepared):
    _, _, psd_truth, weights, _ = prepared
    dof = _effective_dof(N, weights)
    # DC is zero-weighted; interior bins carry two; the Nyquist bin carries one.
    assert dof == 2 * (N // 2 - 1) + 1


def test_global_calibration_of_a_correct_model_is_near_unity(prepared):
    _, calibration, _, _, frequency = prepared
    half = calibration.shape[0] // 2
    result = assess_global_model(calibration[:half], calibration[half:], FS, frequency)
    assert result.chi2_per_dof_mean == pytest.approx(1.0, abs=0.05)
    assert abs(result.score_dispersion - 1.0) < 5.0 * result.score_dispersion_standard_error


def test_local_model_excludes_the_window_it_scores(prepared):
    _, calibration, _, _, frequency = prepared
    result = assess_local_model(calibration[:64], FS, frequency, radius_windows=8)
    assert result.n_windows == 64
    # Leaking the scored window into its own PSD would bias chi^2 *below* one;
    # a correct leave-one-out estimate stays at or above it.
    assert result.chi2_per_dof_mean > 0.9


def test_local_model_rejects_too_few_windows(prepared):
    _, calibration, _, _, frequency = prepared
    with pytest.raises(ValueError):
        assess_local_model(calibration[:5], FS, frequency, radius_windows=8)


def test_overlap_add_is_exact_for_an_identity_filter():
    rng = np.random.default_rng(SEED)
    _, psd = make_powerlaw_psd(N, FS, kind="pink")
    identity = WienerFilter(
        frequencies=np.fft.rfftfreq(N, 1.0 / FS),
        transfer=np.ones(N // 2 + 1),
        signal_psd=psd,
        noise_psd=psd,
        window_samples=N,
        sampling_frequency=FS,
    )
    stream = rng.standard_normal(N * 7 + 137)
    assert np.allclose(denoise_stream(stream, identity), stream, atol=1e-9)


def test_wiener_filter_reduces_error_and_stays_bounded():
    rng = np.random.default_rng(SEED)
    _, psd = make_powerlaw_psd(N, FS, kind="pink")
    t = np.arange(N) / FS
    frequency = nearest_bin_frequency(FS / 8.0, N, FS)
    clean = 5.0 * np.cos(2.0 * np.pi * frequency * t)

    signal_windows = generate_colored_noise(rng, psd, N, FS, 128) + clean[None, :]
    noise_windows = generate_colored_noise(rng, psd, N, FS, 128)
    wiener = build_wiener_filter(signal_windows, noise_windows, FS)

    assert float(wiener.transfer.min()) >= 0.0
    assert float(wiener.transfer.max()) <= 1.0

    stream_clean = np.tile(clean, 8)
    stream = generate_colored_noise(rng, psd, N, FS, 8).ravel() + stream_clean
    denoised = denoise_stream(stream, wiener)
    before = float(np.mean((stream - stream_clean) ** 2))
    after = float(np.mean((denoised - stream_clean) ** 2))
    assert after < 0.5 * before


def test_spectral_subtraction_cannot_produce_an_amplifier():
    total = np.array([1.0, 2.0, 0.5])
    noise = np.array([1.0, 1.0, 1.0])  # third bin: noise exceeds the total
    signal = estimate_signal_psd(total, noise)
    assert np.all(signal >= 0.0)
    transfer = wiener_transfer_function(signal, noise)
    assert np.all((transfer >= 0.0) & (transfer <= 1.0))


def test_denoised_filename_matches_the_release_pattern(tmp_path):
    spec = DenoisedFileSpec(
        source_file=tmp_path / "abra_validation_0007.h5",
        destination_dir=tmp_path / "denoised",
        model_tag="wiener",
    )
    assert spec.destination.name == "abra_validation_denoised_wiener_0007.h5"
