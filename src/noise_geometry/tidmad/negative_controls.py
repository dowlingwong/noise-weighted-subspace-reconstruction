"""Phase 6: negative controls for the TIDMAD stage.

Roadmap: ``docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`` Phase 6.

Why a positive result is not enough
-----------------------------------
Phase 2 shows that the covariance-weighted reconstruction recovers the injected
signal better than the Euclidean one. On its own that is compatible with a
pipeline that would report a win no matter what it was fed. The controls
deliberately break one ingredient at a time and require the primary metric to
degrade. A pipeline that still "wins" after its noise model has been scrambled
is measuring something other than what it claims.

Five controls, matched one-for-one with the CRESST set so the two real-data
stages are falsifiable in the same way:

============================  ============================================
control                       what it breaks
============================  ============================================
``wrong_psd``                 the noise model (white metric instead of Sigma)
``noise_only_train``          the training signal (fit on signal-free windows)
``random_weighted_basis``     the learned subspace (random basis, same rank)
``wrong_frequency``           the signal model (filter at an empty frequency)
``permuted_bins``             the time structure (per-window sample permutation)
============================  ============================================

Each control returns its *per-window* error array so the caller can pair it
against the matching nominal array on the same held-out windows and bootstrap
the difference. A raw point-estimate win or loss is not evidence.

A control that fails to degrade is itself a result. CRESST had one such case
(wrong-PSD weighting on channel 35/58.0) and reported it. The same rule applies
here: report it, do not tune it away.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .analysis import (
    _canonical_helpers,
    _fit_subspace,
    _project,
    _real_weights,
    _rfft_real_views,
    _truth_amplitude,
    score_reconstruction,
)
from .estimators import fit_fixed_phase
from .templates import nearest_bin_frequency, quadrature_templates_rfft

__all__ = ["CONTROL_NOMINAL", "CONTROLS", "run_negative_controls"]

#: Which nominal arm each control is paired against.
CONTROL_NOMINAL: dict[str, str] = {
    "wrong_psd": "empca",
    "noise_only_train": "empca",
    "random_weighted_basis": "empca",
    "wrong_frequency": "empca",
    "permuted_bins": "empca",
}

CONTROLS = tuple(CONTROL_NOMINAL)


def _random_weighted_orthonormal_basis(
    rank: int, n_features: int, weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Random rank-``k`` basis, orthonormal under the diagonal weighted product.

    Orthonormalizing under the *same* metric matters: a basis that is random but
    not properly normalized would degrade partly for a bookkeeping reason rather
    than because it points in uninformative directions.
    """
    A = rng.standard_normal((int(rank), int(n_features)))
    w = np.asarray(weights, dtype=np.float64)
    gram = (A * w[None, :]) @ A.T
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    values = np.clip(values, np.finfo(float).eps, None)
    inv_sqrt = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    return inv_sqrt @ A


def _permute_rows_independently(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_features = X.shape[1]
    perm = np.array([rng.permutation(n_features) for _ in range(X.shape[0])])
    return np.take_along_axis(X, perm, axis=1)


def _score(
    name: str,
    windows,
    reconstruction: np.ndarray,
    basis_f: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    """Return the per-window absolute amplitude error of a reconstruction."""
    scores = score_reconstruction(
        name,
        windows.evaluation_observed,
        reconstruction,
        windows.evaluation_clean,
        windows.weights,
        basis_f,
        truth,
    )
    return np.abs(scores.amplitude_error)


def run_negative_controls(
    windows,
    noise_only_windows: np.ndarray,
    *,
    rank: int = 2,
    bootstrap_reps: int = 300,
    seed: int = 0,
    wrong_frequency_offset_bins: int = 37,
) -> dict[str, Any]:
    """Run all five controls and pair each against the nominal EMPCA arm.

    ``noise_only_windows`` must be signal-free windows disjoint from the
    evaluation set -- science-file stretches on real data, or the calibration
    noise in synthetic mode.
    """
    try:
        from ..cresst.detector_metrics import paired_bootstrap_stats
    except ImportError:  # pragma: no cover - src on path
        from noise_geometry.cresst.detector_metrics import paired_bootstrap_stats  # type: ignore

    rng = np.random.default_rng(int(seed))
    basis_f = windows.basis_rfft()
    truth = _truth_amplitude(windows.evaluation_clean, basis_f, windows.weights)
    w_real = _real_weights(windows.weights)
    fit_features, eval_features, to_time = _rfft_real_views(windows)
    rfft_to_real, _, _ = _canonical_helpers()

    # ---- nominal arm: the Phase 2 EMPCA reconstruction ---------------------
    components, mean = _fit_subspace(fit_features, rank, weights=w_real)
    nominal_reconstruction = to_time(
        _project(eval_features, components, mean, weights=w_real)
    )
    nominal_error = _score("empca", windows, nominal_reconstruction, basis_f, truth)

    errors: dict[str, np.ndarray] = {}

    # ---- 1. wrong PSD: a mismatched but still *coloured* noise model -------
    # Deliberately not the identity metric: the isotropic case is already the
    # Euclidean arm of Phase 2, so using it here would duplicate that contrast
    # rather than test anything new. Instead the estimated PSD is spectrally
    # flipped -- the same set of values assigned to the wrong frequencies --
    # which is the realistic failure (you have a noise model, it is just wrong)
    # and leaves the weight dynamic range unchanged.
    flipped_psd = np.asarray(windows.psd, dtype=np.float64)[::-1].copy()
    try:
        from ..noise import inverse_psd_weights
    except ImportError:  # pragma: no cover - src on path
        from noise_geometry.noise import inverse_psd_weights  # type: ignore

    flipped_weights = _real_weights(inverse_psd_weights(flipped_psd, windows.n_samples))
    wrong_components, wrong_mean = _fit_subspace(fit_features, rank, weights=flipped_weights)
    errors["wrong_psd"] = _score(
        "wrong_psd",
        windows,
        to_time(
            _project(eval_features, wrong_components, wrong_mean, weights=flipped_weights)
        ),
        basis_f,
        truth,
    )

    # ---- 2. noise-only training: no signal in the fit sample ---------------
    noise = np.atleast_2d(np.asarray(noise_only_windows, dtype=np.float64))
    if noise.shape[1] != windows.n_samples:
        raise ValueError("noise-only windows must have the same length as the analysis windows")
    noise_features = rfft_to_real(np.fft.rfft(noise, axis=1))
    noise_components, noise_mean = _fit_subspace(noise_features, rank, weights=w_real)
    errors["noise_only_train"] = _score(
        "noise_only_train",
        windows,
        to_time(_project(eval_features, noise_components, noise_mean, weights=w_real)),
        basis_f,
        truth,
    )

    # ---- 3. random weighted basis at matched rank --------------------------
    random_basis = _random_weighted_orthonormal_basis(
        rank, fit_features.shape[1], w_real, rng
    )
    errors["random_weighted_basis"] = _score(
        "random_weighted_basis",
        windows,
        to_time(_project(eval_features, random_basis, mean, weights=w_real)),
        basis_f,
        truth,
    )

    # ---- 4. wrong frequency: a filter where nothing was injected -----------
    # Scored against the *true* signal, so a filter looking in an empty part of
    # the spectrum must fail. Implemented as a fixed-phase rank-1 filter at an
    # offset bin, which is the cheapest thing a search could get wrong.
    offset_frequency = nearest_bin_frequency(
        windows.frequency_hz
        + int(wrong_frequency_offset_bins) * windows.sampling_frequency / windows.n_samples,
        windows.n_samples,
        windows.sampling_frequency,
    )
    wrong_basis_f = quadrature_templates_rfft(
        windows.n_samples, windows.sampling_frequency, offset_frequency
    )
    wrong_amplitude = np.asarray(
        fit_fixed_phase(
            np.fft.rfft(windows.evaluation_observed, axis=1), wrong_basis_f, windows.weights, phase=0.0
        )
    )
    errors["wrong_frequency"] = np.abs(wrong_amplitude - truth)

    # ---- 5. permuted bins: destroy the time structure ----------------------
    # The permutation is applied to the evaluation windows only, so the fitted
    # subspace and the noise model are untouched and the degradation can only
    # come from the destroyed structure.
    permuted = _permute_rows_independently(windows.evaluation_observed, rng)
    permuted_features = rfft_to_real(np.fft.rfft(permuted, axis=1))
    errors["permuted_bins"] = _score(
        "permuted_bins",
        windows,
        to_time(_project(permuted_features, components, mean, weights=w_real)),
        basis_f,
        truth,
    )

    contrasts, degraded = {}, {}
    for i, name in enumerate(CONTROLS):
        stats = paired_bootstrap_stats(
            errors[name], nominal_error, bootstrap_reps=int(bootstrap_reps), seed=int(seed) + i
        )
        contrasts[name] = stats
        # "Degraded" means the control's error is confidently *larger* than the
        # nominal one: the interval must exclude zero, not merely the mean.
        degraded[name] = bool(stats["exceeds_zero"])

    # An estimator that recovers nothing has |error| = |truth|. When every
    # control lands there, the controls confirm falsifiability but cannot rank
    # severity -- they have all saturated at zero recovery. Saying so beats
    # letting a reader infer an ordering from differences that are noise.
    null_rmse = float(np.sqrt(np.mean(np.asarray(truth) ** 2)))
    control_rmse = {name: float(np.sqrt(np.mean(err**2))) for name, err in errors.items()}
    saturated = {
        name: bool(abs(value - null_rmse) <= 0.05 * null_rmse)
        for name, value in control_rmse.items()
    }

    return {
        "rank": int(rank),
        "nominal": "empca",
        "nominal_amplitude_rmse": float(np.sqrt(np.mean(nominal_error**2))),
        "control_amplitude_rmse": control_rmse,
        "zero_recovery_rmse": null_rmse,
        "controls_saturated_at_zero_recovery": saturated,
        "all_controls_saturated": bool(all(saturated.values())),
        "saturation_note": (
            "A saturated control has collapsed to the zero-recovery level, so the "
            "controls establish that the pipeline is falsifiable but cannot be "
            "read as ranking severity. Raise the SNR if a severity ordering is "
            "wanted."
        ),
        "paired_contrasts": contrasts,
        "degraded": degraded,
        "all_controls_degraded": bool(all(degraded.values())),
        "controls_that_did_not_degrade": [name for name, ok in degraded.items() if not ok],
        "wrong_frequency_hz": float(offset_frequency),
        "wrong_frequency_offset_bins": int(wrong_frequency_offset_bins),
        "note": (
            "A control that fails to degrade is a reportable result, not a "
            "parameter to adjust. CRESST had one such case and published it."
        ),
    }
