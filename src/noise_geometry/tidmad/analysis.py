"""Phase 2 and Phase 3 analysis: metric reversal with truth, and the rank sweep.

Roadmap: ``docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md``.

What makes this stage different from CRESST
-------------------------------------------
CRESST can compare weighted residuals between methods, but it has no clean
signal, so it cannot answer the question the theory actually poses: *does the
reconstruction preferred by isotropic MSE recover the signal worse?* Here the
clean signal is known -- from an injected sinusoid in synthetic mode, and from
the release's reference channel on real data -- so every comparison carries a
third metric that neither of the other two can adjudicate:

    raw MSE            (what ordinary PCA optimizes)
    weighted residual  (what EMPCA optimizes; the Gaussian likelihood)
    error against the injected truth   <-- decides which one was right

Phase 3 adds the rank axis. The unknown phase of a sinusoid spans exactly
span{cos, sin}, so rank 1 is provably deficient, rank 2 is provably sufficient,
and rank >= 3 can only add noise directions. That prediction is closed-form,
which makes it a real test rather than a curve to admire.

Both phases operate on the same prepared window set, so the split, the noise
model, and the leakage audit are shared and cannot drift between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .estimators import fit_fixed_phase, fit_quadrature
from .splits import WindowSplit, chronological_window_split, leakage_audit
from .templates import quadrature_templates_rfft

__all__ = [
    "WindowSet",
    "ReconstructionScores",
    "score_reconstruction",
    "run_metric_reversal",
    "run_rank_sweep",
]


@dataclass(frozen=True)
class WindowSet:
    """Observed windows, their clean counterparts, and the fitted noise model.

    ``clean_*`` is the ground truth: the injected signal in synthetic mode, or
    the reference channel on real data. Everything downstream treats it as
    unavailable to the estimators and available only to the scorer, which is the
    property that makes the comparison meaningful.
    """

    fit_observed: np.ndarray
    fit_clean: np.ndarray
    evaluation_observed: np.ndarray
    evaluation_clean: np.ndarray
    psd: np.ndarray
    weights: np.ndarray
    sampling_frequency: float
    frequency_hz: float
    split: WindowSplit
    audit: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.evaluation_observed.shape[1])

    def basis_rfft(self) -> np.ndarray:
        return quadrature_templates_rfft(self.n_samples, self.sampling_frequency, self.frequency_hz)


@dataclass(frozen=True)
class ReconstructionScores:
    """Per-window scores for one reconstruction method."""

    name: str
    raw_mse: np.ndarray
    weighted_residual: np.ndarray
    clean_mse: np.ndarray
    amplitude_error: np.ndarray
    amplitude_estimate: np.ndarray

    def summary(self) -> dict[str, float]:
        return {
            "raw_mse_mean": float(np.mean(self.raw_mse)),
            "weighted_residual_mean": float(np.mean(self.weighted_residual)),
            "clean_mse_mean": float(np.mean(self.clean_mse)),
            "amplitude_rmse": float(np.sqrt(np.mean(self.amplitude_error**2))),
            "amplitude_bias": float(np.mean(self.amplitude_error)),
        }


def _weighted_residual_rfft(X_f: np.ndarray, R_f: np.ndarray, weights: np.ndarray) -> np.ndarray:
    residual = np.asarray(X_f) - np.asarray(R_f)
    return np.real(np.sum((np.abs(residual) ** 2) * np.asarray(weights)[None, :], axis=1))


def _fit_subspace(
    features: np.ndarray, rank: int, *, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(components, mean)`` of a rank-``k`` subspace fit.

    ``weights=None`` gives the ordinary Euclidean (isotropic) fit; passing the
    inverse-noise weights gives the covariance-weighted fit. Both route through
    the repository's subspace module so the two arms differ only in the metric.
    """
    try:
        from ..subspace import fit_pca, fit_weighted_pca
    except ImportError:  # pragma: no cover - src on path
        from noise_geometry.subspace import fit_pca, fit_weighted_pca  # type: ignore

    if weights is None:
        model = fit_pca(features, int(rank))
    else:
        model = fit_weighted_pca(features, weights, int(rank))
    return model.components, model.mean


def _project(
    features: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Project ``features`` onto a basis in the Euclidean or weighted metric."""
    try:
        from ..subspace import project_onto_basis
    except ImportError:  # pragma: no cover - src on path
        from noise_geometry.subspace import project_onto_basis  # type: ignore

    if weights is None:
        return project_onto_basis(features, components, mean=mean)
    return project_onto_basis(features, components, weights=weights, mean=mean)


def _canonical_helpers():
    try:
        from ...canonical.empca_equivalence_utils import (
            real_to_rfft,
            real_weight_vector,
            rfft_to_real,
        )
    except ImportError:  # pragma: no cover - src on path
        from canonical.empca_equivalence_utils import (  # type: ignore
            real_to_rfft,
            real_weight_vector,
            rfft_to_real,
        )
    return rfft_to_real, real_to_rfft, real_weight_vector


def _real_weights(weights: np.ndarray) -> np.ndarray:
    _, _, real_weight_vector = _canonical_helpers()
    return real_weight_vector(np.asarray(weights, dtype=np.float64))


def _rfft_real_views(windows: "WindowSet"):
    """Return ``(fit_features, eval_features, to_time)`` in the shared representation.

    The representation is the *unwhitened* real form of the rFFT, so that
    whitening enters only through the weight vector handed to the fit and the
    projection. ``to_time`` inverts the representation for scoring, which has to
    happen in the time domain because raw MSE and the clean-signal comparison
    are defined there.
    """
    rfft_to_real, real_to_rfft, _ = _canonical_helpers()
    n_samples = windows.n_samples
    n_bins = n_samples // 2 + 1

    fit_features = rfft_to_real(np.fft.rfft(windows.fit_observed, axis=1))
    eval_features = rfft_to_real(np.fft.rfft(windows.evaluation_observed, axis=1))

    def to_time(features: np.ndarray) -> np.ndarray:
        return np.fft.irfft(real_to_rfft(features, n_bins), n=n_samples, axis=1)

    return fit_features, eval_features, to_time


def score_reconstruction(
    name: str,
    observed: np.ndarray,
    reconstruction: np.ndarray,
    clean: np.ndarray,
    weights: np.ndarray,
    basis_f: np.ndarray,
    truth_amplitude: np.ndarray,
) -> ReconstructionScores:
    """Score one reconstruction on all three metric families.

    ``truth_amplitude`` is the projection of the *true* signal onto its own
    quadrature direction -- the quantity the estimator is trying to recover.
    Comparing along that fixed direction avoids the Rice bias that contaminates
    ``sqrt(a^2 + b^2)`` at low SNR, so the amplitude error stays interpretable
    at every SNR the sweep visits.
    """
    X_f = np.fft.rfft(observed, axis=1)
    R_f = np.fft.rfft(reconstruction, axis=1)

    fit = fit_quadrature(R_f, basis_f, weights)
    truth_fit = fit_quadrature(np.fft.rfft(clean, axis=1), basis_f, weights)
    direction = np.stack([truth_fit.cos_coefficient, truth_fit.sin_coefficient], axis=1)
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    direction = direction / norms
    estimate = fit.cos_coefficient * direction[:, 0] + fit.sin_coefficient * direction[:, 1]

    return ReconstructionScores(
        name=name,
        raw_mse=np.mean((observed - reconstruction) ** 2, axis=1),
        weighted_residual=_weighted_residual_rfft(X_f, R_f, weights),
        clean_mse=np.mean((clean - reconstruction) ** 2, axis=1),
        amplitude_error=estimate - np.asarray(truth_amplitude, dtype=np.float64),
        amplitude_estimate=estimate,
    )


def _truth_amplitude(clean: np.ndarray, basis_f: np.ndarray, weights: np.ndarray) -> np.ndarray:
    fit = fit_quadrature(np.fft.rfft(clean, axis=1), basis_f, weights)
    return np.hypot(fit.cos_coefficient, fit.sin_coefficient)


def _paired(a: np.ndarray, b: np.ndarray, *, reps: int, seed: int) -> dict:
    try:
        from ..cresst.detector_metrics import paired_bootstrap_stats
    except ImportError:  # pragma: no cover - src on path
        from noise_geometry.cresst.detector_metrics import paired_bootstrap_stats  # type: ignore

    return paired_bootstrap_stats(a, b, bootstrap_reps=int(reps), seed=int(seed))


def _block_bootstrap_mean_difference(
    a: np.ndarray, b: np.ndarray, *, block_size: int, reps: int, seed: int
) -> dict:
    """Moving-block bootstrap of a paired difference.

    Consecutive windows of a continuous stream are dependent, so the ordinary
    window-level bootstrap understates the uncertainty. Resampling contiguous
    blocks preserves the local correlation structure. Reported *alongside* the
    window-level interval rather than instead of it, because the discrepancy
    between the two is itself informative.
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = diff.size
    block = max(1, min(int(block_size), n))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts_max = max(1, n - block + 1)

    means = np.empty(int(reps), dtype=np.float64)
    for r in range(int(reps)):
        starts = rng.integers(0, starts_max, size=n_blocks)
        sample = np.concatenate([diff[s : s + block] for s in starts])[:n]
        means[r] = float(np.mean(sample))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "mean_difference": float(np.mean(diff)),
        "ci95_difference": (float(lo), float(hi)),
        "block_size": int(block),
        "bootstrap_reps": int(reps),
        "exceeds_zero": bool(lo > 0.0),
    }


def run_metric_reversal(
    windows: WindowSet,
    *,
    rank: int = 2,
    bootstrap_reps: int = 300,
    block_size: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Phase 2 (gate G2): does isotropic MSE pick the worse reconstruction?

    Two matched-rank subspaces are fitted on the *fit* windows -- one in the
    Euclidean geometry, one in the inverse-noise geometry -- and both are scored
    on held-out windows against the observed data, the likelihood, and the
    truth. The predicted pattern is that PCA wins raw MSE, EMPCA wins the
    weighted residual, and EMPCA wins on the truth. The third row is the claim;
    the first two only establish that the two arms really are optimizing
    different things.
    """
    basis_f = windows.basis_rfft()
    truth = _truth_amplitude(windows.evaluation_clean, basis_f, windows.weights)

    # Both arms are fitted on the *same* representation -- the faithful real
    # form of the rFFT -- and differ only in the metric used to fit and project.
    # This is exactly S5's design, and it is what makes the comparison a test of
    # the geometry rather than of the coordinate system: fitting one arm in the
    # time domain and the other in the frequency domain would confound the two.
    fit_features, eval_features, to_time = _rfft_real_views(windows)

    pca_components, pca_mean = _fit_subspace(fit_features, rank)
    pca_reconstruction = to_time(_project(eval_features, pca_components, pca_mean))

    w_real = _real_weights(windows.weights)
    empca_components, empca_mean = _fit_subspace(fit_features, rank, weights=w_real)
    empca_reconstruction = to_time(
        _project(eval_features, empca_components, empca_mean, weights=w_real)
    )

    scores = {
        "pca": score_reconstruction(
            "pca",
            windows.evaluation_observed,
            pca_reconstruction,
            windows.evaluation_clean,
            windows.weights,
            basis_f,
            truth,
        ),
        "empca": score_reconstruction(
            "empca",
            windows.evaluation_observed,
            empca_reconstruction,
            windows.evaluation_clean,
            windows.weights,
            windows.basis_rfft(),
            truth,
        ),
    }

    pca_s, empca_s = scores["pca"], scores["empca"]
    absolute_error = {
        name: np.abs(score.amplitude_error) for name, score in scores.items()
    }

    contrasts = {
        "raw_mse_pca_minus_empca": _paired(
            pca_s.raw_mse, empca_s.raw_mse, reps=bootstrap_reps, seed=seed
        ),
        "weighted_residual_pca_minus_empca": _paired(
            pca_s.weighted_residual, empca_s.weighted_residual, reps=bootstrap_reps, seed=seed + 1
        ),
        "amplitude_abs_error_pca_minus_empca": _paired(
            absolute_error["pca"], absolute_error["empca"], reps=bootstrap_reps, seed=seed + 2
        ),
        "clean_mse_pca_minus_empca": _paired(
            pca_s.clean_mse, empca_s.clean_mse, reps=bootstrap_reps, seed=seed + 3
        ),
    }
    block_contrasts = {
        "amplitude_abs_error_pca_minus_empca": _block_bootstrap_mean_difference(
            absolute_error["pca"],
            absolute_error["empca"],
            block_size=block_size,
            reps=bootstrap_reps,
            seed=seed + 4,
        ),
        "weighted_residual_pca_minus_empca": _block_bootstrap_mean_difference(
            pca_s.weighted_residual,
            empca_s.weighted_residual,
            block_size=block_size,
            reps=bootstrap_reps,
            seed=seed + 5,
        ),
    }

    reversal = {
        "raw_mse_favours_pca": bool(
            contrasts["raw_mse_pca_minus_empca"]["mean_difference"] < 0.0
        ),
        "weighted_residual_favours_empca": bool(
            contrasts["weighted_residual_pca_minus_empca"]["mean_difference"] > 0.0
        ),
        "truth_favours_empca": bool(
            contrasts["amplitude_abs_error_pca_minus_empca"]["mean_difference"] > 0.0
        ),
        "orderings_reversed": bool(
            contrasts["raw_mse_pca_minus_empca"]["mean_difference"] < 0.0
            and contrasts["weighted_residual_pca_minus_empca"]["mean_difference"] > 0.0
        ),
    }

    return {
        "rank": int(rank),
        "n_fit_windows": int(windows.fit_observed.shape[0]),
        "n_evaluation_windows": int(windows.evaluation_observed.shape[0]),
        "scores": {name: score.summary() for name, score in scores.items()},
        "paired_contrasts_window_level": contrasts,
        "paired_contrasts_block_level": block_contrasts,
        "direction_check": reversal,
        "leakage_audit": windows.audit,
    }


def run_rank_sweep(
    windows: WindowSet,
    *,
    ranks: tuple[int, ...] = (1, 2, 3, 4),
    bootstrap_reps: int = 300,
    saturation_tolerance: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Phase 3 (gate G3): the rank axis, against a closed-form prediction.

    For each rank the weighted subspace is fitted on the fit windows and scored
    on held-out windows. Also reported: the principal angles between the learned
    rank-2 subspace and the *analytic* quadrature subspace, and a single-template
    optimal filter at a fixed phase, which is the estimator a matched-filter
    pipeline uses when it assumes a phase it does not know.
    """
    try:
        from ..subspace import principal_angles
    except ImportError:  # pragma: no cover - src on path
        from noise_geometry.subspace import principal_angles  # type: ignore

    basis_f = windows.basis_rfft()
    truth = _truth_amplitude(windows.evaluation_clean, basis_f, windows.weights)
    w_real = _real_weights(windows.weights)
    fit_features, eval_features, to_time = _rfft_real_views(windows)

    rfft_to_real, _, _ = _canonical_helpers()
    analytic = rfft_to_real(basis_f)

    per_rank: dict[str, Any] = {}
    amplitude_abs_error: dict[int, np.ndarray] = {}
    for rank in ranks:
        components, mean = _fit_subspace(fit_features, rank, weights=w_real)
        reconstruction = to_time(_project(eval_features, components, mean, weights=w_real))
        score = score_reconstruction(
            f"rank{rank}",
            windows.evaluation_observed,
            reconstruction,
            windows.evaluation_clean,
            windows.weights,
            basis_f,
            truth,
        )
        amplitude_abs_error[int(rank)] = np.abs(score.amplitude_error)
        entry = score.summary()
        if int(rank) == 2:
            # Angles are measured in the noise metric: the subspaces are
            # compared where the likelihood lives, not in raw coordinates.
            angles = principal_angles(components, analytic, weights=w_real)
            entry["principal_angles_to_analytic_deg"] = [float(a) for a in angles]
            entry["max_principal_angle_to_analytic_deg"] = float(np.max(angles))
        per_rank[str(int(rank))] = entry

    # Fixed-phase rank-1 optimal filter: the same signal model a phase-assuming
    # pipeline would use. Scored on the same windows, against the same truth.
    X_f = np.fft.rfft(windows.evaluation_observed, axis=1)
    fixed_amp = np.asarray(fit_fixed_phase(X_f, basis_f, windows.weights, phase=0.0))
    fixed_error = np.abs(fixed_amp - truth)

    contrasts = {}
    if 1 in amplitude_abs_error and 2 in amplitude_abs_error:
        contrasts["amplitude_abs_error_rank1_minus_rank2"] = _paired(
            amplitude_abs_error[1], amplitude_abs_error[2], reps=bootstrap_reps, seed=seed
        )
    if 2 in amplitude_abs_error and 3 in amplitude_abs_error:
        contrasts["amplitude_abs_error_rank3_minus_rank2"] = _paired(
            amplitude_abs_error[3], amplitude_abs_error[2], reps=bootstrap_reps, seed=seed + 1
        )
    if 2 in amplitude_abs_error:
        contrasts["amplitude_abs_error_fixed_phase_minus_rank2"] = _paired(
            fixed_error, amplitude_abs_error[2], reps=bootstrap_reps, seed=seed + 2
        )

    rmse_by_rank = {
        int(k): float(np.sqrt(np.mean(v**2))) for k, v in amplitude_abs_error.items()
    }
    best_rank = min(rmse_by_rank, key=rmse_by_rank.get)
    # The raw argmin is not the scientific answer: once the curve has saturated,
    # which of the flat ranks happens to be lowest is decided by noise at the
    # fourth decimal. The reportable quantity is the smallest rank that is
    # within a tolerance of the minimum -- the knee -- which is what the
    # rank-2 prediction is actually about.
    floor = min(rmse_by_rank.values())
    knee = min(
        (r for r, v in rmse_by_rank.items() if v <= floor * (1.0 + float(saturation_tolerance))),
        default=best_rank,
    )
    return {
        "ranks": [int(r) for r in ranks],
        "per_rank": per_rank,
        "amplitude_rmse_by_rank": rmse_by_rank,
        "smallest_rank_within_tolerance": int(knee),
        "saturation_tolerance": float(saturation_tolerance),
        "fixed_phase_rank1": {
            "amplitude_rmse": float(np.sqrt(np.mean((fixed_amp - truth) ** 2))),
            "amplitude_bias": float(np.mean(fixed_amp - truth)),
        },
        "paired_contrasts": contrasts,
        "best_rank_by_amplitude_rmse": int(best_rank),
        "prediction": (
            "rank 2 is sufficient and rank 1 is deficient because the unknown "
            "phase of a sinusoid spans exactly span{cos, sin}"
        ),
    }


def build_window_set(
    observed: np.ndarray,
    clean: np.ndarray,
    psd: np.ndarray,
    weights: np.ndarray,
    sampling_frequency: float,
    frequency_hz: float,
    *,
    fit_fraction: float = 0.7,
    guard_windows: int = 4,
    stride: int | None = None,
) -> WindowSet:
    """Split observed/clean window pairs chronologically, with a leakage audit."""
    X = np.atleast_2d(np.asarray(observed, dtype=np.float64))
    C = np.atleast_2d(np.asarray(clean, dtype=np.float64))
    if X.shape != C.shape:
        raise ValueError("observed and clean windows must have the same shape")

    split = chronological_window_split(
        X.shape[0], fit_fraction=fit_fraction, guard_windows=guard_windows
    )
    audit = leakage_audit(
        split,
        fit_windows=X[split.fit_indices],
        evaluation_windows=X[split.evaluation_indices],
        window_samples=X.shape[1],
        stride=stride if stride is not None else X.shape[1],
        sampling_frequency=sampling_frequency,
    )
    return WindowSet(
        fit_observed=X[split.fit_indices],
        fit_clean=C[split.fit_indices],
        evaluation_observed=X[split.evaluation_indices],
        evaluation_clean=C[split.evaluation_indices],
        psd=np.asarray(psd, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        sampling_frequency=float(sampling_frequency),
        frequency_hz=float(frequency_hz),
        split=split,
        audit=audit,
    )
