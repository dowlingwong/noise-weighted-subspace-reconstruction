"""Detector-facing evaluation metrics for the CRESST validation protocol.

See ``docs/EXPERIMENT_PROTOCOLS.md`` (Metrics / CRESST Validation Protocol):
residual whitening ratio vs. frequency, detection score by noise-weighted
chi2 reduction, trigger efficiency vs. injected amplitude at fixed background
FPR (threshold set on background only), 50%/90% efficiency thresholds,
amplitude RMSE/bias, and bootstrap 95% intervals.
"""

from __future__ import annotations

import numpy as np

from ..metrics import amplitude_resolution, weighted_residual
from ..noise import estimate_psd_rfft
from ..subspace import project_onto_basis
from .injections import inject_direct, inject_measured_psd


def residual_whitening_ratio(
    residuals: np.ndarray,
    noise_traces: np.ndarray,
    *,
    sampling_frequency: float = 1.0,
) -> dict[str, np.ndarray]:
    """``mean(|r_f|^2) / PSD_f`` vs. frequency; near 1 under correct calibration."""
    freqs, resid_psd = estimate_psd_rfft(residuals, sampling_frequency)
    _, noise_psd = estimate_psd_rfft(noise_traces, sampling_frequency)
    ratio = resid_psd / np.maximum(noise_psd, np.finfo(float).tiny)
    return {"frequency_hz": freqs, "whitening_ratio": ratio}


def detection_score(test: np.ndarray, recon: np.ndarray, metric: np.ndarray) -> np.ndarray:
    """Per-trace ``chi2(noise-only) - chi2(signal-subspace)``."""
    test = np.asarray(test, dtype=np.float64)
    chi2_null = weighted_residual(test, np.zeros_like(test), metric, normalize=False)
    chi2_signal = weighted_residual(test, recon, metric, normalize=False)
    return chi2_null - chi2_signal


def bootstrap_ci(values: np.ndarray, *, level: float = 0.95, n_boot: int = 300, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(means, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return (float(lo), float(hi))


def paired_bootstrap_stats(
    a: np.ndarray, b: np.ndarray, *, bootstrap_reps: int = 300, seed: int = 0
) -> dict:
    """Paired ``a - b`` bootstrap: mean difference, 95% CI, and CI-based verdict.

    Used everywhere a gate compares two methods (or a control vs. nominal) on
    the *same* held-out events: a raw point-estimate win/loss is not evidence
    on its own (Reporting Standard #5 -- paired comparisons with uncertainty).
    ``exceeds_zero`` requires the CI's lower bound to be strictly positive,
    i.e. ``a`` is confidently greater than ``b``.
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    lo, hi = bootstrap_ci(diff, level=0.95, n_boot=bootstrap_reps, seed=seed)
    return {
        "mean_difference": float(np.mean(diff)),
        "ci95_difference": (lo, hi),
        "bootstrap_reps": int(bootstrap_reps),
        "exceeds_zero": bool(lo > 0),
        "n_events": int(diff.size),
    }


def background_thresholds(background_scores: np.ndarray, fpr_points: list[float]) -> dict[float, float]:
    """Detection-score thresholds at each FPR, set on background-only scores."""
    scores = np.asarray(background_scores, dtype=np.float64)
    return {float(p): float(np.quantile(scores, 1.0 - p)) for p in fpr_points}


def background_threshold_ci(
    background_scores: np.ndarray, fpr_points: list[float], *, bootstrap_reps: int = 300, seed: int = 0
) -> dict[float, tuple[float, float]]:
    """Bootstrap 95% CI on the background-only *threshold itself* (not its mean).

    With a per-channel background pool of only ~hundreds of traces, a tight
    FPR point (e.g. 0.1%) is set by the top 1-2 order statistics and the
    threshold estimate carries substantial sampling uncertainty; this makes
    that uncertainty explicit rather than reporting a bare point estimate.
    """
    scores = np.asarray(background_scores, dtype=np.float64)
    n = scores.size
    if n < 2:
        return {float(p): (float("nan"), float("nan")) for p in fpr_points}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(bootstrap_reps), n))
    resampled = scores[idx]
    out = {}
    for p in fpr_points:
        boot_thresholds = np.quantile(resampled, 1.0 - p, axis=1)
        lo, hi = np.percentile(boot_thresholds, [2.5, 97.5])
        out[float(p)] = (float(lo), float(hi))
    return out


def efficiency_curve(
    scores_by_amplitude: list[np.ndarray],
    threshold: float,
) -> np.ndarray:
    """Fraction of injected traces above ``threshold``, one entry per amplitude."""
    return np.array([float(np.mean(scores >= threshold)) for scores in scores_by_amplitude])


def efficiency_amplitude_thresholds(
    amplitude_grid: np.ndarray,
    efficiency: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.9),
) -> dict[float, float]:
    """Interpolate the amplitude at which the efficiency curve crosses each level."""
    amplitude_grid = np.asarray(amplitude_grid, dtype=np.float64)
    efficiency = np.asarray(efficiency, dtype=np.float64)
    out = {}
    for level in levels:
        if efficiency[-1] < level:
            out[float(level)] = float("nan")
        else:
            out[float(level)] = float(np.interp(level, efficiency, amplitude_grid))
    return out


def amplitude_error_summary(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Amplitude bias/std/RMSE at a fixed injected amplitude (see ``amplitude_resolution``)."""
    return amplitude_resolution(estimate, truth)


def _condition_curves(
    background: np.ndarray,
    template: np.ndarray,
    amplitude_grid: np.ndarray,
    n_injections: int,
    components: np.ndarray,
    mean: np.ndarray,
    metric: np.ndarray,
    thresholds: dict[float, float],
    fpr_points: list[float],
    bootstrap_reps: int,
    rng: np.random.Generator,
    *,
    measured_psd: bool,
) -> dict:
    efficiency = {float(p): [] for p in fpr_points}
    amplitude_bias, amplitude_rmse = [], []
    amplitude_bias_ci95, amplitude_rmse_ci95 = [], []
    for amp in amplitude_grid:
        amplitudes = np.full(int(n_injections), float(amp))
        if measured_psd:
            injected = inject_measured_psd(background, template, amplitudes, rng)
        else:
            injected, _ = inject_direct(background, template, amplitudes, rng)
        recon = project_onto_basis(injected, components, weights=metric, mean=mean)
        scores = detection_score(injected, recon, metric)
        for p in fpr_points:
            efficiency[float(p)].append(float(np.mean(scores >= thresholds[float(p)])))

        m_template = metric @ template
        denom = float(template @ m_template)
        estimate = (injected @ m_template) / denom if denom > 0 else np.zeros_like(amplitudes)
        err = estimate - amplitudes
        amplitude_bias.append(float(np.mean(err)))
        amplitude_rmse.append(float(np.sqrt(np.mean(err**2))))
        bias_lo, bias_hi = bootstrap_ci(err, n_boot=bootstrap_reps, seed=rng.integers(0, 2**31 - 1))
        rmse_lo, rmse_hi = bootstrap_ci(err**2, n_boot=bootstrap_reps, seed=rng.integers(0, 2**31 - 1))
        amplitude_bias_ci95.append((bias_lo, bias_hi))
        amplitude_rmse_ci95.append(
            (float(np.sqrt(max(rmse_lo, 0.0))), float(np.sqrt(max(rmse_hi, 0.0))))
        )

    efficiency_thresholds = {
        float(p): efficiency_amplitude_thresholds(amplitude_grid, np.asarray(efficiency[float(p)]))
        for p in fpr_points
    }
    return {
        "efficiency": efficiency,
        "efficiency_amplitude_thresholds": efficiency_thresholds,
        "amplitude_bias": amplitude_bias,
        "amplitude_rmse": amplitude_rmse,
        "amplitude_bias_ci95": amplitude_bias_ci95,
        "amplitude_rmse_ci95": amplitude_rmse_ci95,
    }


def evaluate_trigger_efficiency(
    background: np.ndarray,
    template: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    metric: np.ndarray,
    *,
    fpr_points: list[float],
    amplitude_grid: np.ndarray,
    n_injections: int = 200,
    bootstrap_reps: int = 300,
    seed: int = 22,
) -> dict:
    """Full trigger-efficiency evaluation: direct injection (primary) and the
    measured-PSD Gaussian condition (secondary, explicitly labeled), with
    background-only FPR thresholds, 50%/90% efficiency amplitude thresholds,
    amplitude bias/RMSE, and bootstrap 95% CIs.
    """
    rng = np.random.default_rng(int(seed))
    background = np.asarray(background, dtype=np.float64)
    recon_background = project_onto_basis(background, components, weights=metric, mean=mean)
    background_scores = detection_score(background, recon_background, metric)
    thresholds = background_thresholds(background_scores, fpr_points)
    threshold_ci = background_threshold_ci(background_scores, fpr_points, bootstrap_reps=bootstrap_reps, seed=seed)

    direct = _condition_curves(
        background, template, amplitude_grid, n_injections, components, mean, metric,
        thresholds, fpr_points, bootstrap_reps, rng, measured_psd=False,
    )
    measured_psd = _condition_curves(
        background, template, amplitude_grid, n_injections, components, mean, metric,
        thresholds, fpr_points, bootstrap_reps, rng, measured_psd=True,
    )

    degradation_50pct = {}
    for p in fpr_points:
        a_direct = direct["efficiency_amplitude_thresholds"][float(p)][0.5]
        a_measured = measured_psd["efficiency_amplitude_thresholds"][float(p)][0.5]
        degradation_50pct[float(p)] = (
            float(a_measured - a_direct) if np.isfinite(a_direct) and np.isfinite(a_measured) else float("nan")
        )

    return {
        "amplitude_grid": [float(a) for a in amplitude_grid],
        "fpr_points": [float(p) for p in fpr_points],
        "background_thresholds": thresholds,
        "background_threshold_ci95": threshold_ci,
        "n_background": int(background.shape[0]),
        "n_injections_per_amplitude": int(n_injections),
        "bootstrap_reps": int(bootstrap_reps),
        "direct_injection": direct,
        "measured_psd_injection": measured_psd,
        "measured_psd_minus_direct_50pct_amplitude_threshold": degradation_50pct,
    }
