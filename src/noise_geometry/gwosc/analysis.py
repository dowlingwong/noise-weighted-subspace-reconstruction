"""GWOSC cached-event likelihood-geometry analysis."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import chirp, welch
from scipy.signal.windows import tukey

from ..filters import gls_amplitude, psd_amplitude_variance
from ..metrics import mse, weighted_residual
from ..noise import estimate_psd_ensemble, inverse_psd_weights, regularize_psd
from ..utils.paths import dataset_root
from .reference import run_gwpy_reference_check


def load_cached_event(path: str | Path) -> dict[str, Any]:
    """Load a GWOSC event window written by the download helper."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _offsource_windows(
    values: np.ndarray,
    center: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    guard = length // 2
    windows = []
    starts = []
    for start in range(0, values.size - length + 1, length):
        stop = start + length
        if stop < center - guard or start > center + guard:
            windows.append(values[start:stop])
            starts.append(start)
    if not windows:
        raise ValueError("downloaded window is too short for independent off-source PSD windows")
    return np.asarray(windows), np.asarray(starts, dtype=np.int64)


def _split_offsource_windows(
    windows: np.ndarray,
    *,
    calibration_fraction: float,
    seed: int,
    minimum_calibration_windows: int,
    minimum_evaluation_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint calibration and evaluation indices."""
    n_windows = int(windows.shape[0])
    if n_windows < 2:
        raise ValueError("at least two off-source windows are required")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("psd_calibration_fraction must be in (0, 1)")
    minimum_calibration_windows = int(minimum_calibration_windows)
    minimum_evaluation_windows = int(minimum_evaluation_windows)
    if minimum_calibration_windows < 1 or minimum_evaluation_windows < 1:
        raise ValueError("minimum calibration/evaluation windows must be positive")
    required = minimum_calibration_windows + minimum_evaluation_windows
    if n_windows < required:
        raise ValueError(
            f"need at least {required} off-source windows "
            f"({minimum_calibration_windows} calibration + "
            f"{minimum_evaluation_windows} evaluation), found {n_windows}"
        )

    n_calibration = int(round(calibration_fraction * n_windows))
    n_calibration = max(minimum_calibration_windows, n_calibration)
    n_calibration = min(n_windows - minimum_evaluation_windows, n_calibration)
    permutation = np.random.default_rng(int(seed)).permutation(n_windows)
    calibration_indices = np.sort(permutation[:n_calibration])
    evaluation_indices = np.sort(permutation[n_calibration:])
    return calibration_indices, evaluation_indices


def _cached_duration_from_name(path: Path) -> float:
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)s\.npz$", path.name)
    return float(match.group(1)) if match else -1.0


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    if mad <= np.finfo(float).eps:
        return np.zeros_like(array)
    return 0.6744897501960817 * (array - median) / mad


def _calibration_window_quality(
    windows: np.ndarray,
    candidate_indices: np.ndarray,
    sampling_frequency: float,
    *,
    psd_window: str,
    psd_detrend: str | bool,
    highpass_hz: float | None,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Diagnose and exclude glitch-like PSD calibration windows."""
    candidates = np.asarray(windows[candidate_indices], dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[0] < 1:
        raise ValueError("calibration candidates must be a non-empty 2D array")

    detrended = candidates - np.mean(candidates, axis=1, keepdims=True)
    time_rms = np.sqrt(np.mean(detrended**2, axis=1))
    peak_abs = np.max(np.abs(detrended), axis=1)
    crest_factor = peak_abs / np.maximum(time_rms, np.finfo(float).tiny)
    frequencies, periodograms = welch(
        candidates,
        fs=float(sampling_frequency),
        window=psd_window,
        nperseg=candidates.shape[1],
        noverlap=0,
        nfft=candidates.shape[1],
        detrend=psd_detrend,
        return_onesided=True,
        scaling="density",
        average="mean",
        axis=-1,
    )
    band_min = float(config.get("band_min_hz", highpass_hz or 0.0))
    band_max = float(
        config.get("band_max_hz", min(512.0, float(sampling_frequency) / 2.0))
    )
    if not 0.0 <= band_min < band_max <= float(sampling_frequency) / 2.0:
        raise ValueError("PSD quality band must lie within [0, Nyquist]")
    band_mask = (frequencies >= band_min) & (frequencies <= band_max)
    if not np.any(band_mask):
        raise ValueError("PSD quality band contains no frequency bins")
    df = float(frequencies[1] - frequencies[0])
    band_power = np.sum(periodograms[:, band_mask], axis=1) * df

    log_rms_z = _robust_zscore(np.log(np.maximum(time_rms, np.finfo(float).tiny)))
    log_band_power_z = _robust_zscore(
        np.log(np.maximum(band_power, np.finfo(float).tiny))
    )
    robust_z_threshold = float(config.get("robust_z_threshold", 6.0))
    crest_factor_threshold = float(config.get("crest_factor_threshold", 20.0))
    enabled = bool(config.get("enabled", True))

    reasons: list[list[str]] = []
    keep = np.ones(candidates.shape[0], dtype=bool)
    for index in range(candidates.shape[0]):
        current = []
        if abs(log_rms_z[index]) > robust_z_threshold:
            current.append("time_rms_robust_z")
        if abs(log_band_power_z[index]) > robust_z_threshold:
            current.append("band_power_robust_z")
        if crest_factor[index] > crest_factor_threshold:
            current.append("crest_factor")
        reasons.append(current)
        if enabled and current:
            keep[index] = False

    rejected_fraction = float(np.mean(~keep))
    max_rejected_fraction = float(config.get("max_rejected_fraction", 0.25))
    if rejected_fraction > max_rejected_fraction:
        raise ValueError(
            "calibration quality rejected "
            f"{rejected_fraction:.1%} of windows, above configured "
            f"{max_rejected_fraction:.1%}"
        )

    records = []
    for local_index, global_index in enumerate(candidate_indices):
        records.append(
            {
                "window_index": int(global_index),
                "accepted": bool(keep[local_index]),
                "reasons": reasons[local_index],
                "time_rms": float(time_rms[local_index]),
                "time_rms_robust_z": float(log_rms_z[local_index]),
                "crest_factor": float(crest_factor[local_index]),
                "band_power": float(band_power[local_index]),
                "band_power_robust_z": float(log_band_power_z[local_index]),
            }
        )
    diagnostics = {
        "enabled": enabled,
        "band_hz": [band_min, band_max],
        "robust_z_threshold": robust_z_threshold,
        "crest_factor_threshold": crest_factor_threshold,
        "max_rejected_fraction": max_rejected_fraction,
        "n_candidates": int(candidates.shape[0]),
        "n_accepted": int(np.sum(keep)),
        "n_rejected": int(np.sum(~keep)),
        "rejected_fraction": rejected_fraction,
        "accepted_window_indices": candidate_indices[keep].tolist(),
        "rejected_window_indices": candidate_indices[~keep].tolist(),
        "windows": records,
    }
    return keep, diagnostics


def _fit_noise_model(
    offsource: np.ndarray,
    template_f: np.ndarray,
    sampling_frequency: float,
    *,
    split_seed: int,
    calibration_fraction: float,
    minimum_calibration_windows: int,
    minimum_evaluation_windows: int,
    psd_floor_fraction: float,
    highpass_hz: float | None,
    psd_config: dict[str, Any],
    quality_config: dict[str, Any],
) -> dict[str, Any]:
    candidate_indices, evaluation_indices = _split_offsource_windows(
        offsource,
        calibration_fraction=calibration_fraction,
        seed=split_seed,
        minimum_calibration_windows=minimum_calibration_windows,
        minimum_evaluation_windows=minimum_evaluation_windows,
    )
    psd_window = str(psd_config.get("window", "hann"))
    psd_average = str(psd_config.get("average", "median"))
    psd_detrend = psd_config.get("detrend", "constant")
    quality_keep, quality = _calibration_window_quality(
        offsource,
        candidate_indices,
        sampling_frequency,
        psd_window=psd_window,
        psd_detrend=psd_detrend,
        highpass_hz=highpass_hz,
        config=quality_config,
    )
    calibration_indices = candidate_indices[quality_keep]
    if calibration_indices.size < int(minimum_calibration_windows):
        raise ValueError(
            "calibration quality left "
            f"{calibration_indices.size} windows, below required "
            f"{minimum_calibration_windows}"
        )
    calibration = offsource[calibration_indices]
    evaluation = offsource[evaluation_indices]
    _, psd = estimate_psd_ensemble(
        calibration,
        sampling_frequency,
        window=psd_window,
        average=psd_average,
        detrend=psd_detrend,
    )
    psd = regularize_psd(psd, floor_fraction=psd_floor_fraction)
    trace_len = int(offsource.shape[1])
    weights = inverse_psd_weights(psd, trace_len)
    if highpass_hz is not None:
        frequencies = np.fft.rfftfreq(trace_len, d=1.0 / sampling_frequency)
        weights = weights.copy()
        weights[frequencies < highpass_hz] = 0.0
    amplitude_sigma = float(
        np.sqrt(
            psd_amplitude_variance(
                template_f,
                psd,
                weights,
                trace_len=trace_len,
                sampling_frequency=sampling_frequency,
            )
        )
    )
    if not np.isfinite(amplitude_sigma) or amplitude_sigma <= 0:
        raise ValueError("template amplitude variance is not positive and finite")
    evaluation_f = np.fft.rfft(evaluation, axis=1)
    null_amp = np.asarray(gls_amplitude(evaluation_f, template_f, weights))
    null_amplitude_std = (
        float(np.std(null_amp, ddof=1)) if null_amp.size > 1 else 0.0
    )
    null_ratio = (
        null_amplitude_std / amplitude_sigma if null_amp.size > 1 else float("nan")
    )
    return {
        "split_seed": int(split_seed),
        "candidate_calibration_indices": candidate_indices,
        "calibration_indices": calibration_indices,
        "evaluation_indices": evaluation_indices,
        "calibration": calibration,
        "evaluation": evaluation,
        "psd": psd,
        "weights": weights,
        "amplitude_sigma": amplitude_sigma,
        "evaluation_f": evaluation_f,
        "null_amp": null_amp,
        "null_amplitude_std": null_amplitude_std,
        "null_sigma_over_predicted": null_ratio,
        "quality": quality,
        "psd_estimator": {
            "window": psd_window,
            "average": psd_average,
            "detrend": psd_detrend,
            "median_bias_corrected": psd_average == "median",
        },
    }


def _approximate_chirp_template(length: int, sample_rate: float) -> np.ndarray:
    duration = min(0.5, length / sample_rate * 0.5)
    n_chirp = max(16, min(length, int(duration * sample_rate)))
    t = np.arange(n_chirp) / sample_rate
    nyquist = sample_rate / 2.0
    waveform = chirp(t, f0=min(35.0, nyquist * 0.15), f1=min(250.0, nyquist * 0.8), t1=duration)
    waveform *= tukey(n_chirp, alpha=0.4)
    template = np.zeros(length)
    stop = length // 2 + 1
    start = max(0, stop - n_chirp)
    template[start:stop] = waveform[-(stop - start) :]
    norm = np.max(np.abs(template))
    return template / max(norm, np.finfo(float).eps)


def run_gwosc_experiment(config: dict[str, Any], data_root_path: str | Path) -> dict[str, Any]:
    """Run event-centered and off-source injection checks on cached strain."""
    root = dataset_root("gwosc", data_root_path)
    event = str(config.get("event", "GW150914"))
    detectors = list(config.get("detectors", ["H1", "L1"]))
    analysis_duration = float(config.get("analysis_duration_seconds", 4.0))
    target_snr = float(config.get("injection_snr", 8.0))
    calibration_fraction = float(config.get("psd_calibration_fraction", 0.75))
    split_seed = int(config.get("offsource_split_seed", 0))
    minimum_calibration_windows = int(config.get("minimum_psd_calibration_windows", 1))
    minimum_evaluation_windows = int(config.get("minimum_evaluation_windows", 1))
    psd_floor_fraction = float(config.get("psd_floor_fraction", 1e-6))
    psd_config = dict(config.get("psd_estimator", {}))
    quality_config = dict(config.get("psd_quality", {}))
    null_validation_config = dict(config.get("null_calibration_validation", {}))
    validation_split_seeds = [
        int(seed)
        for seed in null_validation_config.get("split_seeds", [split_seed])
    ]
    if split_seed not in validation_split_seeds:
        validation_split_seeds.insert(0, split_seed)
    per_split_bounds = [
        float(value)
        for value in null_validation_config.get(
            "per_split_ratio_bounds",
            [0.5, 1.5],
        )
    ]
    median_bounds = [
        float(value)
        for value in null_validation_config.get(
            "median_ratio_bounds",
            [0.8, 1.2],
        )
    ]
    if len(per_split_bounds) != 2 or len(median_bounds) != 2:
        raise ValueError("null calibration bounds must contain [lower, upper]")
    reference_config = config.get("gwpy_reference", {})
    highpass_hz = config.get(
        "analysis_highpass_hz",
        reference_config.get("highpass_hz"),
    )
    if highpass_hz is not None:
        highpass_hz = float(highpass_hz)
    detector_metrics: dict[str, Any] = {}

    for detector in detectors:
        candidates = sorted((root / "raw" / event).glob(f"{event}_{detector}_*.npz"))
        if not candidates:
            raise FileNotFoundError(
                f"No cached {event} {detector} data under {root / 'raw' / event}. "
                "Run scripts/download/download_gwosc.py --download first."
            )
        cache_file = max(candidates, key=_cached_duration_from_name)
        cached = load_cached_event(cache_file)
        values = np.asarray(cached["value"], dtype=np.float64)
        times = np.asarray(cached["times"], dtype=np.float64)
        sample_rate = float(np.asarray(cached["sample_rate"]))
        gps = float(np.asarray(cached["gps"]))
        length = int(round(analysis_duration * sample_rate))
        length = min(length, values.size // 3)
        center = int(np.argmin(np.abs(times - gps)))
        start = center - length // 2
        stop = start + length
        if start < 0 or stop > values.size:
            raise ValueError("event is too close to the edge of the cached strain window")
        event_trace = values[start:stop]
        offsource, offsource_starts = _offsource_windows(values, center, length)
        template = _approximate_chirp_template(length, sample_rate)
        template_f = np.fft.rfft(template)
        noise_models = {}
        for validation_seed in validation_split_seeds:
            noise_models[validation_seed] = _fit_noise_model(
                offsource,
                template_f,
                sample_rate,
                split_seed=validation_seed,
                calibration_fraction=calibration_fraction,
                minimum_calibration_windows=minimum_calibration_windows,
                minimum_evaluation_windows=minimum_evaluation_windows,
                psd_floor_fraction=psd_floor_fraction,
                highpass_hz=highpass_hz,
                psd_config=psd_config,
                quality_config=quality_config,
            )
        model = noise_models[split_seed]
        calibration_indices = model["calibration_indices"]
        evaluation_indices = model["evaluation_indices"]
        calibration = model["calibration"]
        evaluation = model["evaluation"]
        psd = model["psd"]
        weights = model["weights"]
        amplitude_sigma = float(model["amplitude_sigma"])
        evaluation_f = model["evaluation_f"]
        null_amp = model["null_amp"]
        null_amplitude_std = float(model["null_amplitude_std"])

        split_results = []
        for validation_seed in validation_split_seeds:
            validation_model = noise_models[validation_seed]
            ratio = float(validation_model["null_sigma_over_predicted"])
            split_passed = (
                np.isfinite(ratio)
                and per_split_bounds[0] <= ratio <= per_split_bounds[1]
            )
            split_results.append(
                {
                    "split_seed": validation_seed,
                    "null_sigma_over_predicted": ratio,
                    "passed": bool(split_passed),
                    "n_calibration_candidates": int(
                        validation_model["candidate_calibration_indices"].size
                    ),
                    "n_calibration_windows": int(
                        validation_model["calibration_indices"].size
                    ),
                    "n_evaluation_windows": int(
                        validation_model["evaluation_indices"].size
                    ),
                    "rejected_calibration_windows": validation_model["quality"][
                        "rejected_window_indices"
                    ],
                }
            )
        split_ratios = np.asarray(
            [result["null_sigma_over_predicted"] for result in split_results],
            dtype=np.float64,
        )
        median_ratio = float(np.median(split_ratios))
        null_gate_passed = bool(
            all(result["passed"] for result in split_results)
            and median_bounds[0] <= median_ratio <= median_bounds[1]
        )
        null_calibration_validation = {
            "passed": null_gate_passed,
            "n_split_seeds": len(validation_split_seeds),
            "split_seeds": validation_split_seeds,
            "per_split_ratio_bounds": per_split_bounds,
            "median_ratio_bounds": median_bounds,
            "median_null_sigma_over_predicted": median_ratio,
            "min_null_sigma_over_predicted": float(np.min(split_ratios)),
            "max_null_sigma_over_predicted": float(np.max(split_ratios)),
            "splits": split_results,
        }
        event_f = np.fft.rfft(event_trace)
        event_amp = float(gls_amplitude(event_f, template_f, weights))
        event_recon_f = event_amp * template_f

        injection_amp = target_snr * amplitude_sigma
        injection_f = evaluation_f + injection_amp * template_f[None, :]
        recovered_amp = gls_amplitude(injection_f, template_f, weights)
        unpaired_recovered_score = np.asarray(recovered_amp) / amplitude_sigma
        null_score = null_amp / amplitude_sigma
        paired_recovered_amp = np.asarray(recovered_amp) - null_amp
        paired_recovered_score = paired_recovered_amp / amplitude_sigma
        detector_metrics[detector] = {
            "cache_file": str(cache_file),
            "sample_rate": sample_rate,
            "analysis_duration_seconds": length / sample_rate,
            "n_offsource_windows": int(offsource.shape[0]),
            "n_psd_calibration_windows": int(calibration.shape[0]),
            "n_psd_calibration_candidates": int(
                model["candidate_calibration_indices"].size
            ),
            "n_injection_windows": int(evaluation.shape[0]),
            "psd_calibration_fraction": calibration_fraction,
            "offsource_split_seed": split_seed,
            "psd_calibration_window_indices": calibration_indices.tolist(),
            "evaluation_window_indices": evaluation_indices.tolist(),
            "psd_calibration_window_starts_seconds_from_cache_start": (
                offsource_starts[calibration_indices] / sample_rate
            ).tolist(),
            "evaluation_window_starts_seconds_from_cache_start": (
                offsource_starts[evaluation_indices] / sample_rate
            ).tolist(),
            "analysis_highpass_hz": highpass_hz,
            "psd_estimator": model["psd_estimator"],
            "psd_calibration_quality": model["quality"],
            "null_calibration_validation": null_calibration_validation,
            "event_amplitude": event_amp,
            "event_amplitude_sigma": amplitude_sigma,
            "event_matched_filter_score": event_amp / amplitude_sigma,
            "event_raw_mse": float(mse(event_f, event_recon_f)),
            "event_weighted_residual": float(weighted_residual(event_f, event_recon_f, weights)),
            "injection_target_snr": target_snr,
            "injection_amplitude": injection_amp,
            "injection_amplitude_bias": float(
                np.mean(paired_recovered_amp - injection_amp)
            ),
            "injection_unpaired_amplitude_offset": float(
                np.mean(recovered_amp - injection_amp)
            ),
            "injection_recovered_amplitude_mean": float(np.mean(recovered_amp)),
            "injection_recovered_amplitude_std": (
                float(np.std(recovered_amp, ddof=1))
                if recovered_amp.size > 1
                else 0.0
            ),
            "injection_score_mean": float(np.mean(paired_recovered_score)),
            "injection_score_std": (
                float(np.std(paired_recovered_score, ddof=1))
                if paired_recovered_score.size > 1
                else 0.0
            ),
            "injection_unpaired_score_mean": float(
                np.mean(unpaired_recovered_score)
            ),
            "injection_unpaired_score_std": (
                float(np.std(unpaired_recovered_score, ddof=1))
                if unpaired_recovered_score.size > 1
                else 0.0
            ),
            "injection_paired_score_mean": float(np.mean(paired_recovered_score)),
            "injection_paired_score_std": (
                float(np.std(paired_recovered_score, ddof=1))
                if paired_recovered_score.size > 1
                else 0.0
            ),
            "injection_paired_score_bias": float(
                np.mean(paired_recovered_score) - target_snr
            ),
            "null_amplitude_mean": float(np.mean(null_amp)),
            "null_amplitude_std": null_amplitude_std,
            "null_score_mean": float(np.mean(null_score)),
            "null_score_std": (
                float(np.std(null_score, ddof=1))
                if null_score.size > 1
                else 0.0
            ),
            "null_sigma_over_predicted": (
                null_amplitude_std / amplitude_sigma if null_amp.size > 1 else 0.0
            ),
            "injection_empirical_snr": (
                injection_amp / null_amplitude_std
                if null_amplitude_std > 0
                else None
            ),
        }
        if bool(reference_config.get("enabled", False)):
            detector_metrics[detector]["gwpy_reference"] = run_gwpy_reference_check(
                calibration,
                evaluation,
                sample_rate,
                fduration_seconds=float(reference_config.get("fduration_seconds", 1.0)),
                highpass_hz=highpass_hz,
                detrend=str(reference_config.get("detrend", "constant")),
                floor_fraction=psd_floor_fraction,
                psd_window=str(model["psd_estimator"]["window"]),
                psd_average=str(model["psd_estimator"]["average"]),
                psd_detrend=model["psd_estimator"]["detrend"],
            )

    detector_acceptance = {
        detector: bool(metrics["null_calibration_validation"]["passed"])
        for detector, metrics in detector_metrics.items()
    }
    return {
        "experiment": str(config.get("experiment_id", "P1_GWOSC")),
        "detectors": detector_metrics,
        "acceptance": {
            "passed": bool(all(detector_acceptance.values())),
            "criterion": "held-out null_sigma_over_predicted near one across split seeds",
            "detectors": detector_acceptance,
        },
    }
