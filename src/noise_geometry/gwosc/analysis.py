"""GWOSC cached-event likelihood-geometry analysis."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import welch

from ..filters import gls_amplitude, psd_amplitude_variance
from ..metrics import mse, weighted_residual
from ..noise import estimate_psd_ensemble, inverse_psd_weights, regularize_psd
from ..utils.paths import dataset_root
from .reference import run_gwpy_reference_check
from .waveforms import build_waveform


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


def _window_quality_features(
    windows: np.ndarray,
    indices: np.ndarray,
    sampling_frequency: float,
    *,
    psd_window: str,
    psd_detrend: str | bool,
    highpass_hz: float | None,
    config: dict[str, Any],
) -> dict[str, np.ndarray | float]:
    selected = np.asarray(windows[indices], dtype=np.float64)
    if selected.ndim != 2 or selected.shape[0] < 1:
        raise ValueError("quality-diagnostic windows must be a non-empty 2D array")

    detrended = selected - np.mean(selected, axis=1, keepdims=True)
    time_rms = np.sqrt(np.mean(detrended**2, axis=1))
    peak_abs = np.max(np.abs(detrended), axis=1)
    crest_factor = peak_abs / np.maximum(time_rms, np.finfo(float).tiny)
    frequencies, periodograms = welch(
        selected,
        fs=float(sampling_frequency),
        window=psd_window,
        nperseg=selected.shape[1],
        noverlap=0,
        nfft=selected.shape[1],
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
    return {
        "time_rms": time_rms,
        "crest_factor": crest_factor,
        "band_power": band_power,
        "band_min_hz": band_min,
        "band_max_hz": band_max,
    }


def _quality_reference(features: dict[str, np.ndarray | float]) -> dict[str, float]:
    reference: dict[str, float] = {}
    for name in ("time_rms", "band_power"):
        values = np.log(
            np.maximum(
                np.asarray(features[name], dtype=np.float64),
                np.finfo(float).tiny,
            )
        )
        reference[f"log_{name}_median"] = float(np.median(values))
        reference[f"log_{name}_mad"] = float(
            np.median(np.abs(values - np.median(values)))
        )
    return reference


def _reference_robust_zscore(
    values: np.ndarray,
    *,
    median: float,
    mad: float,
) -> np.ndarray:
    if mad <= np.finfo(float).eps:
        return np.zeros_like(values, dtype=np.float64)
    return 0.6744897501960817 * (values - median) / mad


def _window_quality_diagnostics(
    windows: np.ndarray,
    indices: np.ndarray,
    sampling_frequency: float,
    *,
    psd_window: str,
    psd_detrend: str | bool,
    highpass_hz: float | None,
    config: dict[str, Any],
    reference: dict[str, float] | None = None,
    role: str,
    enforce_rejection_limit: bool,
) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
    """Score windows against a robust calibration-derived quality reference."""
    features = _window_quality_features(
        windows,
        indices,
        sampling_frequency,
        psd_window=psd_window,
        psd_detrend=psd_detrend,
        highpass_hz=highpass_hz,
        config=config,
    )
    if reference is None:
        reference = _quality_reference(features)
    time_rms = np.asarray(features["time_rms"], dtype=np.float64)
    band_power = np.asarray(features["band_power"], dtype=np.float64)
    crest_factor = np.asarray(features["crest_factor"], dtype=np.float64)
    log_rms_z = _reference_robust_zscore(
        np.log(np.maximum(time_rms, np.finfo(float).tiny)),
        median=reference["log_time_rms_median"],
        mad=reference["log_time_rms_mad"],
    )
    log_band_power_z = _reference_robust_zscore(
        np.log(np.maximum(band_power, np.finfo(float).tiny)),
        median=reference["log_band_power_median"],
        mad=reference["log_band_power_mad"],
    )
    robust_z_threshold = float(config.get("robust_z_threshold", 6.0))
    crest_factor_threshold = float(config.get("crest_factor_threshold", 20.0))
    enabled = bool(config.get("enabled", True))

    reasons: list[list[str]] = []
    keep = np.ones(indices.size, dtype=bool)
    for index in range(indices.size):
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
    if enforce_rejection_limit and rejected_fraction > max_rejected_fraction:
        raise ValueError(
            f"{role} quality rejected "
            f"{rejected_fraction:.1%} of windows, above configured "
            f"{max_rejected_fraction:.1%}"
        )

    records = []
    for local_index, global_index in enumerate(indices):
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
        "role": role,
        "enabled": enabled,
        "band_hz": [
            float(features["band_min_hz"]),
            float(features["band_max_hz"]),
        ],
        "robust_z_threshold": robust_z_threshold,
        "crest_factor_threshold": crest_factor_threshold,
        "max_rejected_fraction": max_rejected_fraction,
        "n_candidates": int(indices.size),
        "n_accepted": int(np.sum(keep)),
        "n_rejected": int(np.sum(~keep)),
        "rejected_fraction": rejected_fraction,
        "accepted_window_indices": indices[keep].tolist(),
        "rejected_window_indices": indices[~keep].tolist(),
        "reference": reference,
        "windows": records,
    }
    return keep, diagnostics, reference


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
    keep, diagnostics, _ = _window_quality_diagnostics(
        windows,
        candidate_indices,
        sampling_frequency,
        psd_window=psd_window,
        psd_detrend=psd_detrend,
        highpass_hz=highpass_hz,
        config=config,
        role="calibration",
        enforce_rejection_limit=True,
    )
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
    candidate_indices: np.ndarray | None = None,
    evaluation_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    if candidate_indices is None or evaluation_indices is None:
        candidate_indices, evaluation_indices = _split_offsource_windows(
            offsource,
            calibration_fraction=calibration_fraction,
            seed=split_seed,
            minimum_calibration_windows=minimum_calibration_windows,
            minimum_evaluation_windows=minimum_evaluation_windows,
        )
    else:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        evaluation_indices = np.asarray(evaluation_indices, dtype=np.int64)
        if candidate_indices.size < int(minimum_calibration_windows):
            raise ValueError("explicit calibration split has too few windows")
        if evaluation_indices.size < int(minimum_evaluation_windows):
            raise ValueError("explicit evaluation split has too few windows")
        if np.intersect1d(candidate_indices, evaluation_indices).size:
            raise ValueError("calibration and evaluation windows must be disjoint")
    psd_window = str(psd_config.get("window", "hann"))
    psd_average = str(psd_config.get("average", "median"))
    psd_detrend = psd_config.get("detrend", "constant")
    quality_keep, quality, quality_reference = _window_quality_diagnostics(
        offsource,
        candidate_indices,
        sampling_frequency,
        psd_window=psd_window,
        psd_detrend=psd_detrend,
        highpass_hz=highpass_hz,
        config=quality_config,
        role="calibration",
        enforce_rejection_limit=True,
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
    evaluation_quality_keep, evaluation_quality, _ = _window_quality_diagnostics(
        offsource,
        evaluation_indices,
        sampling_frequency,
        psd_window=psd_window,
        psd_detrend=psd_detrend,
        highpass_hz=highpass_hz,
        config=quality_config,
        reference=quality_reference,
        role="evaluation",
        enforce_rejection_limit=False,
    )
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
    sensitivity_null_amp = null_amp[evaluation_quality_keep]
    sensitivity_null_amplitude_std = (
        float(np.std(sensitivity_null_amp, ddof=1))
        if sensitivity_null_amp.size > 1
        else float("nan")
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
        "evaluation_quality": evaluation_quality,
        "evaluation_quality_keep": evaluation_quality_keep,
        "quality_filtered_sensitivity": {
            "n_evaluation_windows": int(sensitivity_null_amp.size),
            "null_amplitude_std": sensitivity_null_amplitude_std,
            "null_sigma_over_predicted": (
                sensitivity_null_amplitude_std / amplitude_sigma
                if np.isfinite(sensitivity_null_amplitude_std)
                else float("nan")
            ),
            "used_for_acceptance": False,
        },
        "psd_estimator": {
            "window": psd_window,
            "average": psd_average,
            "detrend": psd_detrend,
            "median_bias_corrected": psd_average == "median",
        },
    }


def _load_data_quality_record(
    cache_file: Path,
    detector: str,
) -> dict[str, Any]:
    metadata_path = cache_file.parent / "metadata.json"
    if not metadata_path.is_file():
        return {
            "available": False,
            "metadata_file": str(metadata_path),
            "flag": f"{detector}_DATA",
            "segments": [],
        }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    record = dict(metadata.get("data_quality", {}).get(detector, {}))
    segments = [
        [float(start), float(stop)]
        for start, stop in record.get("segments", [])
    ]
    return {
        "available": bool(record.get("flag")) and bool(segments),
        "metadata_file": str(metadata_path),
        "flag": str(record.get("flag", f"{detector}_DATA")),
        "segments": segments,
    }


def _interval_is_covered(
    start: float,
    stop: float,
    segments: list[list[float]],
) -> bool:
    return any(
        start >= segment_start and stop <= segment_stop
        for segment_start, segment_stop in segments
    )


def _apply_data_quality(
    starts_gps: np.ndarray,
    duration_seconds: float,
    record: dict[str, Any],
    *,
    required: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not record["available"]:
        if required:
            raise ValueError(
                f"required GWOSC data-quality metadata is unavailable: "
                f"{record['metadata_file']}"
            )
        valid = np.ones(starts_gps.size, dtype=bool)
    else:
        valid = np.asarray(
            [
                _interval_is_covered(
                    float(start),
                    float(start + duration_seconds),
                    record["segments"],
                )
                for start in starts_gps
            ],
            dtype=bool,
        )
    diagnostics = {
        **record,
        "required": bool(required),
        "n_candidate_windows": int(starts_gps.size),
        "n_valid_windows": int(np.sum(valid)),
        "n_invalid_windows": int(np.sum(~valid)),
        "valid_window_indices": np.flatnonzero(valid).tolist(),
        "invalid_window_indices": np.flatnonzero(~valid).tolist(),
        "invalid_window_starts_gps": starts_gps[~valid].tolist(),
    }
    return valid, diagnostics


def _blocked_splits(
    n_windows: int,
    *,
    window_starts: np.ndarray,
    n_splits: int,
    calibration_windows: int,
    minimum_calibration_windows: int,
    minimum_evaluation_windows: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return chronological evaluation blocks with nearest-time calibration."""
    if n_splits < 2:
        raise ValueError("blocked validation requires at least two splits")
    starts = np.asarray(window_starts, dtype=np.float64)
    if starts.shape != (n_windows,):
        raise ValueError("window_starts must match n_windows")
    chronological_order = np.argsort(starts, kind="stable")
    blocks = [
        np.asarray(block, dtype=np.int64)
        for block in np.array_split(chronological_order, n_splits)
    ]
    splits = []
    for evaluation_indices in blocks:
        if evaluation_indices.size < minimum_evaluation_windows:
            raise ValueError("blocked evaluation split has too few windows")
        remaining = np.setdiff1d(
            np.arange(n_windows, dtype=np.int64),
            evaluation_indices,
            assume_unique=True,
        )
        count = min(int(calibration_windows), int(remaining.size))
        if count < minimum_calibration_windows:
            raise ValueError("blocked calibration split has too few windows")
        center = float(np.mean(starts[evaluation_indices]))
        order = np.argsort(np.abs(starts[remaining] - center), kind="stable")
        candidate_indices = np.sort(remaining[order[:count]])
        splits.append((candidate_indices, evaluation_indices))
    return splits


def _serialize_noise_model(
    model: dict[str, Any],
    *,
    split_kind: str,
    split_id: int,
    source_window_indices: np.ndarray,
    starts_seconds: np.ndarray,
    starts_gps: np.ndarray,
    per_split_bounds: list[float],
) -> dict[str, Any]:
    ratio = float(model["null_sigma_over_predicted"])
    evaluation_indices = np.asarray(model["evaluation_indices"], dtype=np.int64)
    candidate_indices = np.asarray(
        model["candidate_calibration_indices"],
        dtype=np.int64,
    )
    calibration_indices = np.asarray(model["calibration_indices"], dtype=np.int64)
    evaluation_quality_records = {
        int(record["window_index"]): record
        for record in model["evaluation_quality"]["windows"]
    }
    evaluation_windows = []
    for local_position, eligible_index in enumerate(evaluation_indices):
        quality = evaluation_quality_records[int(eligible_index)]
        evaluation_windows.append(
            {
                "eligible_window_index": int(eligible_index),
                "source_window_index": int(source_window_indices[eligible_index]),
                "start_seconds_from_cache_start": float(starts_seconds[eligible_index]),
                "start_gps": float(starts_gps[eligible_index]),
                "null_amplitude": float(model["null_amp"][local_position]),
                "null_score": float(
                    model["null_amp"][local_position] / model["amplitude_sigma"]
                ),
                "quality_accepted": bool(quality["accepted"]),
                "quality_reasons": list(quality["reasons"]),
                "time_rms_robust_z": float(quality["time_rms_robust_z"]),
                "band_power_robust_z": float(quality["band_power_robust_z"]),
                "crest_factor": float(quality["crest_factor"]),
            }
        )
    return {
        "split_kind": split_kind,
        "split_id": int(split_id),
        "split_seed": int(model["split_seed"]),
        "null_sigma_over_predicted": ratio,
        "passed": bool(
            np.isfinite(ratio)
            and per_split_bounds[0] <= ratio <= per_split_bounds[1]
        ),
        "amplitude_sigma": float(model["amplitude_sigma"]),
        "null_amplitude_std": float(model["null_amplitude_std"]),
        "n_calibration_candidates": int(candidate_indices.size),
        "n_calibration_windows": int(calibration_indices.size),
        "n_evaluation_windows": int(evaluation_indices.size),
        "candidate_calibration_indices": source_window_indices[
            candidate_indices
        ].tolist(),
        "calibration_indices": source_window_indices[calibration_indices].tolist(),
        "evaluation_indices": source_window_indices[evaluation_indices].tolist(),
        "candidate_calibration_starts_seconds_from_cache_start": starts_seconds[
            candidate_indices
        ].tolist(),
        "calibration_starts_seconds_from_cache_start": starts_seconds[
            calibration_indices
        ].tolist(),
        "evaluation_starts_seconds_from_cache_start": starts_seconds[
            evaluation_indices
        ].tolist(),
        "evaluation_starts_gps": starts_gps[evaluation_indices].tolist(),
        "rejected_calibration_windows": source_window_indices[
            np.asarray(model["quality"]["rejected_window_indices"], dtype=np.int64)
        ].tolist(),
        "evaluation_quality": model["evaluation_quality"],
        "quality_filtered_sensitivity": model["quality_filtered_sensitivity"],
        "evaluation_windows": evaluation_windows,
    }


def _summarize_split_validation(
    split_results: list[dict[str, Any]],
    *,
    per_split_bounds: list[float],
    median_bounds: list[float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ratios = np.asarray(
        [result["null_sigma_over_predicted"] for result in split_results],
        dtype=np.float64,
    )
    median_ratio = float(np.median(ratios))
    summary = {
        "passed": bool(
            all(result["passed"] for result in split_results)
            and median_bounds[0] <= median_ratio <= median_bounds[1]
        ),
        "n_splits": len(split_results),
        "per_split_ratio_bounds": per_split_bounds,
        "median_ratio_bounds": median_bounds,
        "median_null_sigma_over_predicted": median_ratio,
        "min_null_sigma_over_predicted": float(np.min(ratios)),
        "max_null_sigma_over_predicted": float(np.max(ratios)),
        "splits": split_results,
    }
    if extra:
        summary.update(extra)
    return summary


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
    data_quality_config = dict(config.get("data_quality", {}))
    null_validation_config = dict(config.get("null_calibration_validation", {}))
    blocked_validation_config = dict(
        config.get("blocked_null_calibration_validation", {})
    )
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
        offsource_all, offsource_starts_all = _offsource_windows(
            values,
            center,
            length,
        )
        offsource_starts_gps_all = times[offsource_starts_all]
        data_quality_record = _load_data_quality_record(cache_file, detector)
        data_quality_required = bool(data_quality_config.get("required", False))
        data_quality_keep, data_quality_diagnostics = _apply_data_quality(
            offsource_starts_gps_all,
            length / sample_rate,
            data_quality_record,
            required=data_quality_required,
        )
        event_start_gps = float(times[start])
        event_stop_gps = event_start_gps + length / sample_rate
        event_data_quality_valid = (
            _interval_is_covered(
                event_start_gps,
                event_stop_gps,
                data_quality_record["segments"],
            )
            if data_quality_record["available"]
            else not data_quality_required
        )
        if data_quality_required and not event_data_quality_valid:
            raise ValueError(f"{detector} event window fails required data-quality flag")
        source_window_indices = np.flatnonzero(data_quality_keep)
        offsource = offsource_all[data_quality_keep]
        offsource_starts = offsource_starts_all[data_quality_keep]
        offsource_starts_gps = offsource_starts_gps_all[data_quality_keep]
        offsource_starts_seconds = offsource_starts / sample_rate
        data_quality_diagnostics["event_window"] = {
            "start_gps": event_start_gps,
            "stop_gps": event_stop_gps,
            "valid": bool(event_data_quality_valid),
        }
        template, waveform_metadata = build_waveform(
            length,
            sample_rate,
            config.get("waveform"),
            data_root=data_root_path,
        )
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
            split_results.append(
                _serialize_noise_model(
                    validation_model,
                    split_kind="random",
                    split_id=validation_seed,
                    source_window_indices=source_window_indices,
                    starts_seconds=offsource_starts_seconds,
                    starts_gps=offsource_starts_gps,
                    per_split_bounds=per_split_bounds,
                )
            )
        null_calibration_validation = _summarize_split_validation(
            split_results,
            per_split_bounds=per_split_bounds,
            median_bounds=median_bounds,
            extra={
                "n_split_seeds": len(validation_split_seeds),
                "split_seeds": validation_split_seeds,
                "split_kind": "random",
            },
        )

        blocked_null_calibration_validation = {
            "enabled": False,
            "required_for_acceptance": False,
            "passed": True,
            "splits": [],
        }
        if bool(blocked_validation_config.get("enabled", False)):
            blocked_per_split_bounds = [
                float(value)
                for value in blocked_validation_config.get(
                    "per_split_ratio_bounds",
                    per_split_bounds,
                )
            ]
            blocked_median_bounds = [
                float(value)
                for value in blocked_validation_config.get(
                    "median_ratio_bounds",
                    median_bounds,
                )
            ]
            n_blocked_splits = int(blocked_validation_config.get("n_splits", 5))
            blocked_minimum_evaluation = int(
                blocked_validation_config.get(
                    "minimum_evaluation_windows",
                    max(2, minimum_evaluation_windows // 2),
                )
            )
            blocked_calibration_windows = int(
                blocked_validation_config.get(
                    "calibration_windows",
                    minimum_calibration_windows,
                )
            )
            blocked_results = []
            for fold, (
                blocked_candidate_indices,
                blocked_evaluation_indices,
            ) in enumerate(
                _blocked_splits(
                    int(offsource.shape[0]),
                    window_starts=offsource_starts_gps,
                    n_splits=n_blocked_splits,
                    calibration_windows=blocked_calibration_windows,
                    minimum_calibration_windows=minimum_calibration_windows,
                    minimum_evaluation_windows=blocked_minimum_evaluation,
                )
            ):
                blocked_model = _fit_noise_model(
                    offsource,
                    template_f,
                    sample_rate,
                    split_seed=fold,
                    calibration_fraction=calibration_fraction,
                    minimum_calibration_windows=minimum_calibration_windows,
                    minimum_evaluation_windows=blocked_minimum_evaluation,
                    psd_floor_fraction=psd_floor_fraction,
                    highpass_hz=highpass_hz,
                    psd_config=psd_config,
                    quality_config=quality_config,
                    candidate_indices=blocked_candidate_indices,
                    evaluation_indices=blocked_evaluation_indices,
                )
                blocked_results.append(
                    _serialize_noise_model(
                        blocked_model,
                        split_kind="chronological_block",
                        split_id=fold,
                        source_window_indices=source_window_indices,
                        starts_seconds=offsource_starts_seconds,
                        starts_gps=offsource_starts_gps,
                        per_split_bounds=blocked_per_split_bounds,
                    )
                )
            blocked_null_calibration_validation = _summarize_split_validation(
                blocked_results,
                per_split_bounds=blocked_per_split_bounds,
                median_bounds=blocked_median_bounds,
                extra={
                    "enabled": True,
                    "required_for_acceptance": bool(
                        blocked_validation_config.get(
                            "required_for_acceptance",
                            False,
                        )
                    ),
                    "split_kind": "chronological_block",
                    "calibration_windows_per_split": blocked_calibration_windows,
                },
            )
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
            "data_quality": data_quality_diagnostics,
            "offsource_source_window_indices": source_window_indices.tolist(),
            "psd_calibration_window_indices": source_window_indices[
                calibration_indices
            ].tolist(),
            "evaluation_window_indices": source_window_indices[
                evaluation_indices
            ].tolist(),
            "psd_calibration_window_starts_seconds_from_cache_start": (
                offsource_starts[calibration_indices] / sample_rate
            ).tolist(),
            "evaluation_window_starts_seconds_from_cache_start": (
                offsource_starts[evaluation_indices] / sample_rate
            ).tolist(),
            "analysis_highpass_hz": highpass_hz,
            "waveform": waveform_metadata,
            "psd_estimator": model["psd_estimator"],
            "psd_calibration_quality": model["quality"],
            "evaluation_window_quality": model["evaluation_quality"],
            "quality_filtered_sensitivity": model[
                "quality_filtered_sensitivity"
            ],
            "null_calibration_validation": null_calibration_validation,
            "blocked_null_calibration_validation": (
                blocked_null_calibration_validation
            ),
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
                template=template,
            )

    detector_acceptance = {
        detector: bool(
            metrics["null_calibration_validation"]["passed"]
            and (
                not metrics["blocked_null_calibration_validation"].get(
                    "required_for_acceptance",
                    False,
                )
                or metrics["blocked_null_calibration_validation"]["passed"]
            )
            and (
                not data_quality_config.get("required", False)
                or (
                    metrics["data_quality"]["available"]
                    and metrics["data_quality"]["event_window"]["valid"]
                )
            )
        )
        for detector, metrics in detector_metrics.items()
    }
    return {
        "experiment": str(config.get("experiment_id", "P1_GWOSC")),
        "detectors": detector_metrics,
        "acceptance": {
            "passed": bool(all(detector_acceptance.values())),
            "criterion": (
                "held-out null_sigma_over_predicted near one across random "
                "split seeds and required chronological blocks, with required "
                "official data-quality coverage"
            ),
            "detectors": detector_acceptance,
        },
    }
