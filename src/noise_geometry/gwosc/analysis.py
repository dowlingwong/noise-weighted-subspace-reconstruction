"""GWOSC cached-event likelihood-geometry analysis."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import chirp
from scipy.signal.windows import tukey

from ..filters import gls_amplitude, psd_amplitude_variance
from ..metrics import mse, weighted_residual
from ..noise import estimate_psd_rfft, inverse_psd_weights, regularize_psd
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
        calibration_indices, evaluation_indices = _split_offsource_windows(
            offsource,
            calibration_fraction=calibration_fraction,
            seed=split_seed,
            minimum_calibration_windows=minimum_calibration_windows,
            minimum_evaluation_windows=minimum_evaluation_windows,
        )
        calibration = offsource[calibration_indices]
        evaluation = offsource[evaluation_indices]
        _, psd = estimate_psd_rfft(calibration, sample_rate)
        psd = regularize_psd(psd, floor_fraction=float(config.get("psd_floor_fraction", 1e-6)))
        weights = inverse_psd_weights(psd, length)
        if highpass_hz is not None:
            frequencies = np.fft.rfftfreq(length, d=1.0 / sample_rate)
            weights = weights.copy()
            weights[frequencies < highpass_hz] = 0.0
        template = _approximate_chirp_template(length, sample_rate)
        template_f = np.fft.rfft(template)
        event_f = np.fft.rfft(event_trace)
        event_amp = float(gls_amplitude(event_f, template_f, weights))
        event_recon_f = event_amp * template_f

        amplitude_variance = psd_amplitude_variance(
            template_f,
            psd,
            weights,
            trace_len=length,
            sampling_frequency=sample_rate,
        )
        amplitude_sigma = float(np.sqrt(amplitude_variance))
        if not np.isfinite(amplitude_sigma) or amplitude_sigma <= 0:
            raise ValueError("template amplitude variance is not positive and finite")
        injection_amp = target_snr * amplitude_sigma
        evaluation_f = np.fft.rfft(evaluation, axis=1)
        null_amp = np.asarray(gls_amplitude(evaluation_f, template_f, weights))
        injection_f = evaluation_f + injection_amp * template_f[None, :]
        recovered_amp = gls_amplitude(injection_f, template_f, weights)
        unpaired_recovered_score = np.asarray(recovered_amp) / amplitude_sigma
        null_score = null_amp / amplitude_sigma
        paired_recovered_amp = np.asarray(recovered_amp) - null_amp
        paired_recovered_score = paired_recovered_amp / amplitude_sigma
        null_amplitude_std = (
            float(np.std(null_amp, ddof=1)) if null_amp.size > 1 else 0.0
        )
        detector_metrics[detector] = {
            "cache_file": str(cache_file),
            "sample_rate": sample_rate,
            "analysis_duration_seconds": length / sample_rate,
            "n_offsource_windows": int(offsource.shape[0]),
            "n_psd_calibration_windows": int(calibration.shape[0]),
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
                floor_fraction=float(config.get("psd_floor_fraction", 1e-6)),
            )

    return {"experiment": str(config.get("experiment_id", "P1_GWOSC")), "detectors": detector_metrics}
