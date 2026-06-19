"""GWOSC cached-event likelihood-geometry analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import chirp
from scipy.signal.windows import tukey

from ..filters import gls_amplitude, matched_filter_score
from ..metrics import mse, weighted_residual
from ..noise import estimate_psd_rfft, inverse_psd_weights, regularize_psd
from ..utils.paths import dataset_root
from .reference import run_gwpy_reference_check


def load_cached_event(path: str | Path) -> dict[str, Any]:
    """Load a GWOSC event window written by the download helper."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _offsource_windows(values: np.ndarray, center: int, length: int) -> np.ndarray:
    guard = length // 2
    windows = []
    for start in range(0, values.size - length + 1, length):
        stop = start + length
        if stop < center - guard or start > center + guard:
            windows.append(values[start:stop])
    if not windows:
        raise ValueError("downloaded window is too short for independent off-source PSD windows")
    return np.asarray(windows)


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
    detector_metrics: dict[str, Any] = {}

    for detector in detectors:
        candidates = sorted((root / "raw" / event).glob(f"{event}_{detector}_*.npz"))
        if not candidates:
            raise FileNotFoundError(
                f"No cached {event} {detector} data under {root / 'raw' / event}. "
                "Run scripts/download/download_gwosc.py --download first."
            )
        cached = load_cached_event(candidates[0])
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
        offsource = _offsource_windows(values, center, length)
        _, psd = estimate_psd_rfft(offsource, sample_rate)
        psd = regularize_psd(psd, floor_fraction=float(config.get("psd_floor_fraction", 1e-6)))
        weights = inverse_psd_weights(psd, length)
        template = _approximate_chirp_template(length, sample_rate)
        template_f = np.fft.rfft(template)
        event_f = np.fft.rfft(event_trace)
        event_amp = float(gls_amplitude(event_f, template_f, weights))
        event_recon_f = event_amp * template_f

        template_norm = float(np.sqrt(np.real(np.sum(np.conj(template_f) * template_f * weights))))
        injection_amp = target_snr / max(template_norm, np.finfo(float).eps)
        injection_f = np.fft.rfft(offsource, axis=1) + injection_amp * template_f[None, :]
        recovered_amp = gls_amplitude(injection_f, template_f, weights)
        recovered_score = matched_filter_score(injection_f, template_f, weights)
        detector_metrics[detector] = {
            "cache_file": str(candidates[0]),
            "sample_rate": sample_rate,
            "analysis_duration_seconds": length / sample_rate,
            "n_offsource_windows": int(offsource.shape[0]),
            "event_amplitude": event_amp,
            "event_matched_filter_score": float(matched_filter_score(event_f, template_f, weights)),
            "event_raw_mse": float(mse(event_f, event_recon_f)),
            "event_weighted_residual": float(weighted_residual(event_f, event_recon_f, weights)),
            "injection_target_snr": target_snr,
            "injection_amplitude": injection_amp,
            "injection_amplitude_bias": float(np.mean(recovered_amp - injection_amp)),
            "injection_score_mean": float(np.mean(recovered_score)),
        }
        reference_config = config.get("gwpy_reference", {})
        if bool(reference_config.get("enabled", False)):
            if offsource.shape[0] < 2:
                raise ValueError("GWpy reference check needs at least two off-source windows")
            detector_metrics[detector]["gwpy_reference"] = run_gwpy_reference_check(
                offsource[:-1],
                offsource[-1],
                sample_rate,
                fduration_seconds=float(reference_config.get("fduration_seconds", 1.0)),
                highpass_hz=reference_config.get("highpass_hz"),
                detrend=str(reference_config.get("detrend", "constant")),
                floor_fraction=float(config.get("psd_floor_fraction", 1e-6)),
            )

    return {"experiment": str(config.get("experiment_id", "P1_GWOSC")), "detectors": detector_metrics}
