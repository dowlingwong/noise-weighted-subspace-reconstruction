"""Documented waveform construction for GWOSC diagnostics."""

from __future__ import annotations

from fractions import Fraction
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly


GW150914_PUBLIC_WAVEFORM_URL = (
    "https://www.gw-openscience.org/s/events/GW150914/P150914/"
    "fig2-unfiltered-waveform-H.txt"
)
GW150914_PUBLIC_WAVEFORM_CITATION = (
    "GWpy TimeSeries injection example using the public GW150914 "
    "numerical-simulation waveform"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _place_at_peak(
    source: np.ndarray,
    length: int,
    sampling_frequency: float,
    peak_time_seconds: float,
) -> tuple[np.ndarray, dict[str, int]]:
    target = np.zeros(int(length), dtype=np.float64)
    source_peak = int(np.argmax(np.abs(source)))
    target_peak = int(round(float(peak_time_seconds) * sampling_frequency))
    target_peak = int(np.clip(target_peak, 0, length - 1))
    target_start = target_peak - source_peak
    source_start = max(0, -target_start)
    target_start = max(0, target_start)
    count = min(source.size - source_start, length - target_start)
    if count <= 0:
        raise ValueError("waveform placement does not overlap the target trace")
    target[target_start : target_start + count] = source[
        source_start : source_start + count
    ]
    return target, {
        "source_peak_index": source_peak,
        "target_peak_index": target_peak,
        "source_start_index": source_start,
        "target_start_index": target_start,
        "copied_samples": int(count),
    }


def _normalize_waveform(
    waveform: np.ndarray,
    normalization: str,
) -> tuple[np.ndarray, float]:
    values = np.asarray(waveform, dtype=np.float64)
    if normalization == "none":
        return values, 1.0
    if normalization == "peak":
        scale = float(np.max(np.abs(values)))
    elif normalization == "l2":
        scale = float(np.linalg.norm(values))
    else:
        raise ValueError("waveform normalization must be 'none', 'peak', or 'l2'")
    if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
        raise ValueError("waveform normalization is not positive and finite")
    return values / scale, scale


def sine_gaussian_waveform(
    length: int,
    sampling_frequency: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct a fully specified analytic sine-Gaussian probe.

    The convention is

    ``h(t) = exp(-(t-t0)^2 / (2 sigma^2)) cos(2 pi f0 (t-t0) + phase)``

    with ``sigma = Q / (2 pi f0)``.
    """
    frequency = float(
        config.get(
            "central_frequency_hz",
            min(150.0, 0.25 * sampling_frequency),
        )
    )
    quality_factor = float(config.get("quality_factor", 8.0))
    peak_time = float(
        config.get("peak_time_seconds", 0.5 * length / sampling_frequency)
    )
    phase = float(config.get("phase_radians", 0.0))
    if not 0.0 < frequency < 0.5 * sampling_frequency:
        raise ValueError("sine-Gaussian frequency must lie below Nyquist")
    if quality_factor <= 0:
        raise ValueError("sine-Gaussian quality factor must be positive")
    sigma = quality_factor / (2.0 * np.pi * frequency)
    time = np.arange(length, dtype=np.float64) / sampling_frequency
    offset = time - peak_time
    waveform = np.exp(-0.5 * (offset / sigma) ** 2) * np.cos(
        2.0 * np.pi * frequency * offset + phase
    )
    normalization = str(config.get("normalization", "peak"))
    waveform, scale = _normalize_waveform(waveform, normalization)
    return waveform, {
        "type": "sine_gaussian",
        "equation": (
            "exp(-(t-t0)^2/(2 sigma^2))*cos(2*pi*f0*(t-t0)+phase), "
            "sigma=Q/(2*pi*f0)"
        ),
        "central_frequency_hz": frequency,
        "quality_factor": quality_factor,
        "sigma_seconds": sigma,
        "peak_time_seconds": peak_time,
        "phase_radians": phase,
        "normalization": normalization,
        "normalization_scale": scale,
    }


def public_text_waveform(
    length: int,
    sampling_frequency: float,
    config: dict[str, Any],
    data_root: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load, resample, and place a cached public two-column waveform."""
    relative_path = Path(
        config.get(
            "relative_path",
            "gwosc/raw/GW150914/waveforms/"
            "fig2-unfiltered-waveform-H.txt",
        )
    )
    path = Path(data_root).expanduser().resolve() / relative_path
    if not path.is_file():
        raise FileNotFoundError(
            f"documented waveform is not cached at {path}; "
            "rerun scripts/download/download_gwosc.py --download"
        )
    table = np.loadtxt(path, comments="#")
    if table.ndim != 2 or table.shape[1] < 2 or table.shape[0] < 2:
        raise ValueError("public waveform must contain at least two numeric columns")
    source_time = np.asarray(table[:, 0], dtype=np.float64)
    source_values = np.asarray(table[:, 1], dtype=np.float64)
    value_scale = float(config.get("value_scale", 1e-21))
    source_values *= value_scale
    source_dt = float(np.median(np.diff(source_time)))
    if source_dt <= 0:
        raise ValueError("public waveform time column must be strictly increasing")
    source_rate = 1.0 / source_dt
    ratio = Fraction(float(sampling_frequency) / source_rate).limit_denominator(
        100000
    )
    resampled = resample_poly(source_values, ratio.numerator, ratio.denominator)
    peak_time = float(
        config.get("peak_time_seconds", 0.5 * length / sampling_frequency)
    )
    placed, placement = _place_at_peak(
        resampled,
        length,
        sampling_frequency,
        peak_time,
    )
    normalization = str(config.get("normalization", "peak"))
    waveform, scale = _normalize_waveform(placed, normalization)
    return waveform, {
        "type": "public_text",
        "name": str(config.get("name", "GW150914_public_NR_waveform_H")),
        "path": str(path),
        "sha256": _sha256(path),
        "source_url": str(
            config.get("source_url", GW150914_PUBLIC_WAVEFORM_URL)
        ),
        "citation": str(
            config.get(
                "citation",
                GW150914_PUBLIC_WAVEFORM_CITATION,
            )
        ),
        "source_samples": int(source_values.size),
        "source_sample_rate_hz": source_rate,
        "source_time_range_seconds": [
            float(source_time[0]),
            float(source_time[-1]),
        ],
        "value_scale": value_scale,
        "resample_up": int(ratio.numerator),
        "resample_down": int(ratio.denominator),
        "target_sample_rate_hz": float(sampling_frequency),
        "peak_time_seconds": peak_time,
        "normalization": normalization,
        "normalization_scale": scale,
        "placement": placement,
    }


def build_waveform(
    length: int,
    sampling_frequency: float,
    config: dict[str, Any] | None,
    *,
    data_root: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a documented waveform and return its complete metadata."""
    waveform_config = dict(config or {})
    waveform_type = str(waveform_config.get("type", "sine_gaussian"))
    if waveform_type == "sine_gaussian":
        return sine_gaussian_waveform(
            length,
            sampling_frequency,
            waveform_config,
        )
    if waveform_type == "public_text":
        if data_root is None:
            raise ValueError("data_root is required for a public waveform")
        return public_text_waveform(
            length,
            sampling_frequency,
            waveform_config,
            data_root,
        )
    raise ValueError(f"unknown waveform type: {waveform_type}")
