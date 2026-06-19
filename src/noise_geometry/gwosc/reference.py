"""Independent GWpy checks for GWOSC PSD and whitening conventions."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

import numpy as np
from scipy.signal import detrend as scipy_detrend

from ..noise import estimate_psd_ensemble, regularize_psd


def gwpy_psd_reference(
    traces: np.ndarray,
    sampling_frequency: float,
    *,
    window: str = "hann",
    average: str = "median",
    detrend: str | bool = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a non-overlapping ensemble PSD through GWpy.

    Input rows are concatenated, then each row is treated as one complete Welch
    segment. This deliberately mirrors :func:`estimate_psd_ensemble`, including
    GWpy/SciPy's FINDCHIRP bias correction for median averaging.
    """
    from gwpy.timeseries import TimeSeries

    samples = np.asarray(traces, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2:
        raise ValueError("traces must have shape (n_traces, n_samples)")
    if samples.shape[0] < 1:
        raise ValueError("at least one calibration trace is required")
    if samples.shape[1] < 2:
        raise ValueError("traces must contain at least two samples")
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive")

    duration = samples.shape[1] / float(sampling_frequency)
    if average not in {"mean", "median"}:
        raise ValueError("average must be 'mean' or 'median'")
    method = "welch" if average == "mean" else "median"
    series = TimeSeries(samples.reshape(-1), sample_rate=float(sampling_frequency))
    reference = series.psd(
        fftlength=duration,
        overlap=0,
        window=window,
        method=method,
        detrend=detrend,
    )
    return (
        np.asarray(reference.frequencies.value, dtype=np.float64),
        np.asarray(reference.value, dtype=np.float64),
    )


def whiten_time_series_rfft(
    trace: np.ndarray,
    psd: np.ndarray,
    sampling_frequency: float,
    *,
    highpass_hz: float | None = None,
    detrend: str = "constant",
    floor_fraction: float = 1e-8,
) -> np.ndarray:
    """Whiten strain with the repository's direct rFFT convention.

    ``sqrt(2 / fs)`` converts division by a one-sided PSD density into a
    unit-variance time series. This is a diagnostic time-domain normalization;
    it does not change the repository's weighted-inner-product convention.
    """
    values = np.asarray(trace, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("trace must be one-dimensional")
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive")
    if detrend == "constant":
        conditioned = values - np.mean(values)
    elif detrend == "linear":
        conditioned = scipy_detrend(values, type="linear")
    else:
        raise ValueError("detrend must be 'constant' or 'linear'")

    spectrum = np.fft.rfft(conditioned)
    density = regularize_psd(psd, floor_fraction=floor_fraction)
    if density.shape != spectrum.shape:
        raise ValueError("PSD length does not match the trace rFFT")
    whitened_spectrum = spectrum / np.sqrt(density)
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / float(sampling_frequency))
    if highpass_hz is not None:
        whitened_spectrum[frequencies < float(highpass_hz)] = 0.0
    whitened_spectrum *= np.sqrt(2.0 / float(sampling_frequency))
    return np.fft.irfft(whitened_spectrum, n=values.size)


def gwpy_whiten_reference(
    trace: np.ndarray,
    psd: np.ndarray,
    sampling_frequency: float,
    *,
    fduration_seconds: float = 1.0,
    highpass_hz: float | None = None,
    detrend: str = "constant",
    floor_fraction: float = 1e-8,
) -> np.ndarray:
    """Whiten a trace through GWpy using a supplied ASD."""
    from gwpy.frequencyseries import FrequencySeries
    from gwpy.timeseries import TimeSeries

    values = np.asarray(trace, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("trace must be one-dimensional")
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive")
    if fduration_seconds <= 0:
        raise ValueError("fduration_seconds must be positive")
    density = regularize_psd(psd, floor_fraction=floor_fraction)
    if density.shape[0] != values.size // 2 + 1:
        raise ValueError("PSD length does not match the trace rFFT")
    asd = FrequencySeries(
        np.sqrt(density),
        f0=0.0,
        df=float(sampling_frequency) / values.size,
    )
    series = TimeSeries(values, sample_rate=float(sampling_frequency))
    whitened = series.whiten(
        asd=asd,
        fduration=float(fduration_seconds),
        highpass=highpass_hz,
        detrend=detrend,
    )
    return np.asarray(whitened.value, dtype=np.float64)


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return float(np.linalg.norm(candidate - reference) / denominator)


def run_gwpy_reference_check(
    calibration_traces: np.ndarray,
    evaluation_traces: np.ndarray,
    sampling_frequency: float,
    *,
    fduration_seconds: float = 1.0,
    highpass_hz: float | None = None,
    detrend: str = "constant",
    floor_fraction: float = 1e-8,
    psd_window: str = "hann",
    psd_average: str = "median",
    psd_detrend: str | bool = "constant",
) -> dict[str, Any]:
    """Compare repository PSD/whitening normalization with an independent GWpy path.

    PSDs are estimated only from ``calibration_traces``. Whitening diagnostics
    are aggregated over one or more independent ``evaluation_traces``.
    """
    calibration = np.asarray(calibration_traces, dtype=np.float64)
    evaluation = np.asarray(evaluation_traces, dtype=np.float64)
    if calibration.ndim != 2:
        raise ValueError("calibration_traces must have shape (n_traces, n_samples)")
    if evaluation.ndim == 1:
        evaluation = evaluation[None, :]
    if evaluation.ndim != 2 or evaluation.shape[1] != calibration.shape[1]:
        raise ValueError("evaluation_traces must match the calibration trace length")
    if evaluation.shape[0] < 1:
        raise ValueError("at least one evaluation trace is required")

    frequencies, repository_psd = estimate_psd_ensemble(
        calibration,
        sampling_frequency,
        window=psd_window,
        average=psd_average,
        detrend=psd_detrend,
    )
    reference_frequencies, reference_psd = gwpy_psd_reference(
        calibration,
        sampling_frequency,
        window=psd_window,
        average=psd_average,
        detrend=psd_detrend,
    )
    positive = (repository_psd > 0) & (reference_psd > 0)
    if not np.any(positive):
        raise ValueError("PSD comparison has no positive bins")
    ratio = repository_psd[positive] / reference_psd[positive]

    edge_samples = int(np.ceil(0.5 * float(fduration_seconds) * float(sampling_frequency)))
    if 2 * edge_samples >= evaluation.shape[1]:
        raise ValueError("GWpy whitening filter leaves no uncorrupted interior samples")
    interior = slice(edge_samples, evaluation.shape[1] - edge_samples)
    repository_interiors = []
    reference_interiors = []
    repository_std_by_window = []
    reference_std_by_window = []
    correlation_by_window = []
    relative_l2_by_window = []
    for trace in evaluation:
        repository_white = whiten_time_series_rfft(
            trace,
            repository_psd,
            sampling_frequency,
            highpass_hz=highpass_hz,
            detrend=detrend,
            floor_fraction=floor_fraction,
        )
        reference_white = gwpy_whiten_reference(
            trace,
            reference_psd,
            sampling_frequency,
            fduration_seconds=fduration_seconds,
            highpass_hz=highpass_hz,
            detrend=detrend,
            floor_fraction=floor_fraction,
        )
        repository_interior = repository_white[interior]
        reference_interior = reference_white[interior]
        repository_interiors.append(repository_interior)
        reference_interiors.append(reference_interior)
        repository_std_by_window.append(float(np.std(repository_interior)))
        reference_std_by_window.append(float(np.std(reference_interior)))
        correlation_by_window.append(
            float(np.corrcoef(repository_interior, reference_interior)[0, 1])
        )
        relative_l2_by_window.append(
            _relative_l2(reference_interior, repository_interior)
        )
    repository_pooled = np.concatenate(repository_interiors)
    reference_pooled = np.concatenate(reference_interiors)
    repository_std = float(np.std(repository_pooled))
    reference_std = float(np.std(reference_pooled))

    def _distribution(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "values": array.tolist(),
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p05": float(np.quantile(array, 0.05)),
            "p95": float(np.quantile(array, 0.95)),
        }

    return {
        "implementation": {
            "gwpy_version": version("gwpy"),
            "psd_reference": (
                "TimeSeries.psd("
                f"method='{'welch' if psd_average == 'mean' else 'median'}', "
                f"window='{psd_window}', "
                "non-overlapping one-window segments)"
            ),
            "whitening_reference": "TimeSeries.whiten(asd=..., inverse-spectrum FIR)",
        },
        "configuration": {
            "n_calibration_traces": int(calibration.shape[0]),
            "n_evaluation_traces": int(evaluation.shape[0]),
            "n_samples": int(calibration.shape[1]),
            "sampling_frequency_hz": float(sampling_frequency),
            "fduration_seconds": float(fduration_seconds),
            "edge_trim_samples_per_side": edge_samples,
            "highpass_hz": highpass_hz,
            "detrend": detrend,
            "psd_window": psd_window,
            "psd_average": psd_average,
            "psd_detrend": psd_detrend,
            "median_bias_corrected": psd_average == "median",
            "psd_floor_fraction": float(floor_fraction),
        },
        "psd": {
            "frequency_max_abs_difference_hz": float(
                np.max(np.abs(frequencies - reference_frequencies))
            ),
            "relative_l2_error": _relative_l2(reference_psd, repository_psd),
            "ratio_median": float(np.median(ratio)),
            "ratio_p05": float(np.quantile(ratio, 0.05)),
            "ratio_p95": float(np.quantile(ratio, 0.95)),
            "max_abs_log10_ratio": float(np.max(np.abs(np.log10(ratio)))),
        },
        "whitening": {
            "repository_interior_mean": float(np.mean(repository_pooled)),
            "repository_interior_std": repository_std,
            "gwpy_interior_mean": float(np.mean(reference_pooled)),
            "gwpy_interior_std": reference_std,
            "std_ratio_repository_over_gwpy": repository_std
            / max(reference_std, np.finfo(float).tiny),
            "interior_correlation": float(
                np.corrcoef(repository_pooled, reference_pooled)[0, 1]
            ),
            "interior_relative_l2_difference": _relative_l2(
                reference_pooled, repository_pooled
            ),
            "repository_std_by_window": _distribution(repository_std_by_window),
            "gwpy_std_by_window": _distribution(reference_std_by_window),
            "correlation_by_window": _distribution(correlation_by_window),
            "relative_l2_difference_by_window": _distribution(relative_l2_by_window),
        },
    }
