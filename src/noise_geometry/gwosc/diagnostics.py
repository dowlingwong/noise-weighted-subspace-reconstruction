"""Predeclared GWOSC filtering and time-local-noise diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import get_window, welch
from scipy.signal import detrend as scipy_detrend

from ..filters import gls_amplitude, psd_amplitude_variance
from ..noise import (
    estimate_psd_ensemble,
    generate_colored_noise,
    inverse_psd_weights,
    make_powerlaw_psd,
    regularize_psd,
)
from ..utils.paths import dataset_root
from .analysis import (
    _apply_data_quality,
    _cached_duration_from_name,
    _load_data_quality_record,
    _offsource_windows,
    _quality_reference,
    _reference_robust_zscore,
    _split_offsource_windows,
    load_cached_event,
)
from .waveforms import build_waveform


def _distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": array.tolist(),
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return float(np.linalg.norm(candidate - reference) / denominator)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _condition_trace(
    trace: np.ndarray,
    detrend: str,
) -> np.ndarray:
    values = np.asarray(trace, dtype=np.float64)
    if detrend == "constant":
        return values - np.mean(values)
    if detrend == "linear":
        return scipy_detrend(values, type="linear")
    raise ValueError("detrend must be 'constant' or 'linear'")


def design_gwpy_whitening_fir(
    psd: np.ndarray,
    sampling_frequency: float,
    trace_length: int,
    *,
    fduration_seconds: float,
    highpass_hz: float | None,
    floor_fraction: float,
    window: str = "hann",
) -> np.ndarray:
    """Design the exact FIR used by GWpy for a supplied trace-grid PSD."""
    from gwpy.signal.filter_design import fir_from_transfer

    density = regularize_psd(psd, floor_fraction=floor_fraction)
    if density.shape != (trace_length // 2 + 1,):
        raise ValueError("PSD length does not match the trace rFFT")
    ntaps = int(round(float(fduration_seconds) * sampling_frequency))
    if ntaps < 2:
        raise ValueError("FIR duration produces fewer than two taps")
    if ntaps % 2:
        ntaps += 1
    frequency_spacing = float(sampling_frequency) / int(trace_length)
    ncorner = (
        int(float(highpass_hz) / frequency_spacing)
        if highpass_hz is not None
        else 0
    )
    return np.asarray(
        fir_from_transfer(
            1.0 / np.sqrt(density),
            ntaps=ntaps,
            window=window,
            ncorner=ncorner,
        ),
        dtype=np.float64,
    )


def _fftconvolve_same(
    values: np.ndarray,
    fir: np.ndarray,
) -> np.ndarray:
    """Linear convolution via explicit FFT multiplication, cropped to ``same``."""
    full_length = values.size + fir.size - 1
    nfft = next_fast_len(full_length)
    full = np.fft.irfft(
        np.fft.rfft(values, n=nfft) * np.fft.rfft(fir, n=nfft),
        n=nfft,
    )[:full_length]
    start = (fir.size - 1) // 2
    return full[start : start + values.size]


def apply_fir_frequency_domain(
    trace: np.ndarray,
    fir: np.ndarray,
    sampling_frequency: float,
    *,
    detrend: str = "constant",
    window: str = "hann",
) -> np.ndarray:
    """Apply GWpy's FIR boundary convention using explicit FFT convolution."""
    conditioned = _condition_trace(trace, detrend)
    pad = int(np.ceil(fir.size / 2))
    taper = get_window(window, fir.size)
    conditioned = conditioned.copy()
    conditioned[:pad] *= taper[:pad]
    conditioned[-pad:] *= taper[-pad:]
    convolved = _fftconvolve_same(conditioned, fir)
    return convolved * np.sqrt(2.0 / float(sampling_frequency))


def apply_fir_gwpy(
    trace: np.ndarray,
    fir: np.ndarray,
    sampling_frequency: float,
    *,
    detrend: str = "constant",
    window: str = "hann",
) -> np.ndarray:
    """Apply the same FIR through GWpy's ``TimeSeries.convolve`` path."""
    from gwpy.timeseries import TimeSeries

    series = TimeSeries(
        np.asarray(trace, dtype=np.float64),
        sample_rate=float(sampling_frequency),
    )
    conditioned = series.detrend(detrend)
    whitened = conditioned.convolve(fir, window=window)
    return np.asarray(whitened.value, dtype=np.float64) * np.sqrt(
        2.0 / float(sampling_frequency)
    )


def _normalized_dot_scores(
    filtered_traces: np.ndarray,
    filtered_template: np.ndarray,
    edge_trim_samples: int,
) -> np.ndarray:
    stop = filtered_template.size - int(edge_trim_samples)
    start = int(edge_trim_samples)
    if stop <= start:
        raise ValueError("edge trim leaves no statistic interior")
    template = filtered_template[start:stop]
    traces = filtered_traces[:, start:stop]
    norm = float(np.linalg.norm(template))
    if norm <= np.finfo(float).tiny:
        raise ValueError("filtered template has zero interior norm")
    return np.asarray(traces @ template / norm, dtype=np.float64)


def _gls_scores(
    traces: np.ndarray,
    template: np.ndarray,
    psd: np.ndarray,
    sampling_frequency: float,
    *,
    highpass_hz: float | None,
) -> np.ndarray:
    template_f = np.fft.rfft(template)
    weights = inverse_psd_weights(psd, template.size)
    if highpass_hz is not None:
        frequencies = np.fft.rfftfreq(
            template.size,
            d=1.0 / sampling_frequency,
        )
        weights = weights.copy()
        weights[frequencies < highpass_hz] = 0.0
    amplitude_sigma = float(
        np.sqrt(
            psd_amplitude_variance(
                template_f,
                psd,
                weights,
                trace_len=template.size,
                sampling_frequency=sampling_frequency,
            )
        )
    )
    amplitudes = np.asarray(
        gls_amplitude(
            np.fft.rfft(traces, axis=1),
            template_f,
            weights,
        )
    )
    return amplitudes / amplitude_sigma


def _prepare_filter_statistics(
    traces: np.ndarray,
    template: np.ndarray,
    psd: np.ndarray,
    sampling_frequency: float,
    *,
    fduration_seconds: float,
    highpass_hz: float | None,
    detrend: str,
    floor_fraction: float,
    window: str = "hann",
) -> dict[str, Any]:
    """Filter traces once for all predeclared edge trims at one FIR duration."""
    samples = np.asarray(traces, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != template.size:
        raise ValueError("traces must be 2D and match the template length")
    fir = design_gwpy_whitening_fir(
        psd,
        sampling_frequency,
        template.size,
        fduration_seconds=fduration_seconds,
        highpass_hz=highpass_hz,
        floor_fraction=floor_fraction,
        window=window,
    )
    frequency_template = apply_fir_frequency_domain(
        template,
        fir,
        sampling_frequency,
        detrend=detrend,
        window=window,
    )
    gwpy_template = apply_fir_gwpy(
        template,
        fir,
        sampling_frequency,
        detrend=detrend,
        window=window,
    )
    frequency_filtered = np.asarray(
        [
            apply_fir_frequency_domain(
                trace,
                fir,
                sampling_frequency,
                detrend=detrend,
                window=window,
            )
            for trace in samples
        ]
    )
    gwpy_filtered = np.asarray(
        [
            apply_fir_gwpy(
                trace,
                fir,
                sampling_frequency,
                detrend=detrend,
                window=window,
            )
            for trace in samples
        ]
    )
    return {
        "samples": samples,
        "template": np.asarray(template, dtype=np.float64),
        "psd": np.asarray(psd, dtype=np.float64),
        "sampling_frequency": float(sampling_frequency),
        "highpass_hz": highpass_hz,
        "fduration_seconds": float(fduration_seconds),
        "fir_taps": int(fir.size),
        "frequency_template": frequency_template,
        "gwpy_template": gwpy_template,
        "frequency_filtered": frequency_filtered,
        "gwpy_filtered": gwpy_filtered,
    }


def _compare_prepared_filter_statistics(
    prepared: dict[str, Any],
    edge_trim_seconds: float,
) -> dict[str, Any]:
    """Calculate the shared matched statistic for one predeclared edge trim."""
    sampling_frequency = float(prepared["sampling_frequency"])
    frequency_template = prepared["frequency_template"]
    gwpy_template = prepared["gwpy_template"]
    frequency_filtered = prepared["frequency_filtered"]
    gwpy_filtered = prepared["gwpy_filtered"]
    edge_trim_samples = int(round(edge_trim_seconds * sampling_frequency))
    frequency_scores = _normalized_dot_scores(
        frequency_filtered,
        frequency_template,
        edge_trim_samples,
    )
    gwpy_scores = _normalized_dot_scores(
        gwpy_filtered,
        gwpy_template,
        edge_trim_samples,
    )
    gls_scores = _gls_scores(
        prepared["samples"],
        prepared["template"],
        prepared["psd"],
        sampling_frequency,
        highpass_hz=prepared["highpass_hz"],
    )
    difference = frequency_scores - gwpy_scores
    return {
        "fduration_seconds": float(prepared["fduration_seconds"]),
        "edge_trim_seconds": float(edge_trim_seconds),
        "edge_trim_samples_per_side": edge_trim_samples,
        "fir_taps": int(prepared["fir_taps"]),
        "trim_at_least_half_fduration": bool(
            edge_trim_seconds >= 0.5 * prepared["fduration_seconds"]
        ),
        "frequency_domain_fir_scores": _distribution(frequency_scores),
        "gwpy_fir_scores": _distribution(gwpy_scores),
        "original_psd_gls_scores": _distribution(gls_scores),
        "identity": {
            "max_abs_score_difference": float(np.max(np.abs(difference))),
            "relative_l2_score_difference": _relative_l2(
                gwpy_scores,
                frequency_scores,
            ),
            "score_correlation": _safe_correlation(
                frequency_scores,
                gwpy_scores,
            ),
            "template_relative_l2_difference": _relative_l2(
                gwpy_template,
                frequency_template,
            ),
            "filtered_trace_relative_l2_median": float(
                np.median(
                    [
                        _relative_l2(reference, candidate)
                        for reference, candidate in zip(
                            gwpy_filtered,
                            frequency_filtered,
                        )
                    ]
                )
            ),
        },
        "original_gls_vs_shared_fir": {
            "correlation": _safe_correlation(gls_scores, gwpy_scores),
            "relative_l2_difference": _relative_l2(gls_scores, gwpy_scores),
            "std_ratio_fir_over_gls": float(
                np.std(gwpy_scores, ddof=1)
                / max(np.std(gls_scores, ddof=1), np.finfo(float).tiny)
            ),
        },
    }


def compare_filter_statistics(
    traces: np.ndarray,
    template: np.ndarray,
    psd: np.ndarray,
    sampling_frequency: float,
    *,
    fduration_seconds: float,
    edge_trim_seconds: float,
    highpass_hz: float | None,
    detrend: str,
    floor_fraction: float,
    window: str = "hann",
) -> dict[str, Any]:
    """Compare one shared FIR matched statistic across FFT and GWpy paths."""
    prepared = _prepare_filter_statistics(
        traces,
        template,
        psd,
        sampling_frequency,
        fduration_seconds=fduration_seconds,
        highpass_hz=highpass_hz,
        detrend=detrend,
        floor_fraction=floor_fraction,
        window=window,
    )
    return _compare_prepared_filter_statistics(prepared, edge_trim_seconds)


def _load_real_offsource(
    config: dict[str, Any],
    data_root_path: str | Path,
    detector: str,
) -> dict[str, Any]:
    root = dataset_root("gwosc", data_root_path)
    event = str(config.get("event", "GW150914"))
    candidates = sorted((root / "raw" / event).glob(f"{event}_{detector}_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"no cached {event} {detector} data")
    cache_file = max(candidates, key=_cached_duration_from_name)
    cached = load_cached_event(cache_file)
    values = np.asarray(cached["value"], dtype=np.float64)
    times = np.asarray(cached["times"], dtype=np.float64)
    sample_rate = float(np.asarray(cached["sample_rate"]))
    gps = float(np.asarray(cached["gps"]))
    duration = float(config.get("analysis_duration_seconds", 4.0))
    length = int(round(duration * sample_rate))
    center = int(np.argmin(np.abs(times - gps)))
    windows, starts = _offsource_windows(values, center, length)
    starts_gps = times[starts]
    data_quality_record = _load_data_quality_record(cache_file, detector)
    keep, data_quality = _apply_data_quality(
        starts_gps,
        duration,
        data_quality_record,
        required=bool(config.get("data_quality", {}).get("required", True)),
    )
    return {
        "cache_file": str(cache_file),
        "windows": windows[keep],
        "starts_samples": starts[keep],
        "starts_seconds": starts[keep] / sample_rate,
        "starts_gps": starts_gps[keep],
        "sample_rate": sample_rate,
        "length": length,
        "data_quality": data_quality,
    }


def _filter_sweep(
    traces: np.ndarray,
    template: np.ndarray,
    psd: np.ndarray,
    sampling_frequency: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    sweep = dict(config.get("fir_sweep", {}))
    durations = [float(value) for value in sweep["durations_seconds"]]
    trims = [float(value) for value in sweep["edge_trims_seconds"]]
    primary_duration = float(sweep["primary_duration_seconds"])
    primary_trim = float(sweep["primary_edge_trim_seconds"])
    if primary_duration not in durations or primary_trim not in trims:
        raise ValueError("primary FIR duration and trim must be in the sweep")
    results = []
    for duration in durations:
        prepared = _prepare_filter_statistics(
            traces,
            template,
            psd,
            sampling_frequency,
            fduration_seconds=duration,
            highpass_hz=(
                float(config["analysis_highpass_hz"])
                if config.get("analysis_highpass_hz") is not None
                else None
            ),
            detrend=str(sweep.get("detrend", "constant")),
            floor_fraction=float(config.get("psd_floor_fraction", 1e-6)),
            window=str(sweep.get("window", "hann")),
        )
        for trim in trims:
            result = _compare_prepared_filter_statistics(prepared, trim)
            result["is_predeclared_primary"] = bool(
                duration == primary_duration and trim == primary_trim
            )
            results.append(result)
    return results


def run_filter_statistic_equivalence(
    config: dict[str, Any],
    data_root_path: str | Path,
) -> dict[str, Any]:
    """Run the shared-FIR statistic sweep on synthetic and real windows."""
    synthetic_config = dict(config.get("synthetic_control", {}))
    synthetic_sample_rate = float(
        synthetic_config.get("sample_rate_hz", 1024.0)
    )
    synthetic_duration = float(
        synthetic_config.get(
            "duration_seconds",
            config.get("analysis_duration_seconds", 4.0),
        )
    )
    synthetic_length = int(round(synthetic_duration * synthetic_sample_rate))
    synthetic_template, synthetic_waveform = build_waveform(
        synthetic_length,
        synthetic_sample_rate,
        config.get("waveform"),
        data_root=data_root_path,
    )
    _, synthetic_psd = make_powerlaw_psd(
        synthetic_length,
        synthetic_sample_rate,
        kind=str(synthetic_config.get("noise_kind", "pink")),
        level=float(synthetic_config.get("noise_level", 1e-42)),
        knee_hz=synthetic_config.get("knee_hz"),
    )
    synthetic_rng = np.random.default_rng(
        int(synthetic_config.get("seed", 20260623))
    )
    synthetic_traces = generate_colored_noise(
        synthetic_rng,
        synthetic_psd,
        synthetic_length,
        synthetic_sample_rate,
        int(synthetic_config.get("n_evaluation_traces", 128)),
    )
    synthetic_results = _filter_sweep(
        synthetic_traces,
        synthetic_template,
        synthetic_psd,
        synthetic_sample_rate,
        config,
    )

    psd_config = dict(config.get("psd_estimator", {}))
    split_seed = int(config.get("offsource_split_seed", 150914))
    real_results: dict[str, Any] = {}
    for detector in config.get("detectors", ["H1", "L1"]):
        real = _load_real_offsource(config, data_root_path, detector)
        calibration_indices, evaluation_indices = _split_offsource_windows(
            real["windows"],
            calibration_fraction=float(
                config.get("psd_calibration_fraction", 0.75)
            ),
            seed=split_seed,
            minimum_calibration_windows=int(
                config.get("minimum_psd_calibration_windows", 32)
            ),
            minimum_evaluation_windows=int(
                config.get("minimum_evaluation_windows", 16)
            ),
        )
        calibration = real["windows"][calibration_indices]
        evaluation = real["windows"][evaluation_indices]
        _, real_psd = estimate_psd_ensemble(
            calibration,
            real["sample_rate"],
            window=str(psd_config.get("window", "hann")),
            average=str(psd_config.get("average", "median")),
            detrend=psd_config.get("detrend", "constant"),
        )
        real_psd = regularize_psd(
            real_psd,
            floor_fraction=float(config.get("psd_floor_fraction", 1e-6)),
        )
        real_template, waveform_metadata = build_waveform(
            real["length"],
            real["sample_rate"],
            config.get("waveform"),
            data_root=data_root_path,
        )
        real_results[detector] = {
            "cache_file": real["cache_file"],
            "data_quality": real["data_quality"],
            "waveform": waveform_metadata,
            "calibration_indices": calibration_indices.tolist(),
            "evaluation_indices": evaluation_indices.tolist(),
            "evaluation_starts_seconds_from_cache_start": real[
                "starts_seconds"
            ][evaluation_indices].tolist(),
            "sweep": _filter_sweep(
                evaluation,
                real_template,
                real_psd,
                real["sample_rate"],
                config,
            ),
        }

    acceptance_config = dict(config.get("acceptance", {}))
    identity_max = float(
        acceptance_config.get("max_abs_identity_difference", 1e-10)
    )
    identity_relative_l2 = float(
        acceptance_config.get("max_identity_relative_l2", 1e-10)
    )
    synthetic_passed = all(
        result["identity"]["max_abs_score_difference"] <= identity_max
        and result["identity"]["relative_l2_score_difference"]
        <= identity_relative_l2
        for result in synthetic_results
    )
    real_identity_passed = {
        detector: all(
            result["identity"]["max_abs_score_difference"] <= identity_max
            and result["identity"]["relative_l2_score_difference"]
            <= identity_relative_l2
            for result in metrics["sweep"]
        )
        for detector, metrics in real_results.items()
    }
    return {
        "experiment": str(
            config.get(
                "experiment_id",
                "P1_GWOSC_FILTER_STATISTIC_EQUIVALENCE",
            )
        ),
        "waveform": {
            "synthetic": synthetic_waveform,
            "real": {
                detector: metrics["waveform"]
                for detector, metrics in real_results.items()
            },
        },
        "predeclared_sweep": config.get("fir_sweep", {}),
        "synthetic_control": {
            "sampling_frequency_hz": synthetic_sample_rate,
            "duration_seconds": synthetic_duration,
            "n_evaluation_traces": int(synthetic_traces.shape[0]),
            "noise_kind": str(synthetic_config.get("noise_kind", "pink")),
            "seed": int(synthetic_config.get("seed", 20260623)),
            "sweep": synthetic_results,
        },
        "real_data": real_results,
        "acceptance": {
            "passed": bool(
                synthetic_passed and all(real_identity_passed.values())
            ),
            "criterion": (
                "the explicitly shared FIR statistic agrees between the "
                "frequency-domain convolution and GWpy convolution paths"
            ),
            "synthetic_control": synthetic_passed,
            "real_data": real_identity_passed,
            "max_abs_identity_difference": identity_max,
            "max_identity_relative_l2": identity_relative_l2,
            "original_psd_gls_comparison_is_diagnostic": True,
        },
    }


def _median_periodogram_bias(n_periodograms: int) -> float:
    """FINDCHIRP finite-sample bias of the periodogram median."""
    indices = 2.0 * np.arange(1.0, (n_periodograms - 1) // 2 + 1)
    return float(1.0 + np.sum(1.0 / (indices + 1.0) - 1.0 / indices))


def _periodogram_matrix(
    windows: np.ndarray,
    sampling_frequency: float,
    *,
    window: str,
    detrend: str | bool,
) -> tuple[np.ndarray, np.ndarray]:
    frequencies, periodograms = welch(
        np.asarray(windows, dtype=np.float64),
        fs=float(sampling_frequency),
        window=window,
        nperseg=windows.shape[1],
        noverlap=0,
        nfft=windows.shape[1],
        detrend=detrend,
        return_onesided=True,
        scaling="density",
        average="mean",
        axis=-1,
    )
    return (
        np.asarray(frequencies, dtype=np.float64),
        np.asarray(periodograms, dtype=np.float64),
    )


def _aggregate_periodograms(
    periodograms: np.ndarray,
    indices: np.ndarray,
    *,
    average: str,
    floor_fraction: float,
) -> np.ndarray:
    selected = np.asarray(periodograms[indices], dtype=np.float64)
    if average == "median":
        psd = np.median(selected, axis=0) / _median_periodogram_bias(
            selected.shape[0]
        )
    elif average == "mean":
        psd = np.mean(selected, axis=0)
    else:
        raise ValueError("PSD average must be 'mean' or 'median'")
    return regularize_psd(psd, floor_fraction=floor_fraction)


def _template_spectral_weights(
    template: np.ndarray,
    sampling_frequency: float,
    highpass_hz: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.fft.rfftfreq(
        template.size,
        d=1.0 / sampling_frequency,
    )
    weights = np.abs(np.fft.rfft(template)) ** 2
    if highpass_hz is not None:
        weights[frequencies < highpass_hz] = 0.0
    total = float(np.sum(weights))
    if total <= np.finfo(float).tiny:
        raise ValueError("template has no spectral energy in the analysis band")
    return frequencies, weights / total


def _quality_features_from_periodograms(
    windows: np.ndarray,
    periodograms: np.ndarray,
    frequencies: np.ndarray,
    sampling_frequency: float,
    highpass_hz: float | None,
    quality_config: dict[str, Any],
) -> dict[str, np.ndarray | float]:
    detrended = windows - np.mean(windows, axis=1, keepdims=True)
    time_rms = np.sqrt(np.mean(detrended**2, axis=1))
    crest_factor = np.max(np.abs(detrended), axis=1) / np.maximum(
        time_rms,
        np.finfo(float).tiny,
    )
    band_min = float(quality_config.get("band_min_hz", highpass_hz or 0.0))
    band_max = float(
        quality_config.get(
            "band_max_hz",
            min(512.0, float(sampling_frequency) / 2.0),
        )
    )
    if not 0.0 <= band_min < band_max <= float(sampling_frequency) / 2.0:
        raise ValueError("PSD quality band must lie within [0, Nyquist]")
    band_mask = (frequencies >= band_min) & (frequencies <= band_max)
    if not np.any(band_mask):
        raise ValueError("PSD quality band contains no frequency bins")
    df = float(frequencies[1] - frequencies[0])
    return {
        "time_rms": time_rms,
        "crest_factor": crest_factor,
        "band_power": np.sum(periodograms[:, band_mask], axis=1) * df,
        "band_min_hz": band_min,
        "band_max_hz": band_max,
    }


def _precomputed_window_quality(
    features: dict[str, np.ndarray | float],
    indices: np.ndarray,
    quality_config: dict[str, Any],
    *,
    role: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected_features = {
        "time_rms": np.asarray(features["time_rms"])[indices],
        "crest_factor": np.asarray(features["crest_factor"])[indices],
        "band_power": np.asarray(features["band_power"])[indices],
    }
    reference = _quality_reference(selected_features)
    time_rms = np.asarray(selected_features["time_rms"], dtype=np.float64)
    band_power = np.asarray(selected_features["band_power"], dtype=np.float64)
    crest_factor = np.asarray(
        selected_features["crest_factor"],
        dtype=np.float64,
    )
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
    robust_z_threshold = float(
        quality_config.get("robust_z_threshold", 6.0)
    )
    crest_factor_threshold = float(
        quality_config.get("crest_factor_threshold", 20.0)
    )
    enabled = bool(quality_config.get("enabled", True))
    reasons = []
    keep = np.ones(indices.size, dtype=bool)
    for local_index in range(indices.size):
        current = []
        if abs(log_rms_z[local_index]) > robust_z_threshold:
            current.append("time_rms_robust_z")
        if abs(log_band_power_z[local_index]) > robust_z_threshold:
            current.append("band_power_robust_z")
        if crest_factor[local_index] > crest_factor_threshold:
            current.append("crest_factor")
        reasons.append(current)
        if enabled and current:
            keep[local_index] = False
    rejected_records = [
        {
            "window_index": int(indices[local_index]),
            "reasons": reasons[local_index],
            "time_rms": float(time_rms[local_index]),
            "time_rms_robust_z": float(log_rms_z[local_index]),
            "crest_factor": float(crest_factor[local_index]),
            "band_power": float(band_power[local_index]),
            "band_power_robust_z": float(log_band_power_z[local_index]),
        }
        for local_index in np.flatnonzero(~keep)
    ]
    return keep, {
        "role": role,
        "enabled": enabled,
        "band_hz": [
            float(features["band_min_hz"]),
            float(features["band_max_hz"]),
        ],
        "robust_z_threshold": robust_z_threshold,
        "crest_factor_threshold": crest_factor_threshold,
        "n_candidates": int(indices.size),
        "n_accepted": int(np.sum(keep)),
        "n_rejected": int(np.sum(~keep)),
        "accepted_window_indices": indices[keep].tolist(),
        "rejected_window_indices": indices[~keep].tolist(),
        "reference": reference,
        "rejected_windows": rejected_records,
    }


def _spectral_diagnostics(
    periodogram: np.ndarray,
    model_psd: np.ndarray,
    frequencies: np.ndarray,
    template_weights: np.ndarray,
    bands_hz: list[list[float]],
) -> dict[str, Any]:
    projected = float(np.sum(periodogram * template_weights))
    projected_model = float(np.sum(model_psd * template_weights))
    bands = []
    df = float(frequencies[1] - frequencies[0])
    for lower, upper in bands_hz:
        mask = (frequencies >= float(lower)) & (frequencies < float(upper))
        observed = float(np.sum(periodogram[mask]) * df)
        predicted = float(np.sum(model_psd[mask]) * df)
        bands.append(
            {
                "band_hz": [float(lower), float(upper)],
                "observed_power": observed,
                "model_power": predicted,
                "observed_over_model": observed
                / max(predicted, np.finfo(float).tiny),
            }
        )
    return {
        "template_projected_psd": projected,
        "model_template_projected_psd": projected_model,
        "template_projected_psd_ratio": projected
        / max(projected_model, np.finfo(float).tiny),
        "narrow_bands": bands,
    }


def _local_model_records(
    windows: np.ndarray,
    starts_seconds: np.ndarray,
    sampling_frequency: float,
    template: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    psd_config = dict(config.get("psd_estimator", {}))
    quality_config = dict(config.get("psd_quality", {}))
    local_config = dict(config.get("local_psd", {}))
    radii = [float(value) for value in local_config["radii_seconds"]]
    primary_radius = float(local_config["primary_radius_seconds"])
    if primary_radius not in radii:
        raise ValueError("primary local PSD radius must be in radii_seconds")
    minimum_calibration = int(
        local_config.get("minimum_calibration_windows", 8)
    )
    average = str(psd_config.get("average", "median"))
    floor_fraction = float(config.get("psd_floor_fraction", 1e-6))
    highpass_hz = (
        float(config["analysis_highpass_hz"])
        if config.get("analysis_highpass_hz") is not None
        else None
    )
    frequencies, periodograms = _periodogram_matrix(
        windows,
        sampling_frequency,
        window=str(psd_config.get("window", "hann")),
        detrend=psd_config.get("detrend", "constant"),
    )
    quality_features = _quality_features_from_periodograms(
        windows,
        periodograms,
        frequencies,
        sampling_frequency,
        highpass_hz,
        quality_config,
    )
    _, template_weights = _template_spectral_weights(
        template,
        sampling_frequency,
        highpass_hz,
    )
    bands_hz = [
        [float(lower), float(upper)]
        for lower, upper in local_config.get(
            "narrow_bands_hz",
            [[20.0, 80.0], [80.0, 150.0], [150.0, 300.0], [300.0, 512.0]],
        )
    ]
    template_f = np.fft.rfft(template)
    model_names = ["global_leave_one_out"] + [
        f"local_radius_{radius:g}s" for radius in radii
    ]
    records: dict[str, list[dict[str, Any]]] = {
        name: [] for name in model_names
    }
    all_indices = np.arange(windows.shape[0], dtype=np.int64)
    for evaluation_index in all_indices:
        candidates_by_model = {
            "global_leave_one_out": all_indices[all_indices != evaluation_index]
        }
        for radius in radii:
            candidates_by_model[f"local_radius_{radius:g}s"] = all_indices[
                (all_indices != evaluation_index)
                & (
                    np.abs(starts_seconds - starts_seconds[evaluation_index])
                    <= radius
                )
            ]
        for model_name, candidate_indices in candidates_by_model.items():
            if candidate_indices.size < minimum_calibration:
                records[model_name].append(
                    {
                        "evaluation_index": int(evaluation_index),
                        "start_seconds": float(starts_seconds[evaluation_index]),
                        "available": False,
                        "n_calibration_candidates": int(candidate_indices.size),
                    }
                )
                continue
            quality_keep, quality = _precomputed_window_quality(
                quality_features,
                candidate_indices,
                role=f"{model_name}_calibration",
                quality_config=quality_config,
            )
            calibration_indices = candidate_indices[quality_keep]
            if calibration_indices.size < minimum_calibration:
                records[model_name].append(
                    {
                        "evaluation_index": int(evaluation_index),
                        "start_seconds": float(starts_seconds[evaluation_index]),
                        "available": False,
                        "n_calibration_candidates": int(candidate_indices.size),
                        "n_calibration_windows": int(calibration_indices.size),
                    }
                )
                continue
            psd = _aggregate_periodograms(
                periodograms,
                calibration_indices,
                average=average,
                floor_fraction=floor_fraction,
            )
            weights = inverse_psd_weights(psd, template.size)
            if highpass_hz is not None:
                weights = weights.copy()
                weights[frequencies < highpass_hz] = 0.0
            amplitude_sigma = float(
                np.sqrt(
                    psd_amplitude_variance(
                        template_f,
                        psd,
                        weights,
                        trace_len=template.size,
                        sampling_frequency=sampling_frequency,
                    )
                )
            )
            amplitude = float(
                gls_amplitude(
                    np.fft.rfft(windows[evaluation_index]),
                    template_f,
                    weights,
                )
            )
            records[model_name].append(
                {
                    "evaluation_index": int(evaluation_index),
                    "start_seconds": float(starts_seconds[evaluation_index]),
                    "available": True,
                    "n_calibration_candidates": int(candidate_indices.size),
                    "n_calibration_windows": int(calibration_indices.size),
                    "calibration_indices": calibration_indices.tolist(),
                    "amplitude_sigma": amplitude_sigma,
                    "amplitude": amplitude,
                    "score": amplitude / amplitude_sigma,
                    "calibration_quality": quality,
                    "spectral": _spectral_diagnostics(
                        periodograms[evaluation_index],
                        psd,
                        frequencies,
                        template_weights,
                        bands_hz,
                    ),
                }
            )

    n_blocks = int(local_config.get("chronological_blocks", 5))
    summaries = {}
    for model_name, model_records in records.items():
        available = [record for record in model_records if record["available"]]
        scores = np.asarray(
            [record["score"] for record in available],
            dtype=np.float64,
        )
        blocks = []
        ordered = sorted(available, key=lambda record: record["start_seconds"])
        for block_id, block_records in enumerate(
            np.array_split(np.asarray(ordered, dtype=object), n_blocks)
        ):
            block_scores = np.asarray(
                [record["score"] for record in block_records],
                dtype=np.float64,
            )
            blocks.append(
                {
                    "block_id": block_id,
                    "n_windows": int(block_scores.size),
                    "start_seconds": float(block_records[0]["start_seconds"]),
                    "stop_seconds": float(block_records[-1]["start_seconds"]),
                    "score_mean": float(np.mean(block_scores)),
                    "score_std": (
                        float(np.std(block_scores, ddof=1))
                        if block_scores.size > 1
                        else 0.0
                    ),
                }
            )
        summaries[model_name] = {
            "available_windows": int(len(available)),
            "unavailable_windows": int(len(model_records) - len(available)),
            "scores": _distribution(scores),
            "chronological_blocks": blocks,
            "is_predeclared_primary": model_name
            == f"local_radius_{primary_radius:g}s",
        }
    return {
        "radii_seconds": radii,
        "primary_radius_seconds": primary_radius,
        "minimum_calibration_windows": minimum_calibration,
        "narrow_bands_hz": bands_hz,
        "records": records,
        "summaries": summaries,
    }


def run_time_local_noise_model(
    config: dict[str, Any],
    data_root_path: str | Path,
) -> dict[str, Any]:
    """Compare global and predeclared local PSD radii on synthetic and real data."""
    synthetic_config = dict(config.get("synthetic_control", {}))
    synthetic_sample_rate = float(
        synthetic_config.get("sample_rate_hz", 512.0)
    )
    synthetic_duration = float(
        synthetic_config.get(
            "duration_seconds",
            config.get("analysis_duration_seconds", 4.0),
        )
    )
    synthetic_length = int(round(synthetic_duration * synthetic_sample_rate))
    synthetic_windows = int(synthetic_config.get("n_windows", 64))
    synthetic_template, synthetic_waveform = build_waveform(
        synthetic_length,
        synthetic_sample_rate,
        config.get("waveform"),
        data_root=data_root_path,
    )
    _, synthetic_psd = make_powerlaw_psd(
        synthetic_length,
        synthetic_sample_rate,
        kind=str(synthetic_config.get("noise_kind", "pink")),
        level=float(synthetic_config.get("noise_level", 1e-42)),
        knee_hz=synthetic_config.get("knee_hz"),
    )
    synthetic_rng = np.random.default_rng(
        int(synthetic_config.get("seed", 20260623))
    )
    synthetic_traces = generate_colored_noise(
        synthetic_rng,
        synthetic_psd,
        synthetic_length,
        synthetic_sample_rate,
        synthetic_windows,
    )
    synthetic_starts = (
        np.arange(synthetic_windows, dtype=np.float64) * synthetic_duration
    )
    synthetic_result = _local_model_records(
        synthetic_traces,
        synthetic_starts,
        synthetic_sample_rate,
        synthetic_template,
        config,
    )

    real_results = {}
    for detector in config.get("detectors", ["H1", "L1"]):
        real = _load_real_offsource(config, data_root_path, detector)
        template, waveform_metadata = build_waveform(
            real["length"],
            real["sample_rate"],
            config.get("waveform"),
            data_root=data_root_path,
        )
        real_results[detector] = {
            "cache_file": real["cache_file"],
            "data_quality": real["data_quality"],
            "waveform": waveform_metadata,
            **_local_model_records(
                real["windows"],
                real["starts_seconds"],
                real["sample_rate"],
                template,
                config,
            ),
        }

    acceptance_config = dict(config.get("acceptance", {}))
    synthetic_bounds = [
        float(value)
        for value in acceptance_config.get(
            "synthetic_score_std_bounds",
            [0.8, 1.2],
        )
    ]
    real_bounds = [
        float(value)
        for value in acceptance_config.get(
            "real_primary_score_std_bounds",
            [0.8, 1.2],
        )
    ]
    primary_radius = float(config["local_psd"]["primary_radius_seconds"])
    primary_name = f"local_radius_{primary_radius:g}s"
    synthetic_primary_std = float(
        synthetic_result["summaries"][primary_name]["scores"]["std"]
    )
    minimum_coverage = float(
        acceptance_config.get("minimum_primary_coverage_fraction", 0.9)
    )
    synthetic_primary_coverage = (
        synthetic_result["summaries"][primary_name]["available_windows"]
        / synthetic_windows
    )
    synthetic_passed = (
        synthetic_bounds[0] <= synthetic_primary_std <= synthetic_bounds[1]
        and synthetic_primary_coverage >= minimum_coverage
    )
    real_primary_coverage = {
        detector: (
            metrics["summaries"][primary_name]["available_windows"]
            / len(metrics["records"][primary_name])
        )
        for detector, metrics in real_results.items()
    }
    real_passed = {
        detector: (
            real_bounds[0]
            <= float(metrics["summaries"][primary_name]["scores"]["std"])
            <= real_bounds[1]
            and real_primary_coverage[detector] >= minimum_coverage
        )
        for detector, metrics in real_results.items()
    }
    real_required = bool(
        acceptance_config.get("real_primary_required", True)
    )
    return {
        "experiment": str(
            config.get(
                "experiment_id",
                "P1_GWOSC_TIME_LOCAL_NOISE",
            )
        ),
        "waveform": {
            "synthetic": synthetic_waveform,
            "real": {
                detector: metrics["waveform"]
                for detector, metrics in real_results.items()
            },
        },
        "predeclared_local_psd": config.get("local_psd", {}),
        "synthetic_control": {
            "sampling_frequency_hz": synthetic_sample_rate,
            "duration_seconds": synthetic_duration,
            "n_windows": synthetic_windows,
            "seed": int(synthetic_config.get("seed", 20260623)),
            **synthetic_result,
        },
        "real_data": real_results,
        "acceptance": {
            "passed": bool(
                synthetic_passed
                and (not real_required or all(real_passed.values()))
            ),
            "criterion": (
                "the predeclared primary local PSD radius calibrates held-out "
                "template scores after a stationary synthetic control"
            ),
            "primary_model": primary_name,
            "synthetic_control": synthetic_passed,
            "synthetic_score_std_bounds": synthetic_bounds,
            "synthetic_primary_coverage_fraction": (
                synthetic_primary_coverage
            ),
            "real_data": real_passed,
            "real_primary_coverage_fraction": real_primary_coverage,
            "real_primary_required": real_required,
            "real_primary_score_std_bounds": real_bounds,
            "minimum_primary_coverage_fraction": minimum_coverage,
            "other_radii_are_sensitivity_analyses": True,
        },
    }
