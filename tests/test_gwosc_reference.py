import importlib.util
import json

import numpy as np
import pytest

from src.noise_geometry.gwosc import (
    run_gwosc_experiment,
    run_gwpy_reference_check,
)
from src.noise_geometry.gwosc.analysis import _calibration_window_quality
from src.noise_geometry.gwosc.analysis import (
    _apply_data_quality,
    _window_quality_diagnostics,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("gwpy") is None,
    reason="GWpy is an optional dependency",
)


def test_gwpy_reference_matches_repository_psd_normalization():
    rng = np.random.default_rng(20260619)
    calibration = rng.normal(size=(16, 1024))
    evaluation = rng.normal(size=(4, 1024))
    time = np.arange(1024) / 256.0
    template = np.sin(2.0 * np.pi * 40.0 * time) * np.hanning(1024)

    result = run_gwpy_reference_check(
        calibration,
        evaluation,
        sampling_frequency=256.0,
        fduration_seconds=0.5,
        template=template,
    )

    assert result["psd"]["frequency_max_abs_difference_hz"] == 0.0
    assert result["psd"]["relative_l2_error"] < 1e-12
    assert abs(result["psd"]["ratio_median"] - 1.0) < 1e-12
    assert result["configuration"]["n_evaluation_traces"] == 4
    assert 0.7 < result["whitening"]["repository_interior_std"] < 1.3
    assert 0.7 < result["whitening"]["gwpy_interior_std"] < 1.3
    assert 0.9 < result["whitening"]["std_ratio_repository_over_gwpy"] < 1.1
    assert result["whitening"]["interior_correlation"] > 0.98
    assert len(result["whitening"]["repository_std_by_window"]["values"]) == 4
    matched = result["matched_filter"]
    assert matched["n_evaluation_traces"] == 4
    assert len(matched["repository_gls_scores"]) == 4
    assert matched["correlation_repository_gls_vs_whitened"] > 0.95
    assert np.isfinite(matched["correlation_repository_gls_vs_gwpy"])
    assert not matched["used_for_acceptance"]


def test_cached_gwosc_runner_records_gwpy_reference(tmp_path):
    sample_rate = 256.0
    duration = 16.0
    n = int(sample_rate * duration)
    start = 1000.0
    gps = start + duration / 2.0
    times = start + np.arange(n) / sample_rate
    rng = np.random.default_rng(9)
    raw = tmp_path / "gwosc/raw/GWTEST"
    raw.mkdir(parents=True)
    np.savez_compressed(
        raw / "GWTEST_H1_16s.npz",
        value=rng.normal(scale=1e-21, size=n),
        times=times,
        sample_rate=sample_rate,
        detector="H1",
        event="GWTEST",
        gps=gps,
        start=start,
        end=start + duration,
    )
    short_duration = 8.0
    short_start = gps - short_duration / 2.0
    short_n = int(sample_rate * short_duration)
    np.savez_compressed(
        raw / "GWTEST_H1_8s.npz",
        value=rng.normal(scale=1e-21, size=short_n),
        times=short_start + np.arange(short_n) / sample_rate,
        sample_rate=sample_rate,
        detector="H1",
        event="GWTEST",
        gps=gps,
        start=short_start,
        end=short_start + short_duration,
    )
    (raw / "metadata.json").write_text(
        json.dumps(
            {
                "data_quality": {
                    "H1": {
                        "flag": "H1_DATA",
                        "segments": [[start, start + duration]],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_gwosc_experiment(
        {
            "experiment_id": "P1_GWPY_REFERENCE_TEST",
            "event": "GWTEST",
            "detectors": ["H1"],
            "analysis_duration_seconds": 2.0,
            "injection_snr": 5.0,
            "data_quality": {"required": True},
            "gwpy_reference": {
                "enabled": True,
                "fduration_seconds": 0.5,
                "highpass_hz": 20.0,
            },
        },
        tmp_path,
    )

    reference = result["detectors"]["H1"]["gwpy_reference"]
    assert reference["configuration"]["n_calibration_traces"] >= 1
    assert reference["configuration"]["n_evaluation_traces"] >= 2
    assert reference["psd"]["relative_l2_error"] < 1e-12
    assert np.isfinite(reference["whitening"]["std_ratio_repository_over_gwpy"])
    metrics = result["detectors"]["H1"]
    assert metrics["cache_file"].endswith("GWTEST_H1_16s.npz")
    assert metrics["data_quality"]["available"]
    assert metrics["data_quality"]["event_window"]["valid"]
    assert metrics["n_psd_calibration_windows"] >= 1
    assert metrics["n_injection_windows"] >= 2
    assert metrics["event_amplitude_sigma"] > 0
    assert metrics["injection_amplitude"] == pytest.approx(
        metrics["injection_target_snr"] * metrics["event_amplitude_sigma"]
    )
    assert metrics["injection_paired_score_mean"] == pytest.approx(
        metrics["injection_target_snr"],
        abs=1e-10,
    )
    assert metrics["injection_score_mean"] == pytest.approx(
        metrics["injection_target_snr"],
        abs=1e-10,
    )
    assert abs(metrics["injection_amplitude_bias"]) < (
        abs(metrics["injection_amplitude"]) * 1e-10
    )
    assert metrics["injection_paired_score_std"] < 1e-10


def test_calibration_quality_rejects_impulsive_glitch():
    rng = np.random.default_rng(22)
    windows = rng.normal(size=(20, 1024))
    windows[7, 512] = 100.0

    keep, diagnostics = _calibration_window_quality(
        windows,
        np.arange(windows.shape[0]),
        256.0,
        psd_window="hann",
        psd_detrend="constant",
        highpass_hz=20.0,
        config={
            "enabled": True,
            "band_min_hz": 20.0,
            "band_max_hz": 100.0,
            "robust_z_threshold": 5.0,
            "crest_factor_threshold": 20.0,
            "max_rejected_fraction": 0.25,
        },
    )

    assert not keep[7]
    assert diagnostics["rejected_window_indices"] == [7]
    assert "crest_factor" in diagnostics["windows"][7]["reasons"]


def test_evaluation_quality_uses_calibration_reference():
    rng = np.random.default_rng(31)
    windows = rng.normal(size=(24, 1024))
    calibration_indices = np.arange(20)
    evaluation_indices = np.arange(20, 24)
    windows[22, 512] = 100.0

    _, _, reference = _window_quality_diagnostics(
        windows,
        calibration_indices,
        256.0,
        psd_window="hann",
        psd_detrend="constant",
        highpass_hz=20.0,
        config={
            "enabled": True,
            "band_min_hz": 20.0,
            "band_max_hz": 100.0,
            "robust_z_threshold": 5.0,
            "crest_factor_threshold": 20.0,
            "max_rejected_fraction": 0.25,
        },
        role="calibration",
        enforce_rejection_limit=True,
    )
    keep, diagnostics, _ = _window_quality_diagnostics(
        windows,
        evaluation_indices,
        256.0,
        psd_window="hann",
        psd_detrend="constant",
        highpass_hz=20.0,
        config={
            "enabled": True,
            "band_min_hz": 20.0,
            "band_max_hz": 100.0,
            "robust_z_threshold": 5.0,
            "crest_factor_threshold": 20.0,
            "max_rejected_fraction": 0.25,
        },
        reference=reference,
        role="evaluation",
        enforce_rejection_limit=False,
    )

    assert not keep[2]
    assert diagnostics["rejected_window_indices"] == [22]
    assert diagnostics["role"] == "evaluation"


def test_data_quality_marks_only_fully_covered_windows():
    starts = np.asarray([100.0, 104.0, 108.0, 112.0])
    keep, diagnostics = _apply_data_quality(
        starts,
        4.0,
        {
            "available": True,
            "metadata_file": "metadata.json",
            "flag": "H1_DATA",
            "segments": [[100.0, 110.0], [112.0, 116.0]],
        },
        required=True,
    )

    assert keep.tolist() == [True, True, False, True]
    assert diagnostics["invalid_window_indices"] == [2]


def test_multisplit_null_calibration_passes_stationary_white_noise(tmp_path):
    sample_rate = 128.0
    duration = 128.0
    n = int(sample_rate * duration)
    start = 2000.0
    gps = start + duration / 2.0
    rng = np.random.default_rng(4)
    raw = tmp_path / "gwosc/raw/GWTEST"
    raw.mkdir(parents=True)
    np.savez_compressed(
        raw / "GWTEST_H1_128s.npz",
        value=rng.normal(scale=1e-21, size=n),
        times=start + np.arange(n) / sample_rate,
        sample_rate=sample_rate,
        detector="H1",
        event="GWTEST",
        gps=gps,
        start=start,
        end=start + duration,
    )

    result = run_gwosc_experiment(
        {
            "event": "GWTEST",
            "detectors": ["H1"],
            "analysis_duration_seconds": 2.0,
            "analysis_highpass_hz": 10.0,
            "psd_calibration_fraction": 0.75,
            "offsource_split_seed": 1,
            "minimum_psd_calibration_windows": 24,
            "minimum_evaluation_windows": 12,
            "psd_estimator": {
                "window": "hann",
                "average": "median",
                "detrend": "constant",
            },
            "psd_quality": {
                "enabled": True,
                "band_min_hz": 10.0,
                "band_max_hz": 60.0,
                "robust_z_threshold": 6.0,
                "crest_factor_threshold": 20.0,
                "max_rejected_fraction": 0.25,
            },
            "null_calibration_validation": {
                "split_seeds": [1, 2, 3, 4, 5],
                "per_split_ratio_bounds": [0.5, 1.5],
                "median_ratio_bounds": [0.8, 1.2],
            },
            "blocked_null_calibration_validation": {
                "enabled": True,
                "required_for_acceptance": False,
                "n_splits": 5,
                "calibration_windows": 24,
                "minimum_evaluation_windows": 8,
                "per_split_ratio_bounds": [0.4, 2.0],
                "median_ratio_bounds": [0.7, 1.3],
            },
            "gwpy_reference": {"enabled": False},
        },
        tmp_path,
    )

    gate = result["detectors"]["H1"]["null_calibration_validation"]
    assert result["acceptance"]["passed"]
    assert gate["passed"]
    assert gate["n_split_seeds"] == 5
    assert 0.8 <= gate["median_null_sigma_over_predicted"] <= 1.2
    first_split = gate["splits"][0]
    assert len(first_split["evaluation_windows"]) == first_split[
        "n_evaluation_windows"
    ]
    assert len(first_split["evaluation_indices"]) == first_split[
        "n_evaluation_windows"
    ]
    assert "null_score" in first_split["evaluation_windows"][0]
    blocked = result["detectors"]["H1"][
        "blocked_null_calibration_validation"
    ]
    assert blocked["enabled"]
    assert blocked["n_splits"] == 5
    assert all(
        split["split_kind"] == "chronological_block"
        for split in blocked["splits"]
    )
