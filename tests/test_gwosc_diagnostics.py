import json

import numpy as np

from src.noise_geometry.gwosc.diagnostics import (
    _local_model_records,
    compare_filter_statistics,
    run_filter_statistic_equivalence,
    run_time_local_noise_model,
)
from src.noise_geometry.gwosc.waveforms import build_waveform
from src.noise_geometry.noise import (
    generate_colored_noise,
    make_powerlaw_psd,
)


def _write_cached_fixture(tmp_path, *, sample_rate=128.0, duration=128.0):
    n = int(sample_rate * duration)
    start = 2000.0
    gps = start + duration / 2.0
    rng = np.random.default_rng(20260623)
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


def test_public_text_waveform_is_resampled_and_documented(tmp_path):
    waveform_dir = tmp_path / "gwosc/raw/GWTEST/waveforms"
    waveform_dir.mkdir(parents=True)
    source_rate = 512.0
    time = np.arange(256) / source_rate
    values = np.sin(2.0 * np.pi * 40.0 * time)
    source = waveform_dir / "waveform.txt"
    np.savetxt(source, np.column_stack([time, values]))

    waveform, metadata = build_waveform(
        512,
        256.0,
        {
            "type": "public_text",
            "relative_path": "gwosc/raw/GWTEST/waveforms/waveform.txt",
            "source_url": "https://example.invalid/waveform.txt",
            "value_scale": 1.0,
            "peak_time_seconds": 1.0,
            "normalization": "peak",
        },
        data_root=tmp_path,
    )

    assert waveform.shape == (512,)
    assert np.max(np.abs(waveform)) == 1.0
    assert metadata["source_sample_rate_hz"] == 512.0
    assert metadata["target_sample_rate_hz"] == 256.0
    assert len(metadata["sha256"]) == 64


def test_shared_fir_statistic_is_identical_across_paths():
    sample_rate = 256.0
    length = 1024
    template, _ = build_waveform(
        length,
        sample_rate,
        {
            "type": "sine_gaussian",
            "central_frequency_hz": 40.0,
            "quality_factor": 8.0,
        },
    )
    _, psd = make_powerlaw_psd(
        length,
        sample_rate,
        kind="pink",
        level=1e-3,
    )
    traces = generate_colored_noise(
        np.random.default_rng(5),
        psd,
        length,
        sample_rate,
        24,
    )

    result = compare_filter_statistics(
        traces,
        template,
        psd,
        sample_rate,
        fduration_seconds=0.5,
        edge_trim_seconds=0.25,
        highpass_hz=10.0,
        detrend="constant",
        floor_fraction=1e-6,
    )

    assert result["identity"]["max_abs_score_difference"] < 1e-10
    assert result["identity"]["relative_l2_score_difference"] < 1e-10
    assert result["identity"]["score_correlation"] > 1.0 - 1e-12
    assert result["original_gls_vs_shared_fir"]["correlation"] > 0.99


def test_local_psd_records_template_and_narrow_band_diagnostics():
    sample_rate = 128.0
    length = 256
    n_windows = 32
    template, _ = build_waveform(
        length,
        sample_rate,
        {
            "type": "sine_gaussian",
            "central_frequency_hz": 30.0,
            "quality_factor": 6.0,
        },
    )
    _, psd = make_powerlaw_psd(
        length,
        sample_rate,
        kind="white",
        level=1e-3,
    )
    windows = generate_colored_noise(
        np.random.default_rng(8),
        psd,
        length,
        sample_rate,
        n_windows,
    )
    config = {
        "analysis_highpass_hz": 10.0,
        "psd_floor_fraction": 1e-6,
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
        "local_psd": {
            "radii_seconds": [8.0, 16.0, 24.0],
            "primary_radius_seconds": 16.0,
            "minimum_calibration_windows": 4,
            "chronological_blocks": 4,
            "narrow_bands_hz": [[10.0, 30.0], [30.0, 60.0]],
        },
    }

    result = _local_model_records(
        windows,
        np.arange(n_windows) * 2.0,
        sample_rate,
        template,
        config,
    )

    primary = result["summaries"]["local_radius_16s"]
    assert primary["available_windows"] == n_windows
    assert primary["is_predeclared_primary"]
    record = result["records"]["local_radius_16s"][0]
    assert "template_projected_psd_ratio" in record["spectral"]
    assert len(record["spectral"]["narrow_bands"]) == 2


def test_diagnostic_experiment_entry_points_use_cached_real_windows(tmp_path):
    _write_cached_fixture(tmp_path)
    common = {
        "event": "GWTEST",
        "detectors": ["H1"],
        "analysis_duration_seconds": 2.0,
        "analysis_highpass_hz": 10.0,
        "waveform": {
            "type": "sine_gaussian",
            "central_frequency_hz": 30.0,
            "quality_factor": 6.0,
            "peak_time_seconds": 1.0,
            "normalization": "peak",
        },
        "psd_floor_fraction": 1e-6,
        "psd_estimator": {
            "window": "hann",
            "average": "median",
            "detrend": "constant",
        },
        "psd_calibration_fraction": 0.75,
        "offsource_split_seed": 1,
        "minimum_psd_calibration_windows": 24,
        "minimum_evaluation_windows": 12,
        "data_quality": {"required": True},
    }
    filter_result = run_filter_statistic_equivalence(
        {
            **common,
            "fir_sweep": {
                "durations_seconds": [0.25, 0.5],
                "edge_trims_seconds": [0.125, 0.25],
                "window": "hann",
                "detrend": "constant",
                "primary_duration_seconds": 0.5,
                "primary_edge_trim_seconds": 0.25,
            },
            "synthetic_control": {
                "sample_rate_hz": 128.0,
                "duration_seconds": 2.0,
                "n_evaluation_traces": 16,
                "noise_kind": "white",
                "noise_level": 1e-3,
                "seed": 3,
            },
            "acceptance": {
                "max_abs_identity_difference": 1e-10,
                "max_identity_relative_l2": 1e-10,
            },
        },
        tmp_path,
    )
    assert filter_result["acceptance"]["passed"]
    assert len(filter_result["real_data"]["H1"]["sweep"]) == 4
    assert sum(
        result["is_predeclared_primary"]
        for result in filter_result["real_data"]["H1"]["sweep"]
    ) == 1

    local_result = run_time_local_noise_model(
        {
            **common,
            "psd_quality": {
                "enabled": True,
                "band_min_hz": 10.0,
                "band_max_hz": 60.0,
                "robust_z_threshold": 6.0,
                "crest_factor_threshold": 20.0,
                "max_rejected_fraction": 0.25,
            },
            "local_psd": {
                "radii_seconds": [8.0, 16.0, 24.0],
                "primary_radius_seconds": 16.0,
                "minimum_calibration_windows": 4,
                "chronological_blocks": 4,
                "narrow_bands_hz": [[10.0, 30.0], [30.0, 60.0]],
            },
            "synthetic_control": {
                "sample_rate_hz": 128.0,
                "duration_seconds": 2.0,
                "n_windows": 32,
                "noise_kind": "white",
                "noise_level": 1e-3,
                "seed": 4,
            },
            "acceptance": {
                "synthetic_score_std_bounds": [0.4, 2.0],
                "real_primary_score_std_bounds": [0.1, 5.0],
                "minimum_primary_coverage_fraction": 0.5,
                "real_primary_required": False,
            },
        },
        tmp_path,
    )
    assert local_result["acceptance"]["synthetic_control"]
    assert "local_radius_16s" in local_result["real_data"]["H1"]["summaries"]
