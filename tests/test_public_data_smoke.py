from pathlib import Path

import numpy as np

from src.noise_geometry.cresst import run_cresst_experiment
from src.noise_geometry.gwosc import run_gwosc_experiment


def test_gwosc_cached_event_smoke(tmp_path):
    sample_rate = 256.0
    duration = 8.0
    n = int(sample_rate * duration)
    start = 1000.0
    gps = start + duration / 2.0
    times = start + np.arange(n) / sample_rate
    rng = np.random.default_rng(41)
    values = rng.normal(scale=1e-21, size=n)
    raw = tmp_path / "gwosc/raw/GWTEST"
    raw.mkdir(parents=True)
    np.savez_compressed(
        raw / "GWTEST_H1_8s.npz",
        value=values,
        times=times,
        sample_rate=sample_rate,
        detector="H1",
        event="GWTEST",
        gps=gps,
        start=start,
        end=start + duration,
    )
    result = run_gwosc_experiment(
        {
            "experiment_id": "P1_TEST",
            "event": "GWTEST",
            "detectors": ["H1"],
            "analysis_duration_seconds": 2.0,
            "injection_snr": 5.0,
        },
        tmp_path,
    )
    metrics = result["detectors"]["H1"]
    assert metrics["n_offsource_windows"] >= 2
    assert metrics["n_psd_calibration_windows"] == 1
    assert metrics["n_injection_windows"] == 1
    assert np.isfinite(metrics["event_weighted_residual"])
    np.testing.assert_allclose(
        metrics["injection_paired_score_mean"],
        metrics["injection_target_snr"],
        rtol=0.0,
        atol=1e-10,
    )


def test_cresst_npz_reconstruction_smoke(tmp_path):
    rng = np.random.default_rng(42)
    n_traces = 180
    n_samples = 128
    t = np.linspace(0.0, 1.0, n_samples)
    pulse = np.exp(-((t - 0.45) ** 2) / 0.008)
    labels = np.ones(n_traces, dtype=int)
    labels[:50] = 0
    traces = rng.normal(scale=0.03, size=(n_traces, n_samples))
    traces[50:] += rng.uniform(0.5, 1.5, size=(n_traces - 50, 1)) * pulse[None, :]
    raw = tmp_path / "cresst/raw"
    raw.mkdir(parents=True)
    np.savez_compressed(raw / "fixture.npz", traces=traces, labels=labels)

    result = run_cresst_experiment(
        {
            "experiment_id": "P2_TEST",
            "input_file": "fixture.npz",
            "trace_key": "traces",
            "label_key": "labels",
            "noise_label": 0,
            "max_traces": n_traces,
            "max_features": 64,
            "rank": 3,
            "covariance_shrinkage": 0.2,
        },
        tmp_path,
    )
    assert result["n_noise_traces"] > 10
    assert result["ae_empca_max_principal_angle_deg"] < 1e-4
    assert np.isfinite(result["empca_weighted_residual"])
