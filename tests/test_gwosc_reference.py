import importlib.util

import numpy as np
import pytest

from src.noise_geometry.gwosc import (
    run_gwosc_experiment,
    run_gwpy_reference_check,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("gwpy") is None,
    reason="GWpy is an optional dependency",
)


def test_gwpy_reference_matches_repository_psd_normalization():
    rng = np.random.default_rng(20260619)
    calibration = rng.normal(size=(16, 1024))
    evaluation = rng.normal(size=1024)

    result = run_gwpy_reference_check(
        calibration,
        evaluation,
        sampling_frequency=256.0,
        fduration_seconds=0.5,
    )

    assert result["psd"]["frequency_max_abs_difference_hz"] == 0.0
    assert result["psd"]["relative_l2_error"] < 1e-12
    assert abs(result["psd"]["ratio_median"] - 1.0) < 1e-12
    assert 0.7 < result["whitening"]["repository_interior_std"] < 1.3
    assert 0.7 < result["whitening"]["gwpy_interior_std"] < 1.3
    assert 0.9 < result["whitening"]["std_ratio_repository_over_gwpy"] < 1.1
    assert result["whitening"]["interior_correlation"] > 0.98


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

    result = run_gwosc_experiment(
        {
            "experiment_id": "P1_GWPY_REFERENCE_TEST",
            "event": "GWTEST",
            "detectors": ["H1"],
            "analysis_duration_seconds": 2.0,
            "injection_snr": 5.0,
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
    assert reference["psd"]["relative_l2_error"] < 1e-12
    assert np.isfinite(reference["whitening"]["std_ratio_repository_over_gwpy"])
