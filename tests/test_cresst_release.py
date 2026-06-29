"""Tests for the public CRESST release loader, runner path, and downloader."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from src.noise_geometry.cresst import (
    load_cresst_release,
    run_cresst_experiment,
    select_cresst_subsets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_release(raw: Path, *, n_noise: int = 120, n_pulse: int = 220, n_samples: int = 256):
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, n_samples)
    pulse = np.exp(-((t - 0.45) ** 2) / 0.006)
    offset = rng.uniform(-2.0, 2.0, size=(n_noise + n_pulse, 1))
    noise = rng.normal(scale=0.03, size=(n_noise, n_samples)) + offset[:n_noise]
    pulses = rng.normal(scale=0.03, size=(n_pulse, n_samples))
    pulses += rng.uniform(0.5, 1.5, size=(n_pulse, 1)) * pulse[None, :] + offset[n_noise:]
    X = np.vstack([noise, pulses]).astype(np.float32)
    np.save(raw / "X_test.npy", X)
    np.save(raw / "y_test.npy", np.concatenate([-np.ones(n_noise), np.ones(n_pulse)]).astype(int))

    is_noise = ["True"] * n_noise + ["False"] * n_pulse
    is_clean = ["False"] * n_noise + ["True"] * n_pulse
    lines = ["run,channel,noise,clean,pulse_height"]
    for i in range(n_noise + n_pulse):
        lines.append(f"34,1,{is_noise[i]},{is_clean[i]},{float(np.max(X[i])):.4f}")
    (raw / "features_test.csv").write_text("\n".join(lines) + "\n")
    return n_noise, n_pulse


def test_load_release_and_subsets(tmp_path):
    raw = tmp_path / "cresst" / "raw"
    n_noise, n_pulse = _write_release(raw)
    traces, features, meta = load_cresst_release(raw, split="test", mmap=True)
    assert traces.shape == (n_noise + n_pulse, 256)
    assert len(features) == n_noise + n_pulse
    assert meta["n_total"] == n_noise + n_pulse

    subsets = select_cresst_subsets(traces, features)
    assert subsets["n_noise"] == n_noise
    assert subsets["n_pulse"] == n_pulse
    assert subsets["noise_traces"].shape[0] == n_noise
    assert subsets["pulse_traces"].shape[0] == n_pulse


def test_release_subsample_is_reproducible(tmp_path):
    raw = tmp_path / "cresst" / "raw"
    _write_release(raw)
    a, fa, _ = load_cresst_release(raw, split="test", max_traces=50, seed=7)
    b, fb, _ = load_cresst_release(raw, split="test", max_traces=50, seed=7)
    assert a.shape[0] == 50
    np.testing.assert_array_equal(a, b)


def test_run_cresst_experiment_release_path(tmp_path):
    raw = tmp_path / "cresst" / "raw"
    _write_release(raw)
    result = run_cresst_experiment(
        {
            "experiment_id": "P2_CRESST_TEST",
            "release_format": True,
            "split": "test",
            "max_features": 64,
            "rank": 3,
            "covariance_shrinkage": 0.2,
            "seed": 1,
        },
        tmp_path,
    )
    assert result["n_noise_traces"] > 10
    assert result["n_features"] == 64
    assert result["ae_empca_max_principal_angle_deg"] < 1e-3
    assert np.isfinite(result["empca_weighted_residual"])


def _load_downloader():
    path = REPO_ROOT / "scripts" / "download" / "download_cresst.py"
    spec = importlib.util.spec_from_file_location("download_cresst", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # required for dataclass annotation resolution
    spec.loader.exec_module(module)
    return module


def test_downloader_verify_npy_shape(tmp_path):
    dl = _load_downloader()
    good = tmp_path / "X_test.npy"
    np.save(good, np.zeros((10, 512), dtype=np.float32))
    spec_ok = dl.ReleaseFile("X_test.npy", "test", "npy", shape=(10, 512), dtype="float32")
    record = dl._verify(spec_ok, good)
    assert record["shape"] == [10, 512]
    assert "sha256" in record

    spec_bad = dl.ReleaseFile("X_test.npy", "test", "npy", shape=(11, 512))
    with pytest.raises(ValueError):
        dl._verify(spec_bad, good)


def test_downloader_verify_csv_columns(tmp_path):
    dl = _load_downloader()
    csv = tmp_path / "features_test.csv"
    csv.write_text("run,channel,noise,clean,pulse_height\n34,1,True,False,0.1\n")
    spec_ok = dl.ReleaseFile("features_test.csv", "test", "csv", rows=1,
                             required_columns=("run", "noise", "clean"))
    record = dl._verify(spec_ok, csv)
    assert record["rows"] == 1

    spec_missing = dl.ReleaseFile("features_test.csv", "test", "csv",
                                  required_columns=("does_not_exist",))
    with pytest.raises(ValueError):
        dl._verify(spec_missing, csv)
