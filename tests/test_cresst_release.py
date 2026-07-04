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
    rise_time = list(rng.uniform(0.02, 0.08, size=n_noise + n_pulse))
    decay_time = list(rng.uniform(0.1, 0.3, size=n_noise + n_pulse))
    # Hours decoupled from row order/type, so noise and pulse rows are both
    # spread across the full time range within each channel (a real
    # chronological split needs background events on both sides of it).
    hours = rng.permutation(n_noise + n_pulse).astype(float)
    lines = ["run,channel,noise,clean,pulse_height,rise_time,decay_time,hours"]
    for i in range(n_noise + n_pulse):
        # Two channels (not one) so the per-channel time-disjoint split has >= 2 channels.
        channel = 1 if i % 2 == 0 else 2
        lines.append(
            f"34,{channel},{is_noise[i]},{is_clean[i]},{float(np.max(X[i])):.4f},"
            f"{rise_time[i]:.6f},{decay_time[i]:.6f},{hours[i]:.6f}"
        )
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


def _write_grouped_release(raw: Path, *, groups, n_per_group=300, n_samples=128, seed=0, missing_resolution_for=None):
    """Multi-channel release fixture for the Stage C3 per-channel validation pipeline.

    Each (run, channel) is an independent replication with its own noise
    level and OF resolution; ``hours`` is decoupled from row order/type so a
    chronological split leaves both noise and pulse rows on each side.
    """
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_samples)
    blocks, rows, resolution_rows = [], [], []
    for gi, (run, channel) in enumerate(groups):
        noise_scale = 0.02 + 0.01 * gi  # distinct per-channel noise level, like real detectors
        n_noise = max(20, n_per_group // 6)
        n_pulse = n_per_group - n_noise
        offset = rng.uniform(-2.0, 2.0, size=(n_noise + n_pulse, 1))
        noise = rng.normal(scale=noise_scale, size=(n_noise, n_samples)) + offset[:n_noise]
        rise = rng.uniform(0.02, 0.08, size=n_pulse)
        decay = rng.uniform(0.1, 0.3, size=n_pulse)
        pulses = np.zeros((n_pulse, n_samples))
        for i in range(n_pulse):
            pulses[i] = np.where(
                t < 0.45,
                np.exp(-((t - 0.45) ** 2) / (2 * rise[i] ** 2)),
                np.exp(-(t - 0.45) / decay[i]),
            )
        amp = rng.uniform(0.5, 1.5, size=(n_pulse, 1))
        pulses = pulses * amp + offset[n_noise:] + rng.normal(scale=noise_scale, size=pulses.shape)
        block = np.vstack([noise, pulses]).astype(np.float32)
        blocks.append(block)
        is_noise = ["True"] * n_noise + ["False"] * n_pulse
        is_clean = ["False"] * n_noise + ["True"] * n_pulse
        rise_col = [0.0] * n_noise + list(rise)
        decay_col = [0.0] * n_noise + list(decay)
        hours = rng.permutation(n_noise + n_pulse).astype(float)
        for i in range(n_noise + n_pulse):
            rows.append(
                f"{run},{channel},{is_noise[i]},{is_clean[i]},"
                f"{float(np.max(block[i])):.4f},{rise_col[i]:.6f},{decay_col[i]:.6f},{hours[i]:.6f}"
            )
        if (run, channel) != missing_resolution_for:
            resolution_rows.append(f"{run},{channel},{noise_scale:.6f}")
    X = np.vstack(blocks)
    np.save(raw / "X_test.npy", X)
    lines = ["run,channel,noise,clean,pulse_height,rise_time,decay_time,hours", *rows]
    (raw / "features_test.csv").write_text("\n".join(lines) + "\n")
    res_lines = ["run,channel,of_resolution", *resolution_rows]
    (raw / "resolutions_test.csv").write_text("\n".join(res_lines) + "\n")
    return X.shape[0]


def test_run_cresst_experiment_validation_pipeline(tmp_path):
    raw = tmp_path / "cresst" / "raw"
    groups = [(33, 36), (33, 37), (35, 26), (35, 27), (35, 56)]
    n_total = _write_grouped_release(raw, groups=groups, n_per_group=300)

    config = {
        "experiment_id": "P2_CRESST_VALIDATION_TEST",
        "release_format": True,
        "split": "test",
        "seed": 22,
        "rank": 3,
        "max_traces": None,
        "max_features": 96,
        "baseline_fraction": 0.2,
        "covariance_shrinkage": 0.1,
        "channel_calibration_fraction": 0.7,
        "template_bank": {
            "rise_time_quantiles": [0.5],
            "decay_time_quantiles": [0.5],
            "t0_grid_samples": [-2, 0, 2],
            "min_cell_pulses": 2,
        },
        "fpr_points": [0.01, 0.05],
        "bootstrap_reps": 30,
        "detector_metrics": {
            "amplitude_grid_multiples_of_sigma_of": [0, 1, 2, 4],
            "n_injections_per_amplitude": 40,
        },
    }
    result = run_cresst_experiment(config, tmp_path)

    assert result["n_traces"] <= n_total
    assert result["n_channels_total"] == len(groups)
    assert result["n_channels_used"] >= 1
    split = result["split"]
    assert split["method"] == "per_channel_time_disjoint"
    assert len(split["channels"]) == len(groups)
    assert result["n_duplicate_traces"] == 0

    used_channels = [c for c in result["per_channel"].values() if not c["skipped"]]
    assert used_channels, "expected at least one usable channel"
    for channel_result in used_channels:
        assert channel_result["sigma_of_source"] == "resolutions_test.csv"
        assert channel_result["multi_template_of"]["n_templates"] >= 1
        assert np.isfinite(channel_result["multi_template_of"]["weighted_residual_mean"])
        assert set(channel_result["negative_controls"].keys()) == {
            "wrong_psd", "permuted_time_bins", "random_weighted_basis",
            "noise_only_train", "mismatched_of_template",
        }
        dm = channel_result["detector_metrics"]
        assert len(dm["residual_whitening_ratio"]["frequency_hz"]) > 0
        trigger = dm["trigger_efficiency"]
        assert trigger["fpr_points"] == [0.01, 0.05]
        assert "direct_injection" in trigger and "measured_psd_injection" in trigger

    assert "pooled" in result
    assert "of_vs_empca" in result["pooled"]
    assert "acceptance" in result
    assert isinstance(result["acceptance"]["passed"], bool)
    assert result["acceptance"]["fpr_diagnostic_only"] == [0.001]  # default, not overridden here

    import json

    json.dumps(result)  # must be JSON-serializable, matching write_run_record


def test_missing_channel_resolution_falls_back_to_crb_estimate(tmp_path):
    """A real release gap: e.g. test-split channel (35, 56) has no row in
    resolutions_test.csv. The fallback must be the analytic CRB amplitude-
    estimator std (same physical definition as the published resolution),
    not the spread of physical pulse amplitudes -- those differ by ~100-1000x
    for this detector (of_resolution is a noise-floor precision, pulse_height
    is the signal scale), so a same-scale fallback is essential for the
    amplitude grid to remain SNR-comparable across channels.
    """
    raw = tmp_path / "cresst" / "raw"
    groups = [(33, 36), (33, 37)]
    _write_grouped_release(raw, groups=groups, n_per_group=300, missing_resolution_for=(33, 37))

    config = {
        "experiment_id": "P2_CRESST_MISSING_RES_TEST",
        "release_format": True,
        "split": "test",
        "seed": 22,
        "rank": 3,
        "max_traces": None,
        "max_features": 96,
        "channel_calibration_fraction": 0.7,
        "template_bank": {"rise_time_quantiles": [0.5], "decay_time_quantiles": [0.5], "min_cell_pulses": 2},
        "fpr_points": [0.05],
        "bootstrap_reps": 20,
        "detector_metrics": {"amplitude_grid_multiples_of_sigma_of": [0, 1, 2], "n_injections_per_amplitude": 20},
    }
    result = run_cresst_experiment(config, tmp_path)
    channel_37 = result["per_channel"]["33_37"]
    assert not channel_37["skipped"]
    assert channel_37["sigma_of_source"] == "fallback_of_crb_no_published_resolution"
    channel_36 = result["per_channel"]["33_36"]
    assert channel_36["sigma_of_source"] == "resolutions_test.csv"
    # the fallback should land within an order of magnitude of the published
    # sibling channel's resolution, not the ~100-1000x-larger pulse-amplitude scale
    assert 0.1 < channel_37["sigma_of"] / channel_36["sigma_of"] < 10


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


def test_downloader_verify_npy_headerless_raw(tmp_path):
    """Some DMDC-hosted files carry .npy names but are headerless tofile() dumps."""
    dl = _load_downloader()
    raw = tmp_path / "X_test.npy"
    arr = np.arange(10 * 512, dtype=np.float32).reshape(10, 512)
    arr.tofile(raw)  # no npy magic/header, unlike np.save
    spec = dl.ReleaseFile("X_test.npy", "test", "npy", shape=(10, 512), dtype="float32")
    record = dl._verify(spec, raw)
    assert record["shape"] == [10, 512]
    assert record["format"] == "raw_headerless"

    spec_bad = dl.ReleaseFile("X_test.npy", "test", "npy", shape=(11, 512), dtype="float32")
    with pytest.raises(ValueError):
        dl._verify(spec_bad, raw)


def test_load_release_headerless_raw_traces(tmp_path):
    """load_cresst_release must also tolerate headerless X_{split}.npy dumps."""
    raw = tmp_path / "cresst" / "raw"
    raw.mkdir(parents=True)
    n_noise, n_pulse, n_samples = 5, 7, 256
    n = n_noise + n_pulse
    X = np.random.default_rng(0).normal(size=(n, n_samples)).astype(np.float32)
    X.tofile(raw / "X_test.npy")  # headerless, unlike np.save

    is_noise = ["True"] * n_noise + ["False"] * n_pulse
    is_clean = ["False"] * n_noise + ["True"] * n_pulse
    lines = ["run,channel,noise,clean,pulse_height"]
    for i in range(n):
        lines.append(f"34,1,{is_noise[i]},{is_clean[i]},{float(np.max(X[i])):.4f}")
    (raw / "features_test.csv").write_text("\n".join(lines) + "\n")

    traces, features, meta = load_cresst_release(raw, split="test", mmap=True)
    assert traces.shape == (n, n_samples)
    np.testing.assert_allclose(np.asarray(traces), X)
    assert meta["n_total"] == n


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
