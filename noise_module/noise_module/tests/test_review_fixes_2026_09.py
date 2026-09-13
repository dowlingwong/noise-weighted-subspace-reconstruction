"""Regression tests for the review findings of 12 Sep 2026.

1. ``build_optimal_filter()`` on the default grid must work whether or not the
   generated Phase 2 artifact is present.
2. The Phase 2 resampling default must not fold the 30 MHz budget into the
   band (a band-limited readout is the physical default); folding stays
   available as an explicit opt-in.
3. ``noise_power`` must never be a silent no-op: absolute composite spectra
   reject a non-unity ``noise_power``, ``psd_scale`` is the level knob there,
   and ``TemporalNoiseWrapper`` scales the level through the correct knob and
   reports the scale it actually applied.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from noise_module import (
    NoiseGenerator,
    TemporalNoiseWrapper,
    White,
    PowerLaw,
)
from noise_module import al2o3_athermal as A
from noise_module.config import NoiseConfig
from noise_module.fit_al2o3_athermal import (
    build_psd_artifact,
    load_all_components,
    native_grid_psd,
)


# --------------------------------------------------------------------------
# 1. default optimal filter never depends on a generated file being present
# --------------------------------------------------------------------------


def test_default_optimal_filter_builds_without_phase2_artifact(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(A, "PSD_FILE", tmp_path / "missing.npy")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        of = A.build_optimal_filter()
    assert any("falling back" in str(w.message) for w in caught)
    assert of.psd.shape == (A.DEFAULT_SAMPLES // 2 + 1,)
    assert np.all(np.isfinite(of.template))
    # analytic composite on the same grid must be what was used
    _, analytic = A.noise_generator().build_psd_density(A.DEFAULT_SAMPLES)
    analytic = analytic.copy()
    analytic[analytic <= 0] = np.inf
    assert np.allclose(of.psd[1:], analytic[1:])


def test_default_optimal_filter_prefers_artifact_when_present(tmp_path) -> None:
    components = load_all_components()
    freq, native_psd = native_grid_psd(components)
    out = tmp_path / "artifact.npy"
    build_psd_artifact(
        freq,
        native_psd,
        sampling_frequency=A.DEFAULT_SAMPLING_FREQUENCY,
        n_samples=A.DEFAULT_SAMPLES,
        out_path=out,
    )
    of = A.build_optimal_filter(psd_file=out)
    _, analytic = A.noise_generator().build_psd_density(A.DEFAULT_SAMPLES)
    ratio = of.psd[1:] / analytic[1:]
    # in-band interpolation of the raw budget vs the parametric fit: agree to
    # interpolation precision everywhere in band
    assert np.max(np.abs(ratio - 1.0)) < 2e-2


def test_explicit_missing_psd_file_is_an_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        A.build_optimal_filter(n_samples=4096, psd_file=tmp_path / "nope.npy")


# --------------------------------------------------------------------------
# 2. alias folding is opt-in and its effect is what the docstring says
# --------------------------------------------------------------------------


def test_phase2_default_is_band_limited_and_fold_is_opt_in(tmp_path) -> None:
    components = load_all_components()
    freq, native_psd = native_grid_psd(components)
    fs, n = 1.0e6, 8192
    meta_default = build_psd_artifact(
        freq, native_psd, sampling_frequency=fs, n_samples=n, out_path=tmp_path / "a.npy"
    )
    meta_fold = build_psd_artifact(
        freq, native_psd, sampling_frequency=fs, n_samples=n, out_path=tmp_path / "b.npy",
        alias_fold=True,
    )
    assert meta_default["method"] == "inband_interpolation"
    assert meta_fold["method"] == "alias_fold"
    inband = np.load(tmp_path / "a.npy")[1]
    folded = np.load(tmp_path / "b.npy")[1]
    f = np.load(tmp_path / "a.npy")[0]
    hi = (f > 1.0e4) & (f < f[-1])  # interior bins; Nyquist is a one-sided boundary bin
    white_floor = components["Johnson"][1][-1] ** 2 + components["SQUID"][1][-1] ** 2
    assert np.allclose(inband[hi], white_floor, rtol=5e-2)
    # ~2*ceil(30 MHz / 1 MHz) aliases of a white floor land in band
    assert np.all(folded[hi] / inband[hi] > 40.0)
    # a band-limited fold reproduces the in-band answer
    meta_bl = build_psd_artifact(
        freq, native_psd, sampling_frequency=fs, n_samples=n, out_path=tmp_path / "c.npy",
        alias_fold=True, anti_alias_cutoff_hz=fs / 2,
    )
    band_limited = np.load(tmp_path / "c.npy")[1]
    assert np.allclose(band_limited[1:-1], inband[1:-1], rtol=1e-6)
    assert "band-limited" in meta_bl["readout_bandwidth_assumption"]


# --------------------------------------------------------------------------
# 3. noise_power is never a silent no-op
# --------------------------------------------------------------------------


def _absolute_composite(fs: float = 1000.0, psd_scale: float = 1.0) -> dict:
    return {
        "noise_type": White(scale=0.5) + PowerLaw(scale=2.0, exponent=-1.0),
        "noise_power": 1.0,
        "sampling_frequency": fs,
        "composite_psd_scaling": "absolute",
        "psd_scale": psd_scale,
    }


def test_absolute_composite_rejects_non_unity_noise_power() -> None:
    cfg = _absolute_composite()
    cfg["noise_power"] = 2.0
    with pytest.raises(ValueError, match="noise_power has no effect"):
        NoiseGenerator(cfg)


def test_psd_scale_multiplies_absolute_composite_density() -> None:
    base = NoiseGenerator(_absolute_composite(), seed=0)
    scaled = NoiseGenerator(_absolute_composite(psd_scale=0.25), seed=0)
    _, a, meta_a = base.build_psd_density(1024, return_metadata=True)
    _, b, meta_b = scaled.build_psd_density(1024, return_metadata=True)
    assert np.allclose(b, 0.25 * a)
    assert meta_a["level_knob"] == "psd_scale"
    assert np.isclose(meta_b["expected_variance"], 0.25 * meta_a["expected_variance"])


def test_level_knob_property() -> None:
    assert NoiseConfig(noise_type="white", noise_power=1.0, sampling_frequency=10.0).level_knob == "noise_power"
    assert NoiseConfig.from_mapping(_absolute_composite()).level_knob == "psd_scale"
    cfg = _absolute_composite()
    cfg["composite_psd_scaling"] = "normalize"
    assert NoiseConfig.from_mapping(cfg).level_knob == "noise_power"


def test_piecewise_level_variation_is_applied_to_absolute_composite() -> None:
    fs, n = 1000.0, 8192
    base = NoiseGenerator(_absolute_composite(fs), seed=1)
    wrapper = TemporalNoiseWrapper(
        {
            "mode": "piecewise",
            "n_segments": 2,
            "crossfade_len": 0,
            "noise_power_scale_range": [0.1, 0.1],
            "boundary_policy": "hard",
        },
        seed=2,
    )
    y, meta = wrapper.generate_piecewise(n, base, return_metadata=True)
    ref = np.mean([base.generate_noise(n).var() for _ in range(20)])
    assert all(np.isclose(s["level_scale"], 0.1) for s in meta["segments"])
    assert all(s["level_knob"] == "psd_scale" for s in meta["segments"])
    assert all(np.isclose(s["psd_scale"], 0.1) for s in meta["segments"])
    # power ratio 0.1 must show up in the data, not only in the metadata
    assert 0.05 < y.var() / ref < 0.2


def test_piecewise_level_variation_still_uses_noise_power_for_analytic() -> None:
    fs, n = 1000.0, 4096
    base = NoiseGenerator({"noise_type": "white", "noise_power": 4.0, "sampling_frequency": fs}, seed=1)
    wrapper = TemporalNoiseWrapper(
        {"mode": "piecewise", "n_segments": 2, "crossfade_len": 0, "noise_power_scale_range": [0.5, 0.5]},
        seed=2,
    )
    y, meta = wrapper.generate_piecewise(n, base, return_metadata=True)
    assert all(s["level_knob"] == "noise_power" for s in meta["segments"])
    assert all(np.isclose(s["noise_power"], 2.0) for s in meta["segments"])
    assert 1.6 < y.var() < 2.4


def test_al2o3_preset_piecewise_metadata_is_truthful() -> None:
    gen = A.noise_generator(sampling_frequency=1.0e6, seed=3)
    wrapper = TemporalNoiseWrapper(
        {"mode": "piecewise", "n_segments": 3, "crossfade_len": 0, "noise_power_scale_range": [0.3, 0.3]},
        seed=4,
    )
    y, meta = wrapper.generate_piecewise(16384, gen, return_metadata=True)
    ref = gen.generate_noise(16384).var()
    assert all(s["level_knob"] == "psd_scale" for s in meta["segments"])
    assert 0.2 < y.var() / ref < 0.45


# --------------------------------------------------------------------------
# 4. validation whitening must ignore zero-power bins (found while building
#    the demo notebook: the DC bin round-off, divided by a 1e-30 floor, made
#    the whitened lag-1 gate fail for the absolute-scale Al2O3 preset)
# --------------------------------------------------------------------------


def test_validation_whitened_lag1_is_finite_for_absolute_scale_preset() -> None:
    from noise_module import ValidationConfig, validate_stationary_gaussian

    result = validate_stationary_gaussian(
        A.noise_generator(sampling_frequency=1.0e6, seed=7),
        4096,
        ValidationConfig(ensemble_size=64),
    )
    lag1 = result.statistics["whitened_lag1_autocorrelation"]
    assert abs(lag1) < 0.05
    assert result.thresholds["gates"]["whitened_lag1"]
    assert result.passed


# --------------------------------------------------------------------------
# 5. optimal-filter resolution in physical units (added for the demo)
# --------------------------------------------------------------------------


def test_optimal_filter_resolution_matches_empirical_scatter() -> None:
    n, fs = 16384, 1.0e6
    of = A.build_optimal_filter(n_samples=n, sampling_frequency=fs)
    gen = A.noise_generator(sampling_frequency=fs, seed=5)
    amp = 50.0 * of.amplitude_resolution
    traces = gen.generate_ensemble(400, n) + amp * of.template[None, :]
    est = np.array([of.estimate_amplitude(x) for x in traces])
    assert abs(np.mean(est) - amp) < 5.0 * of.amplitude_resolution / np.sqrt(400)
    # empirical sigma of a 400-draw sample has ~3.5% relative uncertainty
    assert 0.85 < np.std(est, ddof=1) / of.amplitude_resolution < 1.15
    assert of.amplitude_resolution != 1.0 / np.sqrt(of.normalization)
