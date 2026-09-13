# Changelog

## 0.3.1 (2026-09-12, review fixes)

- `al2o3_athermal.build_optimal_filter()` no longer fails on the default
  1 MHz / 131072 grid when the generated Phase 2 artifact
  `al2o3_athermal_total_psd.npy` is absent: it falls back to the analytic
  Phase 1 composite with a `RuntimeWarning`. An explicitly passed missing
  `psd_file` is a `FileNotFoundError`. The artifact is regenerated and
  shipped (in-band, see next item).
- `fit_al2o3_athermal.build_psd_artifact` / CLI: alias folding of the 30 MHz
  budget is now **opt-in** (`--alias-fold`); the default is in-band
  interpolation, i.e. a band-limited readout/DAQ. Folding the white
  SQUID+Johnson floor with no bandwidth limit raised the in-band PSD ~60x and
  the optimal-filter baseline sigma ~5.7x. `--anti-alias-cutoff-hz` /
  `--anti-alias-transition-hz` express the readout bandwidth in either mode;
  the artifact metadata records `readout_bandwidth_assumption`.
- `noise_power` is never a silent no-op any more. `NoiseConfig` rejects a
  non-unity `noise_power` together with `composite_psd_scaling="absolute"`;
  `psd_scale` is the level multiplier for absolute composite spectra (applied
  in `build_psd_density`, reported in metadata as `psd_scale` /
  `level_knob`). `NoiseConfig.level_knob` names the field that carries the
  level for any configuration. `TemporalNoiseWrapper` scales the level
  through that knob (switching custom PSDs from `absolute` to `scale` mode
  when needed) and its segment/window metadata reports the applied
  `level_scale` and `level_knob`; previously the wrapper scaled
  `noise_power`, which absolute configurations ignored while the metadata
  reported the scaled value.
- `validate_stationary_gaussian` whitens only on bins with non-zero expected
  power. Dividing the round-off in the (zero-power) DC bin by a 1e-30 floor
  produced a spurious constant that dominated the whitened series and failed
  the `whitened_lag1` gate for spectra with a large absolute scale (e.g. the
  Al2O3 preset in µΦ₀²/Hz).
- `OptimalFilter.amplitude_resolution` (and `rfft_power`): the baseline
  1-sigma of `estimate_amplitude` in template-amplitude units, computed with
  the rFFT-bin power convention and double-counted interior bins.
  `OptimalFilter.normalization` was being read as a resolution; its docstring
  now says why it is not one.

## 0.3.0

- Corrected baseline/local-RMS artifact scaling for multichannel inputs and
  made temporal component decomposition exactly reconstruct piecewise output.
- Made target-CSD repair input-immutable and implemented direct low-rank
  factor synthesis without allocating dense CSD matrices.
- Added boundary-covered evolutionary overlap-add, inhomogeneous Poisson
  arrivals, Hawkes total-count expectations, and rising-edge calibration
  event counts.
- Replaced unconditional diagnostic passes with separate power, PSD,
  Gaussianity, CSD magnitude, phase, coherence, covariance, artifact, and
  local-nonstationarity gates.
- Enabled strict configuration validation by default, exact integer
  validation, fixed-grid buffered streaming, JSON metadata conversion, and
  explicit two-dimensional calibration layouts.
- Removed undefined artifact `merge` overlap behavior, canonicalized `allow`
  to `superpose`, and renamed legacy `snr` amplitude units to the scientifically
  explicit `rms_energy_ratio`.
- Consolidated package version metadata and removed the test-only
  `wk7.noise_module` import alias.

## 0.2.0

- Added typed and versioned configuration contracts.
- Added custom/composite PSD models and exact covariance simulation.
- Added nonstationary, artifact, non-Gaussian, and target-CSD models.
- Added confidence-aware validation and empirical calibration presets.
- Added vectorized ensembles, deterministic streaming, benchmarks, examples,
  deprecation warnings, and standalone packaging.

## 0.1.0

- Initial modular stationary noise generator.
