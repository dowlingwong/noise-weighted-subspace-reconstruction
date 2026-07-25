# Changelog

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
