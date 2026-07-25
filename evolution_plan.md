# Noise Module Evolution Plan

## Purpose

This document is the implementation and validation roadmap for evolving
`noise_module` into a reproducible, statistically defensible simulator for
stationary, nonstationary, non-Gaussian, artifact-contaminated, and
multichannel noise.

The architecture remains compositional:

```text
stationary Gaussian core
    -> temporal/nonstationary transformation
    -> artifact injection
    -> optional multichannel correlation model
```

Each layer must have an explicit statistical contract, stable units,
reproducible random-number handling, metadata sufficient to reconstruct a
realization, and ensemble-level tests.

## Definitions and conventions

- A trace has `N` samples, sample frequency `fs`, duration `T = N / fs`, and
  one-sided rFFT frequencies `f[k] = k fs / N`.
- PSD density has units `signal_unit^2 / Hz`.
- CSD density has units `signal_unit_i * signal_unit_j / Hz`.
- Unless `power_definition="mean_square"` is selected, `noise_power` means
  expected variance and the DC PSD bin is zero.
- For NumPy's unnormalized FFT convention, expected rFFT coefficient power is
  `S[k] fs N` at DC and Nyquist and `S[k] fs N / 2` at interior bins.
- Event rates are in events per second. For a homogeneous Poisson process,
  `K ~ Poisson(rate_hz * N / fs)`.
- All public random operations accept either `seed` or `rng`, never both.
- A target CSD matrix must be Hermitian positive semidefinite at every
  frequency. DC and Nyquist CSD matrices must be real.

## Current implementation baseline

- [x] Separate physical PSD density from synthesis-domain rFFT power.
- [x] Draw complex Gaussian interior Fourier coefficients.
- [x] Handle real DC and Nyquist coefficients.
- [x] Use local NumPy generators and spawned child generators.
- [x] Support analytic white, pink, brownian, blue, and violet spectra.
- [x] Support custom PSD files.
- [x] Compose temporal, artifact, and multichannel components.
- [x] Preserve an artifact-disabled multichannel input exactly.
- [x] Define custom PSD scaling modes.
- [x] Normalize transient rates by trace duration.
- [x] Implement continuous piecewise PSD-slope variation.
- [x] Add target-CSD multichannel synthesis.
- [x] Add initial ensemble PSD, power, Gaussianity, event-rate, and covariance
  tests.

## Phase 1 — Harden configuration and public contracts

### 1.1 Typed configuration models

- [x] Introduce typed configuration objects for the stationary, temporal,
  artifact, and multichannel layers.
- [x] Keep dictionary input as a compatibility adapter.
- [x] Reject unknown keys in strict mode.
- [x] Validate all scalars as finite.
- [x] Validate `fs > 0`, `N > 0`, powers and rates as non-negative, probabilities
  in `[0, 1]`, ordered ranges, positive durations, and valid channel counts.
- [x] Attach units and parameter meanings to every field's documentation.
- [x] Version serialized configuration and metadata schemas.

Implementation guidance:

1. Define one model per module instead of a single global configuration.
2. Perform validation at construction time, before consuming RNG state.
3. Store the fully resolved configuration, including defaults, in metadata.
4. Provide a migration function for legacy configurations.

Acceptance criteria:

- Invalid configurations fail before generating partial output.
- A configuration serialized to JSON and loaded again produces identical
  output for the same seed and software version.
- Compatibility tests cover all currently used repository configurations.

### 1.2 Power semantics

- [x] Default analytic generation to expected variance with zero DC.
- [x] Retain an explicit `mean_square` mode for stochastic DC.
- [x] Validate non-negative finite `noise_power`.
- [x] Rename or deprecate ambiguous `build_psd()` in favor of
  `build_psd_density()` and `build_rfft_power()`.
- [x] Add metadata fields for expected variance, expected mean-square power,
  DC power, and the discrete integration rule.
- [x] Add an optional deterministic mean parameter separate from stochastic
  DC noise.
- [x] Define behavior for `N=1` and document that a zero-DC variance process
  has no supported random degree of freedom in that case.

Acceptance criteria:

- Across at least 1,000 independent realizations, ensemble mean variance agrees
  with requested variance within a tolerance derived from Monte Carlo standard
  error.
- Parseval equality holds to floating-point tolerance for every generated
  realization.
- Odd and even `N` tests cover endpoint handling.

### 1.3 Custom PSD scaling and interpolation

- [x] Implement `absolute`: file values remain physical PSD densities.
- [x] Implement `normalize`: file values are treated as a shape normalized to
  `noise_power`.
- [x] Implement `multiply`: physical file density is multiplied by
  `noise_power`.
- [x] Validate custom PSD shape, monotonic frequencies, finiteness, and
  non-negative density.
- [x] Rename `multiply` to `scale`, with a dedicated
  `psd_scale` field, while retaining a compatibility alias.
- [x] Support explicit out-of-band policies: `error`, `zero`, `edge`, and
  model-based extrapolation.
- [x] Add log-frequency/log-density interpolation for spectra spanning several
  decades, with defined handling of zero-valued bins.
- [x] Store source frequency range, interpolation policy, scale mode, and
  resulting integral in metadata.

Acceptance criteria:

- `normalize` integrates to `noise_power` on every target rFFT grid.
- `absolute` reproduces the in-band source density at source grid points.
- Resampling tests cover target grids both inside and outside the source band.

## Phase 2 — Strengthen stationary spectral models

### 2.1 General power-law and composite PSDs

- [x] Support a continuous `psd_exponent` where `S(f) proportional to f^alpha`.
- [x] Add configurable low-frequency and high-frequency cutoffs to prevent
  divergent or physically implausible power-law behavior.
- [x] Add a compositional PSD API:

  ```python
  total_psd = White(level=...) + PowerLaw(alpha=...) + Lorentzian(...) + Line(...)
  ```

- [x] Add Lorentzian, band-limited, resonance, and roll-off components.
- [x] Permit parameters to be expressed in physical units or normalized to an
  integrated power budget.
- [x] Record each component's integrated contribution in metadata.

Acceptance criteria:

- Analytic component integrals agree with discrete-grid integrals where an
  analytic result exists.
- Ensemble periodograms recover both total PSD and component-dominant regions.

### 2.2 Alternative exact time-domain simulation

- [x] Evaluate circulant embedding for users requiring a finite vector with an
  explicitly specified Toeplitz covariance.
- [x] Clearly label FFT spectral synthesis as periodic/circulant on the finite
  grid.
- [x] Provide covariance error diagnostics comparing requested and implied
  finite-sample covariance.

Use this only when exact finite-vector covariance is important; retain spectral
synthesis as the default for PSD-driven simulation.

## Phase 3 — Nonstationary and temporal models

### 3.1 Separate generation from transformation

- [x] Replace ambiguous `apply(x, base_generator=...)` behavior with:

  ```python
  generate_piecewise(N, base_generator)
  modulate_variance(x)
  add_drift(x)
  ```

- [x] Deprecate any method that silently discards its input.
- [x] Allow a pipeline to return each additive component and multiplicative
  envelope separately.

Acceptance criteria:

- Every `apply`-style method is identity when all relevant effects are disabled.
- Metadata identifies whether data was regenerated, multiplied, or added.

### 3.2 Piecewise and smoothly evolving spectra

- [x] Implement piecewise continuous power-law exponent variation.
- [x] Support arbitrary local PSD parameter changes, not only power and slope.
- [x] Offer boundary policies: hard transition, overlap-add, and continuous
  parameter interpolation.
- [x] Quantify crossfade effects on local power and PSD.
- [x] Add smoothly time-varying spectral synthesis using windowed overlap-add
  or an evolutionary-spectrum model.
- [x] Make crossfade window and overlap explicit and enforce constant overlap-add
  where applicable.

Acceptance criteria:

- Windowed local PSD estimates recover configured segment parameters within
  estimator uncertainty away from boundaries.
- Boundary tests quantify rather than hide transition bias.
- Output length and segment indexing are exact for all `N`, segment sizes, and
  overlaps.

### 3.3 Variance envelopes and drift

- [x] Generate positive envelopes in log amplitude rather than clipping cubic
  splines after overshoot.
- [x] Add PCHIP and low-pass stochastic-envelope options.
- [x] Parameterize envelope strength as local variance ratio.
- [x] Parameterize drift by RMS and characteristic timescale or cutoff
  frequency.
- [x] Distinguish integrated random walk, stationary low-frequency noise, and
  deterministic baseline drift.
- [x] Support shared, partially shared, and private multichannel envelopes and
  drifts.

Acceptance criteria:

- The realized envelope remains positive and within documented bounds.
- Local variance tracks the requested envelope after accounting for estimator
  uncertainty.
- Drift PSD is concentrated in its configured low-frequency region.

## Phase 4 — Artifact and non-Gaussian models

### 4.1 Artifact identity and amplitude semantics

- [x] Ensure disabled multichannel artifact injection is an exact identity.
- [x] Apply channel jitter only to the injected artifact, not the baseline.
- [x] Specify artifact amplitude in one of: raw units, baseline RMS, local RMS,
  or matched-filter SNR.
- [x] Reject line frequencies outside `[0, fs/2]` unless an explicit aliasing
  mode is selected.
- [x] Preserve phase and all sampled template parameters in metadata.
- [x] Return optional artifact-only arrays and boolean/sample masks.

Acceptance criteria:

- With all artifacts disabled, output is bitwise equal to input.
- Artifact-only output plus baseline reconstructs the combined output exactly.
- Requested peak amplitude or SNR is recovered by a template-specific test.

### 4.2 Time-normalized event processes

- [x] Interpret glitch and burst rates as events per second.
- [x] Draw homogeneous counts using `Poisson(rate_hz * N / fs)`.
- [x] Report rate and expected count in metadata.
- [x] Sample event times as a point process directly, then handle boundary
  truncation explicitly.
- [x] Add nonhomogeneous Poisson rates `lambda(t)`.
- [x] Add clustered/Hawkes-like events only when supported by empirical data.
- [x] Define overlap policy: allow, merge, reject/resample, or superpose.
- [x] Express duration distributions in seconds, with sample-based fields kept
  only as compatibility aliases.

Acceptance criteria:

- Ensemble mean and variance of counts equal `rate_hz * duration` within
  Monte Carlo confidence intervals.
- Concatenating two traces produces the same expected event count as generating
  one trace of their combined duration.
- Boundary and overlap behavior is deterministic given the RNG stream.

### 4.3 First-class non-Gaussian baselines

- [x] Add Student-t innovations with degrees of freedom greater than two when
  finite variance is required.
- [x] Add Laplace and Gaussian-scale-mixture models.
- [x] Add compound-Poisson shot noise.
- [x] Consider alpha-stable noise only with explicit warnings that variance and
  PSD power may be undefined.
- [x] Separate marginal non-Gaussianity from temporal dependence: filtering
  non-Gaussian innovations generally does not preserve a simple target
  marginal.
- [x] Provide skewness, kurtosis, tail-exceedance, and QQ diagnostics.

Acceptance criteria:

- Tests verify theoretical moments only where those moments exist.
- PSD and marginal-distribution validation are reported separately.
- Metadata states whether `noise_power` is a variance, scale parameter, or is
  undefined for the selected family.

## Phase 5 — Multichannel correlation

### 5.1 Convenience correlation models

- [x] Support independent channels.
- [x] Support shared-plus-private channels.
- [x] Support low-rank latent channels.
- [x] Report the implied covariance/correlation matrix, not only mean
  off-diagonal correlation.
- [x] Distinguish requested mixing strength from resulting correlation when
  gains and private strengths vary.
- [x] Avoid per-realization normalization by default; when selected, document
  that it changes the sampling distribution.

Acceptance criteria:

- Closed-form implied covariance agrees with ensemble covariance.
- Metadata exposes per-channel variance and pairwise correlation.

### 5.2 Target-CSD synthesis

- [x] Accept target one-sided CSD density with shape `(F, C, C)`.
- [x] Validate frequency count, finiteness, Hermitian symmetry, and positive
  semidefiniteness.
- [x] Enforce real endpoint matrices at DC and Nyquist.
- [x] Factor each matrix by eigendecomposition and clip only numerically small
  negative eigenvalues.
- [x] Generate complex Gaussian interior vectors and real endpoint vectors.
- [x] Convert physical CSD density to NumPy rFFT covariance with correct
  endpoint factors.
- [x] Accept frequency grids and interpolate CSDs while preserving positive
  semidefiniteness.
- [x] Add batched factorization and realization generation for performance.
- [x] Add a low-rank factor input that avoids constructing dense `C x C`
  matrices.
- [x] Add configurable policies for nearly indefinite empirical CSDs:
  `error`, eigenvalue clipping, diagonal loading, or nearest-PSD projection.
- [x] Report the magnitude and location of any regularization.
- [x] Support frequency-dependent phase/coherence validation.

Implementation guidance for interpolation:

1. Do not interpolate matrix entries independently unless the result is
   projected back to the PSD cone.
2. Prefer interpolating a spectral factor, a matrix logarithm for positive
   definite matrices, or physically meaningful latent factors.
3. Preserve conjugate symmetry and real endpoint constraints.

Acceptance criteria:

- Ensemble cross-periodograms recover diagonal PSD, cross-spectrum magnitude,
  phase, and coherence within confidence bounds.
- Ensemble time-domain covariance equals the integral of the target CSD.
- Singular low-rank target matrices are accepted.
- Materially indefinite matrices are rejected unless an explicit repair policy
  is selected.

## Phase 6 — Statistical validation framework

### 6.1 Ensemble test design

- [x] Add initial ensemble variance validation.
- [x] Add initial ensemble periodogram validation.
- [x] Add skewness and excess-kurtosis checks for Gaussian marginals.
- [x] Add Poisson rate validation.
- [x] Add target-CSD covariance recovery.
- [x] Centralize ensemble size, confidence level, seed set, and tolerances.
- [x] Base pass/fail thresholds on sampling distributions or bootstrap
  intervals instead of arbitrary constants.
- [x] Mark long statistical tests separately from fast deterministic unit
  tests.
- [x] Record diagnostic statistics even when assertions pass.

Recommended layers:

```text
unit tests          exact shapes, errors, units, identities, reproducibility
statistical tests   ensemble properties with confidence-aware thresholds
integration tests   full composed pipelines and metadata
calibration tests   comparison to real reference datasets
```

### 6.2 Stationary Gaussian validation matrix

- [x] Test every analytic color for odd and even `N`.
- [x] Test DC and Nyquist coefficient distributions.
- [x] Test interior real/imaginary Gaussianity and independence.
- [x] Test exponential normalized periodogram ordinates.
- [x] Test whitened autocorrelation using confidence bands.
- [x] Test PSD recovery using averaged periodograms/Welch estimates.
- [x] Test Parseval consistency realization by realization.
- [x] Test seed reproducibility and child-stream independence.

Avoid relying exclusively on normality-test p-values: with small samples they
have low power, and with very large samples they reject negligible deviations.
Report effect sizes such as skewness, excess kurtosis, covariance error, and
quantile error alongside calibrated hypothesis tests.

### 6.3 Nonstationary validation matrix

- [x] Measure local mean, variance, PSD slope, and band power using windows
  shorter than the evolution timescale.
- [x] Test known transition locations.
- [x] Test time-frequency recovery of smoothly varying spectra.
- [x] Quantify crossfade bias separately from segment interiors.
- [x] Validate shared versus private multichannel temporal components.

### 6.4 Artifact validation matrix

- [x] Validate event count and inter-arrival distributions.
- [x] Validate duration and amplitude distributions.
- [x] Validate line-bin frequency, amplitude, phase, and harmonic content.
- [x] Validate injected masks and metadata against exact samples.
- [x] Validate overlap and boundary policies.
- [x] Validate non-Gaussian tail statistics of the composed output.

### 6.5 CSD validation matrix

- [x] Compare ensemble CSD to target at each frequency or frequency band.
- [x] Compare target and observed coherence.
- [x] Compare cross-spectrum phase.
- [x] Compare integrated CSD with time-domain covariance.
- [x] Cover diagonal, dense, complex-phase, singular low-rank, and nearly
  indefinite inputs.

## Phase 7 — Empirical calibration and reproducible presets

- [x] Define a standard input format for clean and contaminated reference data.
- [x] Estimate PSD/CSD using documented windows, overlap, detrending, and
  uncertainty.
- [x] Estimate segment power, drift timescales, line distributions, artifact
  rates, duration/amplitude distributions, and tail statistics.
- [x] Split calibration and validation records to avoid tuning and evaluating
  on the same data.
- [x] Store presets with dataset identifier, channel selection, acquisition
  conditions, preprocessing, units, estimator settings, code revision, and
  calibration date.
- [x] Add posterior or bootstrap parameter uncertainty where feasible.
- [x] Permit simulator parameters to be sampled from calibration uncertainty,
  rather than using only point estimates.

Acceptance criteria:

- A preset can be regenerated from source data and documented commands.
- Held-out real data falls within predeclared simulator diagnostic intervals
  for the statistics the preset claims to model.
- Presets state explicitly which properties are not modeled.

## Phase 8 — Performance, packaging, and API stability

- [x] Fix repository/package layout so the full test suite imports consistently
  without test-only aliases.
- [x] Declare runtime and test dependencies in package metadata.
- [x] Vectorize ensemble and multichannel generation.
- [x] Benchmark memory and runtime over representative `(realizations, C, N)`.
- [x] Add chunked/streaming generation with deterministic RNG behavior.
- [x] Ensure chunked and one-shot generation contracts are documented; exact
  sample equality is required only if deliberately supported.
- [x] Add semantic versioning and deprecation warnings.
- [x] Provide end-to-end examples for every supported statistical contract.

Acceptance criteria:

- A clean environment can install the package and run the complete test suite.
- Public examples execute in continuous integration.
- Performance regressions above an agreed threshold are detected.

## Immediate next implementation checklist

These are the recommended next tasks, in order:

- [x] Fix the two full-suite collection failures involving
  `noise_module.noise_module` and `QPSimulator`.
- [x] Add odd-`N`, DC, Nyquist, zero-power, and `N=1` tests.
- [x] Add explicit frequency-grid input and PSD-preserving interpolation to
  target-CSD synthesis.
- [x] Add complex cross-spectrum phase/coherence ensemble tests.
- [x] Refactor temporal generation so no `apply` method discards input.
- [x] Change artifact durations to seconds and add exact event masks.
- [x] Add confidence-aware shared statistical-test utilities.
- [ ] Update tutorials and the older modular design specification to use the
  new power, custom-PSD, and events-per-second semantics.

## Definition of done for a paper-ready release

- [ ] Every public parameter has an unambiguous definition and unit.
- [ ] Every generator exposes its expected first- and second-order statistics
  where those quantities exist.
- [ ] Ensemble tests validate every claimed statistical property.
- [ ] Nonstationary behavior is validated locally in time, not only globally.
- [ ] Multichannel behavior is validated through PSD, CSD, coherence, phase,
  and covariance.
- [ ] Non-Gaussian behavior is validated through distributional and tail
  diagnostics independently of PSD.
- [ ] All presets have provenance and held-out-data validation.
- [ ] The full suite installs and runs in a clean environment.
- [ ] Metadata plus versioned configuration is sufficient to reproduce each
  generated dataset.

## Post-review corrective implementation — schema/package 0.3.0

The following checklist records the 19 findings from the scientific and
engineering review. Each correction has a focused regression test in
`noise_module/noise_module/tests/test_review_fixes.py`.

### Scientific correctness and synthesis

- [x] **1. Multichannel artifact scaling:** generate each channel's artifact
  against its real baseline, then apply a positive unit-mean log-normal
  channel gain. Baseline-RMS, local-RMS, and energy-ratio units no longer
  collapse to the numerical RMS floor.
- [x] **2. Temporal component identity:** include the piecewise crossfaded
  amplitude scale in the returned multiplicative envelope. Combined
  piecewise scaling, variance modulation, and drift now satisfy
  `output = input * envelope + drift`.
- [x] **3. CSD input immutability:** copy target CSD input before Hermitian
  symmetrization or PSD repair, including complex arrays that `asarray` would
  otherwise alias.
- [x] **4. Genuine factor-direct CSD synthesis:** synthesize from `(F,C,R)`
  factors using rank-`R` Gaussian innovations. Do not allocate `(F,C,C)` or
  call the dense eigensynthesis path.
- [x] **5. Evolutionary OLA boundaries:** add pre-roll/post-roll window
  coverage, normalize every output sample by accumulated squared-window
  weight, and reject uncovered hop/window combinations.
- [x] **6. Event-process rates:** use Poisson interval counts for
  nonhomogeneous intensities, permit multiple events per interval, and report
  immigrant versus branching-process total expectations for Hawkes clusters.
- [x] **7. Calibration event counts:** count only rising threshold crossings,
  not both entry and exit edges of an exceedance run.
- [x] **8. Artifact SNR terminology:** migrate the old `snr` name to
  `rms_energy_ratio` and state its exact norm/RMS definition. Do not present it
  as colored-noise matched-filter SNR.
- [x] **9. Single-record coherence:** use Welch segment averaging for one
  record; retain exact-grid ensemble periodogram averaging for `(R,C,N)`.
- [x] **10. Multichannel covariance metadata:** integrate the actual base PSD
  on the requested grid and transform the population covariance when
  per-channel normalization is enabled.

### Validation and calibration

- [x] **11. Validation gates:** expose independent confidence-aware gates for
  power, PSD, Fourier/time Gaussianity, periodogram distribution, whitening,
  artifact identity/counts, CSD magnitude, phase, coherence, and covariance.
  Diagnostic-only local validation returns `passed=None`.
- [x] **12. Held-out calibration:** require at least two records and gate
  held-out power coverage, PSD log error/interval coverage, CSD Frobenius
  error, and tail exceedance rather than accepting on power alone.

### Engineering, API, streaming, and packaging

- [x] **13. Strict configuration:** reject unknown fields by default; legacy
  permissive parsing requires an explicit `strict_config=False`.
- [x] **14. Integer validation:** reject fractional values for channel,
  latent, segment, crossfade, knot, and window-count fields instead of
  validating through truncating `int(...)`.
- [x] **15. Invalid spectral input:** reject negative and nonfinite rFFT
  powers rather than silently clipping them.
- [x] **16. Streaming semantics:** generate only fixed-grid underlying blocks
  and serve short requests from a buffer, with explicit cross-block and
  one-shot contracts.
- [x] **17. Artifact overlap API:** canonicalize `allow` to `superpose`,
  retain implemented `reject`/`resample`, and reject undefined `merge`
  semantics.
- [x] **18. Calibration axes:** require an explicit interpretation for 2-D
  reference arrays so `(records,samples)` and `(channels,samples)` cannot be
  silently confused.
- [x] **19. Packaging and metadata:** remove the test-only `wk7` alias, derive
  the wheel version from `noise_module.__version__`, provide recursive JSON
  metadata conversion, and validate an isolated wheel import.

Verification baseline:

- `92 passed` for the standalone noise-module suite.
- The `modular-noise-simulator-0.3.0` wheel builds and imports in an isolated
  target directory.
- Repository-wide tests reach `209 passed, 6 skipped`; two unrelated GWOSC
  diagnostics require the optional `gwpy` dependency.
