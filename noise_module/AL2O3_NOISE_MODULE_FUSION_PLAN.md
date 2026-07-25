# Al2O3/Al and noise-module fusion plan

## Objective

Use the fitted Al2O3/Al noise budget and pulse as a calibrated detector preset
inside the general noise module, without exposing every low-level parameter in
normal experiments.

The integration should answer two different questions:

1. What does the calibrated reference detector predict?
2. How does reconstruction behave when one controlled detector or acquisition
   property changes?

It should not turn the reference fit, generic noise generator, temporal model,
artifact injector, and multichannel model into one flat configuration.

## Design principle

Use composition with explicit ownership:

```text
Al2O3 reference preset
    stationary physical PSD + physical pulse + acquisition grid
             |
             v
small detector adjustments
    source gains, selected corners/exponents, pulse variation
             |
             v
optional acquisition stressors
    drift, lines, glitches, non-Gaussianity, nonstationarity
             |
             v
optional multichannel coupling
             |
             v
trace + truth-separated components + resolved metadata
```

The calibrated preset remains immutable. Every simulation run creates a
resolved copy with a recorded list of adjustments.

## Parameter ownership

### Layer 1: reference detector

Owned by the Al2O3 preset and normally frozen:

- Johnson PSD level;
- SQUID white level and 1/f coefficient;
- TD plateau and corner;
- Er amplitude and exponent;
- pulse rise and decay constants;
- sampling frequency;
- reference record length and pretrigger.

These values come from the fitted reference files. Changing them means
simulating a detector different from the reference.

### Layer 2: detector adjustments

Expose a small set of dimensionless, interpretable knobs:

```yaml
detector:
  johnson_scale: 1.0
  squid_scale: 1.0
  td_scale: 1.0
  er_scale: 1.0
  td_corner_scale: 1.0
  pulse_rise_scale: 1.0
  pulse_decay_scale: 1.0
```

Default all scales to one. Prefer scaling a complete physical source over
editing its internal coefficients independently.

Only add a new public knob when it corresponds to a detector-design or
operating-condition hypothesis.

### Layer 3: acquisition stressors

Owned by the general noise module:

- baseline drift;
- variance or PSD evolution;
- line pickup and ringing;
- glitches, impulses, bursts, and pile-up;
- non-Gaussian innovations;
- multichannel shared/private structure.

These effects must remain separate from the stationary physical budget. They
should not silently modify the fitted Johnson, SQUID, TD, or Er parameters.

### Layer 4: expert overrides

Allow the existing low-level noise-module configuration only through an
explicit `expert_overrides` section. Runs using it should be marked
`reference_comparable: false` unless the override is known to preserve the
reference contract.

## Public configuration

Use a compact experiment-facing schema:

```yaml
preset: al2o3_al_athermal_v1
seed: 1234

acquisition:
  sampling_frequency_hz: 1000000
  n_samples: 131072
  pretrigger_fraction: 0.10

detector:
  johnson_scale: 1.0
  squid_scale: 1.0
  td_scale: 1.0
  er_scale: 1.0
  td_corner_scale: 1.0
  pulse_rise_scale: 1.0
  pulse_decay_scale: 1.0

stressors:
  profile: none

multichannel:
  profile: single_channel
```

Named stressor profiles should cover common studies:

- `none`;
- `mild_drift`;
- `line_pickup`;
- `sparse_glitches`;
- `nonstationary_noise`;
- `heavy_tailed`;
- `shared_readout`.

Profiles are versioned configurations, not hidden random choices. A profile
may expose one severity scalar, but its resolved parameters must be saved.

## Control mechanisms

### 1. Adjustment budget

Each experiment declares which knobs may vary. Reject changes to undeclared
knobs.

```yaml
sweep:
  allowed_parameters: [detector.squid_scale]
  detector.squid_scale: [0.5, 1.0, 2.0]
```

This prevents accidental multi-parameter sweeps and makes causal comparisons
clear.

### 2. One-factor and structured sweeps

Begin with one-factor-at-a-time source scaling. Use factorial or joint sweeps
only for physically motivated interactions. Do not launch an unconstrained
search over all simulator parameters.

### 3. Parameter bounds

Provide conservative default ranges, for example:

- source scales: 0.25 to 4;
- corner and pulse-time scales: 0.5 to 2;
- stressor severity: 0 to 1.

Going outside them requires expert mode and a justification string.

### 4. Truth-separated output

Every generated record should optionally return:

```text
stationary_total
johnson
squid
td
er
drift
lines
glitches
pileup
pulse
combined_trace
```

The components must reconstruct the combined trace exactly. This is essential
for source-recovery experiments and prevents ambiguous post-hoc labeling.

### 5. Resolved manifest

Save, for every run:

- preset name and version;
- fitted reference checksum;
- requested compact configuration;
- fully resolved low-level configuration;
- RNG seed/child seeds;
- component power before and after adjustment;
- record grid;
- enabled stressors;
- whether the run remains reference-comparable.

## Integration API

Add one facade rather than extending every existing class:

```python
simulator = Al2O3Simulation.from_preset(
    "al2o3_al_athermal_v1",
    detector={"squid_scale": 2.0},
    stressor_profile="mild_drift",
    seed=1234,
)

batch, truth, metadata = simulator.generate_batch(
    n_records=256,
    return_components=True,
)
```

Internally the facade composes the existing classes:

1. build the adjusted `CompositeSpectrum`;
2. generate stationary source components using independent child RNGs;
3. create the physical pulse;
4. apply temporal transformations;
5. add artifacts;
6. apply multichannel coupling;
7. verify component reconstruction and emit metadata.

Do not duplicate spectral synthesis, artifact injection, or multichannel
logic inside the facade.

## Validation gates

### Reference gate

With every scale at one and all stressors disabled:

- PSD matches the fitted Al2O3 composite;
- pulse matches `signal.dat`;
- empirical artifact and analytic spectrum agree on the selected grid;
- generated ensemble Welch PSD agrees within Monte Carlo uncertainty.

### Adjustment gate

For each source scale:

- only the requested source power changes;
- PSD ratios match the analytical expectation;
- source components remain independent unless coupling is requested;
- setting a source scale to zero removes only that source.

### Composition gate

- truth components sum exactly to the output;
- disabling a layer is an exact identity;
- fixed configuration and seed reproduce bitwise-identical output;
- changing one layer does not alter unrelated child RNG streams.

### Stressor gate

- event counts, amplitudes, drift RMS, and line frequencies recover their
  requested values statistically;
- stressor masks and labels match injected truth;
- clean-reference metrics are reported beside contaminated metrics.

### Inference gate

Before using the fusion for scientific claims:

- recover source scales from blinded simulated mixtures;
- quantify bias, coverage, degeneracy, and failure cases;
- demonstrate that EMPCA component labels agree with injected truth only
  when the proposed classification diagnostics pass;
- include negative controls with intentionally indistinguishable spectra.

## Implementation phases

### Phase A: preset and facade

- Define the versioned preset schema.
- Implement immutable reference loading.
- Implement the seven detector-scale knobs.
- Return source-separated stationary noise.
- Add resolved metadata and checksums.

Acceptance: the default facade reproduces the current analytical Al2O3
generator and pulse.

### Phase B: controlled stressors

- Add versioned named profiles.
- Preserve truth-separated temporal and artifact components.
- Implement adjustment budgets and bounds.

Acceptance: every profile passes component reconstruction and statistical
contract tests.

### Phase C: multichannel extension

- Define which physical sources are shared, partially shared, or private.
- Prefer source-specific mixing vectors/CSDs over one global correlation
  coefficient.
- Expose only named coupling scenarios initially.

Acceptance: empirical CSD and coherence recover the configured source-level
coupling.

### Phase D: experiment registry

- Store each sweep as a small declarative configuration.
- Validate allowed parameters before execution.
- Save manifests and aggregate source-level metrics.

Acceptance: every figure and table can be regenerated from a manifest without
manual parameter reconstruction.

## Recommended first experiment

Keep stressors disabled and sweep one source at a time:

```text
Johnson scale: 0.5, 1, 2
SQUID scale:   0.5, 1, 2
TD scale:      0.5, 1, 2
Er scale:      0.5, 1, 2
```

For each condition, measure:

- total and source-wise integrated power;
- optimal-filter amplitude variance;
- EMPCA subspace stability and rank;
- held-out reconstruction residual;
- recovery of the blinded source scale.

This establishes whether each physical source has a distinguishable impact
before adding drift, artifacts, non-Gaussianity, or multichannel complexity.

## Decision rule

The integration is controllable when:

1. normal users see fewer than roughly ten detector knobs;
2. every knob has a physical interpretation;
3. only declared knobs can vary in an experiment;
4. the full low-level configuration remains available but explicit;
5. generated sources and artifacts remain truth-separated;
6. reference behavior is protected by regression tests;
7. manifests make every run reproducible.

If an experiment needs many simultaneous expert overrides, it should be
treated as a new detector preset rather than another configuration of the
Al2O3 reference.
