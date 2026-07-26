# Time-domain EMPCA: signal, noise, and artifact content

## Question

When EMPCA is trained on aligned detector traces in the time domain, do
higher-order principal components represent detector noise and time-localized
artifacts?

## Short answer

Not in general. Component order ranks weighted variance explained by the
training sample; it does not label a component as signal, noise, or artifact.

- Reproducible pulse-shape variation can occupy higher components.
- A coherent or high-amplitude artifact can occupy an early component.
- Correctly whitened stationary Gaussian noise produces an approximately
  isotropic eigenvalue bulk. Directions inside that bulk are sample-dependent
  and should not be interpreted as physical templates.
- Unwhitened colored noise can produce smooth or oscillatory PCs that look
  structured even though they contain no pulse physics.
- Sparse glitches violate the Gaussian residual model used by EMPCA and can
  rotate the entire fitted subspace rather than being confined to the last PCs.

Therefore a high-order PC represents noise or an artifact only after an
independent diagnostic establishes that interpretation.

## What EMPCA optimizes

For centered traces `x_i` and a rank-k basis `B`, the relevant model is

```text
x_i = B a_i + epsilon_i
epsilon_i ~ N(0, Sigma)
```

and the fitted subspace minimizes

```text
sum_i (x_i - B a_i)^T Sigma^-1 (x_i - B a_i).
```

The repository implements this as weighted PCA in
`src/noise_geometry/subspace/pca.py`. With colored stationary noise, a
time-domain implementation must use the full inverse covariance `Sigma^-1`,
or equivalently prewhiten each trace. A vector of time-bin weights describes
heteroskedastic but independent errors; it does not describe temporal
correlation.

The production frequency-domain implementation uses inverse-PSD weights. For
a stationary process, this is the computationally natural representation of
the same covariance metric. A direct time-domain comparison is valid only
when its whitening operator matches that PSD, including DC/Nyquist and finite
record conventions.

Bailey's weighted EM-PCA method was designed to prevent heteroskedastic
measurement errors and missing samples from pulling the recovered
eigenvectors away from the underlying variation. It does not make the model
robust to unmodeled sparse outliers or automatically attach physical meaning
to component number.

## Expected components for detector pulses

For a smooth pulse family `s(t; theta)`, a local expansion gives

```text
s(t; theta + delta_theta)
  ~= s(t; theta)
   + sum_j delta_theta_j * partial s(t; theta) / partial theta_j.
```

This gives a useful interpretation to low-rank pulse subspaces:

| Variation in training pulses | Expected direction |
|---|---|
| Amplitude | Mean pulse/template |
| Small arrival-time jitter | Time derivative `ds/dt` |
| Rise-time variation | `partial s / partial tau_rise` |
| Decay-time variation | `partial s / partial tau_decay` |
| Baseline offset or slope | Constant or ramp direction |
| Pile-up with similar delay | Shifted-pulse direction(s) |

The ordering among these directions depends on their population variances
after whitening. Timing jitter can therefore be PC2 in one dataset and PC4 in
another. PCA has been used successfully for microcalorimeter pulse-shape
variation and anomalous-pulse discrimination, but those applications infer
meaning from scores and pulse morphology, not from component index alone.

## What noise looks like

### Correctly whitened stationary noise

If the estimated PSD/covariance is correct, noise-only traces should have
covariance near the identity in whitened coordinates. Their sample
eigenvalues form a finite-sample noise bulk. Within that bulk:

- eigenvectors are unstable across bootstrap resamples;
- eigenvalues depend on trace count and record length;
- adding more PCs improves training reconstruction by construction;
- held-out likelihood or reconstruction need not improve;
- the inverse-transformed PCs can look smooth because whitening colors them
  again.

Consequently, a stable interpretation should stop before the noise bulk, not
declare every component above a fixed index to be "noise PCs."

### Incorrectly weighted or nonstationary noise

PSD mismatch, drift, line noise, changing gain, and time-varying variance
break isotropy after whitening. They can produce stable PCs:

- constant/ramp-like PCs from baseline drift;
- sinusoidal PCs, often in quadrature pairs, from spectral lines;
- localized variance PCs when the noise envelope is locked to trigger time;
- run-dependent PCs from changing noise state.

These are real structures in the sample, but they are nuisance subspaces, not
stationary noise directions.

## What artifacts look like

Artifacts divide into three cases:

1. **Rare and sparse:** glitches, impulses, clipped samples. Ordinary EMPCA is
   not robust to them. Large events can dominate early PCs; smaller events may
   be spread across many PCs.
2. **Repeated with a stable shape:** ringing, pickup, or trigger-synchronous
   transients. These form a low-rank nuisance subspace and may yield stable
   early or middle PCs.
3. **Variable timing or morphology:** pile-up and randomly placed glitches.
   Their shifted copies generally require many PCs, so the resulting spectrum
   can resemble a gradual high-rank tail rather than one artifact component.

For sparse corruptions, a low-rank-plus-sparse model, robust loss, explicit
artifact masks, or iterative residual rejection is better matched than plain
EMPCA. Stable Principal Component Pursuit formalizes the distinction between
small dense noise and gross sparse errors.

## Proposed background experiment

Use the 1 MHz Al2O3/Al analytical model and physical pulse implemented in
`noise_module` so every latent contribution is known.

### Dataset

- Record: 131072 samples, 131.072 ms, `df = 7.629 Hz`.
- Pretrigger: 10%.
- Training/validation/test: independent seeds and no trace reuse.
- Center using training-set statistics only.
- Estimate the PSD on independent noise-only traces.
- Compare frequency-domain inverse-PSD EMPCA with time-domain covariance
  whitening on the same finite grid.

Generate the following nested conditions:

| Condition | Pulse variation | Noise/artifact addition |
|---|---|---|
| C0 | Amplitude only | Stationary Gaussian |
| C1 | Amplitude + arrival jitter | Stationary Gaussian |
| C2 | Add rise/decay variation | Stationary Gaussian |
| C3 | Same as C2 | Baseline offset and slope |
| C4 | Same as C2 | 50/60 Hz line and ringing |
| C5 | Same as C2 | Sparse random glitches |
| C6 | Same as C2 | Random-delay pile-up |
| C7 | Same as C2 | Nonstationary PSD/variance |

Sweep rank 1 through at least 12, artifact rate/amplitude, timing jitter, and
the number of training traces. Repeat the full experiment over independent
seeds.

### Component diagnostics

Do not classify PCs by visual inspection alone. For every rank/component,
measure:

1. **Whitened eigenspectrum:** compare signal training to a noise-only null
   with the same sample count.
2. **Bootstrap stability:** principal angle or matched correlation of each
   component across resampled training sets.
3. **Known-template overlap:** weighted cosine with pulse, `ds/dt`, time
   constant derivatives, baseline/ramp, line sinusoids, and injected artifact
   templates.
4. **Score association:** regress PC scores against true amplitude, arrival
   time, rise/decay constants, artifact label, and baseline parameters.
5. **Localization and spectrum:** time concentration, kurtosis, and spectral
   entropy of each component.
6. **Held-out incremental value:** change in weighted residual, amplitude
   RMSE/bias, and artifact ROC when component k is added.
7. **Null exceedance:** component eigenvalue and stability relative to
   noise-only bootstrap or parallel analysis.
8. **Contamination sensitivity:** principal angle between clean-trained and
   contaminated-trained signal subspaces.

### Operational labels

Assign labels only when the evidence is consistent:

- **Pulse physics:** stable, exceeds the noise null, overlaps a pulse
  derivative, and its score tracks the injected physical parameter.
- **Coherent nuisance:** stable and exceeds the null, but tracks baseline,
  line, run state, or artifact labels rather than pulse parameters.
- **Noise-dominated:** does not reliably exceed the noise-only null and is
  unstable across bootstrap samples.
- **Sparse-artifact contaminated:** strongly associated with artifact labels
  or heavy-tailed scores and rotates/disappears after robust masking.
- **Unresolved:** none of the above gates is met.

## Rank selection

Explained variance alone is not adequate because it always rewards additional
components and can select nuisance variance. Choose rank using a combination
of:

- noise-only parallel analysis or a probabilistic-PCA evidence criterion;
- held-out weighted likelihood/residual;
- stability across seeds/bootstrap samples;
- the downstream quantity of interest, such as unbiased energy resolution;
- a contamination test showing that the selected signal subspace is not
  artifact-driven.

The smallest rank passing the pulse-physics and held-out gates is preferable.
Artifact detection may use additional nuisance components, but they should be
kept semantically separate from the reconstruction signal subspace.

## Decision for this project

The hypothesis to test is not "high PCs are noise." It is:

> After correct whitening, pulse-family directions form a stable low-rank
> subspace above a noise-only null; coherent artifacts form separately
> diagnosable nuisance directions; remaining unstable directions are
> noise-dominated.

This formulation is falsifiable and matches the existing S6 result that timing
variation requires higher rank. It also preserves the archived warning that a
universal rank-2 debiasing/whitening claim was not supported.

## Primary references

- Stephen Bailey, [Principal Component Analysis with Noisy and/or Missing
  Data](https://arxiv.org/abs/1208.4122), 2012.
- Daikang Yan et al., [Processing of X-ray Microcalorimeter Data with Pulse
  Shape Variation using Principal Component
  Analysis](https://arxiv.org/abs/1601.01651), 2016.
- A. Armatol et al., [Pulse Shape Discrimination in CUPID-Mo using Principal
  Component Analysis](https://arxiv.org/abs/2010.04033), 2021.
- Zihan Zhou et al., [Stable Principal Component
  Pursuit](https://arxiv.org/abs/1001.2363), 2010.
- Thomas Minka, [Automatic Choice of Dimensionality for
  PCA](https://proceedings.neurips.cc/paper_files/paper/2000/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf),
  2000.
