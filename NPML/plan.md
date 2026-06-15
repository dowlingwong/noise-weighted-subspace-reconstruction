# NPML Experiment Implementation Plan

This directory is the top-level NPML control plan. The runnable notebook package
already exists under `paper2/npml/`; this file specifies the stronger
experiment package needed for the NPML talk and paper claims.

The plan is based on the current repository implementation:

- Simulation: `QP_simulator/QPSimulator.py`, `QP_simulator/noise_module/*`,
  and helper generators in `implementation/notebook_support.py`.
- Classical models: `src/OptimumFilter.py`, `src/of.py`, and OF/GLS helpers in
  `implementation/notebook_support.py`.
- Linear representation models: PCA/weighted PCA utilities in
  `implementation/notebook_support.py` and EMPCA in `src/EMPCA/`.
- Structured representation models: NFPA code in `src/NFPA/nfpa_demo.py` and
  callable support in `paper2/npml/npml_support.py`.
- Neural reconstruction models: `paper2/models/*`, losses in `paper2/losses/*`,
  and trainers/configs under `paper2/trainers/` and `paper2/configs/`.
- Real detector dataset: `data/k_alpha/k_alpha_traces.h5`,
  `data/k_alpha/k_alpha_rqs.h5`, and `data/k_alpha/template_K_alpha_tight.npy`.
- Noise PSDs: `data/Noise_PSD/noise_psd_from_MMC.npy`, analytic PSDs generated
  by `NoiseGenerator`, and detector noise component tables under
  `data/noise_samples/Al2O3_Al_athermal/`.

## Core Claim

The experiments must support this statement:

> Low-SNR detector reconstruction is governed by detector noise geometry.
> Optimal filtering, PCA/EMPCA, NFPA, and autoencoders are connected by the same
> likelihood geometry, but coverage and architecture decide what can actually
> be learned.

That claim requires more than "EMPCA beats OF on one simulation." Each
experiment below states the simulation, model, dataset, implementation steps,
proof obligation, and figure standard.

## Global Rules

1. Use fixed train/validation/test splits per seed. Do not tune on the test set.
2. Run at least 5 random seeds for talk figures and 10 seeds for paper figures.
3. Every plotted point must have uncertainty:
   - bootstrap 95% confidence intervals over held-out events, and
   - seed-to-seed intervals when training is stochastic.
4. Avoid simple bar plots. Use:
   - line plots with error ribbons for sweeps,
   - point-range or forest plots for categorical comparisons,
   - violin/ECDF/residual histograms for distributions,
   - ROC/efficiency curves with confidence bands,
   - heatmaps only when each cell has uncertainty annotations or a companion
     table with confidence intervals.
5. Report both ML-facing and detector-facing metrics:
   - weighted residual / Mahalanobis residual,
   - native reconstruction MSE,
   - amplitude or energy RMSE and bias,
   - timing RMSE when `t0` is varied,
   - trigger efficiency at fixed false positive rate,
   - subspace angle / component stability where relevant.
6. Save each experiment under a reproducible output directory:

```text
NPML/results/<experiment_id>/
  config.yaml
  metrics_by_seed.csv
  metrics_summary.csv
  bootstrap_intervals.csv
  figures/
  manifest.json
```

For existing Paper 2 neural runs, mirror or reference:

```text
paper2/results/<experiment_name>/
  config.yaml
  metrics.json
  curves.csv
  analysis_metrics.json
  checkpoint_best.pt
  predictions_test.h5
```

## Immediate Commands

Inventory and regenerate the current NPML notebook outputs:

```bash
PYTHONPATH=. python paper2/npml/generate_npml_notebooks.py
```

Dry-run the current Paper 2 training suite:

```bash
PYTHONPATH=. python scripts/run_paper2_training_suite.py --suite all --dry-run
```

Regenerate the real metric x coverage matrix from existing runs:

```bash
PYTHONPATH=. python -m paper2.analysis.real_metric_coverage_matrix
```

Train and analyze the real metric x coverage matrix when needed:

```bash
PYTHONPATH=. python -m paper2.analysis.real_metric_coverage_matrix --train --analyze
```

## Experiment 01: OF Recovery Sanity Test

### Purpose

Show that the framework contains the classical optimum when the OF assumptions
are true.

### Simulation

Use `QPSimulator` in the rank-1 fixed-template limit.

- `tau_decay_range=(3e6, 3e6)`
- `t0_jitter_range=(0.0, 0.0)`
- fixed `n_QP` for pure shape recovery, or a controlled `n_QP_range` for
  amplitude recovery
- Gaussian stationary noise from `NoiseGenerator`
- noise types: `white`, `pink`, `brownian`, and MMC PSD weighting where possible

Preferred helper path:

- `implementation.notebook_support.make_clean_qp_trace`
- `implementation.notebook_support.stationary_noise_generator`
- `implementation.notebook_support.prepare_bundle`

### Dataset

Synthetic only. Use this as a controlled theorem/sanity dataset, not as the
final detector claim.

### Models

- Single-template OF: `src/OptimumFilter.py` or
  `implementation.notebook_support.compute_of_amplitudes`.
- Rank-1 weighted PCA / EMPCA:
  `exact_weighted_subspace(..., k=1)` and `fit_weighted_empca(..., k=1)`.
- Rank-1 ordinary PCA: `exact_isotropic_subspace(..., k=1)`.
- Optional AE sanity: `paper2/models/reconstruction_ae.py`, but it should not
  be claimed to beat OF in this limit.

### Implementation

1. Generate train/test traces from one fixed clean template plus known PSD noise.
2. Compute the OF template direction in the noise-weighted inner product.
3. Fit rank-1 PCA and rank-1 weighted PCA/EMPCA on the training traces.
4. Evaluate:
   - weighted cosine between OF template and learned component,
   - principal angle in degrees,
   - amplitude RMSE on test events,
   - OF Fisher/CRB normalized resolution.
5. Repeat over `N_train = [10, 30, 100, 300, 1000, 3000]` and seeds.

### What To Prove

Analytically show that for `x = A s + n`, `n ~ N(0, Sigma)`, the maximum
likelihood amplitude estimator is the OF estimator and the rank-1 weighted
subspace direction is `Sigma^{-1/2} s` in whitened coordinates. Empirically,
PCA should match only for white noise; EMPCA should converge to the OF direction
for colored noise.

### Figure

Use a two-panel figure:

- Panel A: learned-vs-OF principal angle vs `N_train`, with 95% CI ribbons.
- Panel B: amplitude RMSE / OF CRB vs `N_train`, with seed markers and CI.

No bars.

## Experiment 02: Signal-Shape Variation / Template Mismatch

### Purpose

Show why OF is insufficient when detector signals form a manifold instead of a
single template.

### Simulation

Use `QPSimulator.generate_family`.

Start with currently supported physical/proxy latents:

- amplitude: `n_QP_range`
- timing: `t0_jitter_range`
- pulse shape: `tau_decay_range`

For position, use the current proxy in `paper2/npml/npml_support.py`:
`broaden_trace(...)` and linear position-like distortion. For paper-level
claims, extend `QPSimulator.generate_family` with a native position/channel
sharing latent instead of relying only on the proxy.

### Dataset

Synthetic QP family with full held-out latent variation.

Suggested sweep:

```text
shape_spread:
  tau_decay_range = [(3e6, 3e6), (2.5e6, 3.5e6), (1.5e6, 4.5e6), (1.0e6, 5.0e6)]
timing_spread:
  t0_jitter_range = [(0, 0), (-2e4, 2e4), (-5e4, 5e4), (-1e5, 1e5)]
```

### Models

- Single-template OF.
- Multi-template OF: build templates at representative `tau_decay` and `t0`
  grid points; choose maximum likelihood or solve a small GLS template bank.
- PCA.
- Weighted PCA / EMPCA.
- Optional AE: `ae_prewhite_mahalanobis` for nonlinear reference.

### Implementation

1. Generate a clean test family with full latent variation.
2. For each variation level, generate training traces with the same split size.
3. Fit each model using the same training traces and PSD weights.
4. Learn a linear map from latent coefficients to amplitude/timing labels on
   the training split when a model produces latent coefficients.
5. Evaluate on the same full-variation held-out set.

Relevant current helpers:

- `simulate_controlled_family` via `implementation.notebook_support`
- `rankk_gls_coefficients`
- `_linear_map_from_coeff`
- `_predict_from_linear_map`
- `compute_of_statistics`

### What To Prove

OF is optimal only for the rank-1 fixed-template model. When the signal family
has additional latent directions, a rank-k noise-weighted subspace reduces
expected detector-metric reconstruction error. Multi-template OF is the strong
classical baseline; EMPCA must beat single-template OF and should be competitive
with or better than multi-template OF as the latent family becomes smooth and
low-rank.

### Figure

Use a sweep plot:

- x-axis: template mismatch / latent spread.
- y-axis: amplitude RMSE, timing RMSE, or weighted residual.
- curves: OF, multi-template OF, PCA, EMPCA, optional AE.
- uncertainty: seed CI ribbon plus bootstrap test-event CI.

Add a residual-spectrum panel for one high-mismatch setting to show which
frequencies drive the failure.

## Experiment 03: Noise Covariance Ablation

### Purpose

Prove that "noise-aware" is not just language for PCA.

### Simulation

Use the same signal family as Experiment 02, but sweep the noise model.

Single-channel conditions:

- white
- pink
- brownian
- MMC PSD from `data/Noise_PSD/noise_psd_from_MMC.npy`

Multichannel extension:

- `MultiChannelNoiseGenerator.generate_shared_private`
- `corr_strength = [0.0, 0.1, 0.3, 0.5, 0.8]`

### Dataset

Synthetic train/test families with identical clean signals and changed noise
covariance. Use one held-out clean signal bank per seed so the noise condition
is the only changing factor.

### Models

- PCA on raw traces.
- PCA after whitening.
- Weighted PCA / EMPCA.
- OF / joint OF for rank-1 or template-bank reference.

### Implementation

1. Build the clean traces once per seed.
2. Add each noise condition.
3. Fit raw PCA and weighted PCA/EMPCA.
4. For multichannel traces, flatten `(channel, time)` for EMPCA and preserve
   `(channel, time)` for NFPA/structured variants.
5. Evaluate weighted residual with the true covariance and also with a
   deliberately misspecified covariance for robustness diagnostics.

Current runnable support:

- `paper2/npml/npml_support.run_metric_ablation`
- `paper2/data/whitening.py`
- `QP_simulator/noise_module/multichannel_noise.py`

### What To Prove

For white noise, PCA and EMPCA should converge to the same subspace. For
colored or correlated noise, the principal angle between PCA and EMPCA should
increase and the detector-weighted residual or energy resolution should favor
the noise-aware method.

### Figure

Use a line plot:

- x-axis: noise color/correlation strength or ordered covariance condition.
- y-axis: weighted residual, amplitude RMSE, AUC, or subspace angle.
- curves: PCA, whitened PCA, EMPCA, OF/template bank.
- uncertainty: 95% CI.

For categorical noise types, use point-range plots with CIs, not bars.

## Experiment 04: Detector-Level Trigger Efficiency

### Purpose

Move the validation from ML loss to detector sensitivity.

### Simulation

Use low-energy QP injections:

- energy or amplitude grid: use `n_QP_range` bins corresponding to low to high
  injected amplitudes.
- background: noise-only traces from the same PSD/noise generator.
- signal: clean QP trace plus noise.

Use both:

- synthetic Gaussian PSD noise,
- measured-PSD noise from MMC PSD,
- real/semi-real noise in Experiment 05 when available.

### Dataset

Synthetic signal plus matched noise-only background. For a real-data extension,
use `data/k_alpha` as signal-like detector traces and explicit baseline/noise
segments only if they can be isolated without signal leakage.

### Models / Detection Statistics

- OF amplitude.
- Multi-template OF maximum likelihood score.
- PCA/EMPCA latent matched-subspace projection score.
- Noise-weighted residual reduction:
  `chi2(noise-only model) - chi2(signal-subspace model)`.
- AE score: reconstruction likelihood or residual reduction.

### Implementation

1. Build a background set with `A=0`.
2. Build signal sets over injected energy/amplitude bins.
3. Fit models only on the training/calibration set.
4. Choose thresholds on background only to hit fixed FPR values:
   `0.1%`, `1%`, and `5%`.
5. Evaluate efficiency per injected energy bin on held-out signal traces.
6. Bootstrap events within each energy bin and background set.

### What To Prove

The method improves low-energy efficiency at fixed background rate, or lowers
the 50%/90% efficiency threshold, not merely MSE. This is the highest-value
physics plot for the talk.

### Figure

Primary figure:

- x-axis: injected energy/amplitude.
- y-axis: trigger efficiency.
- curves: OF, multi-template OF, PCA, EMPCA, optional AE.
- one panel per noise condition or line style for synthetic vs measured PSD.
- confidence bands from bootstrap.

Secondary figure:

- ROC curve with bootstrap bands at a representative low-energy bin.

## Experiment 05: Real-Noise / Measured-PSD Injection

### Purpose

Bridge toy simulation and detector reality.

### Dataset Status

Current repository has:

- real K-alpha detector traces and RQs in `data/k_alpha/`,
- MMC one-sided PSD in `data/Noise_PSD/noise_psd_from_MMC.npy`,
- detector noise component tables in `data/noise_samples/Al2O3_Al_athermal/`.

The `total_noise.dat` file is a tabulated spectrum-like file, not a bank of
event-level noise traces. If true noise-only event traces become available,
prefer direct trace injection. Until then, use measured-PSD Gaussian injection
and label it accurately as "measured PSD injection," not "measured noise trace
injection."

### Simulation

Signal: `QPSimulator.generate_family`.

Noise options:

1. synthetic analytic PSDs from `NoiseGenerator`,
2. measured MMC PSD loaded through `paper2/data/whitening.py`,
3. converted `total_noise.dat` spectrum if normalized into the one-sided PSD
   convention expected by the simulator/trainer.

### Models

Same as Experiment 04.

### Implementation

1. Build identical clean injection sets.
2. Add analytic noise and measured-PSD noise with matched integrated variance.
3. Run the same trained/evaluated methods.
4. Report degradation from analytic to measured-PSD conditions.
5. If true noise-only traces are added later, implement direct trace sampling:
   randomly select a baseline trace, circularly shift or crop it, then add the
   simulated signal.

### What To Prove

The performance gain is not an artifact of unrealistically clean white or pink
noise. The expected outcome can include degradation; the point is to quantify
that degradation and identify failure modes.

### Figure

Use trigger-efficiency curves with confidence bands:

- panel A: analytic Gaussian PSD noise,
- panel B: measured MMC PSD injection,
- optional panel C: true measured noise trace injection once available.

Include a residual PSD ratio plot:

```text
mean(|r_f|^2 / PSD_f) vs frequency
```

with bootstrap bands for OF vs EMPCA.

## Experiment 06: Calibration Sample-Efficiency Sweep

### Purpose

Show that noise-aware low-rank learning is useful when calibration data are
limited, and distinguish it from deep models that need more events.

### Simulation / Dataset

Use both:

- synthetic manifold data from Experiment 02,
- real K-alpha traces from `data/k_alpha` for AE/transformer sample-size runs.

### Models

- OF with fixed template.
- PCA.
- EMPCA / weighted PCA.
- AE `paper2/configs/ae_prewhite_mahalanobis.yaml`.
- Transformer `paper2/configs/transformer_prewhite_mahalanobis.yaml` if compute
  permits.

### Implementation

1. Fix one large held-out test set.
2. Sweep calibration sizes:

```text
N_train = [10, 30, 100, 300, 1000, 3000, 10000]
```

3. For each seed and `N_train`, sample a training subset without replacement.
4. Fit PCA/EMPCA and train neural models with the same early-stop validation
   rule.
5. For neural models, reduce epochs for exploratory runs but keep final paper
   runs comparable.

Current neural command pattern:

```bash
PYTHONPATH=. python scripts/run_paper2_training_suite.py \
  --suite best \
  --max-events 2048 \
  --run-suffix sample_efficiency_probe
```

For an exact sample sweep, create copied YAML configs with changed
`data.max_events` and `experiment.name`.

### What To Prove

EMPCA should approach its asymptotic performance with fewer calibration traces
than generic neural models, while OF remains stable but biased under signal
shape variation.

### Figure

Use log-x sample-efficiency curves:

- x-axis: `N_train`.
- y-axis: amplitude RMSE, weighted residual, or trigger threshold.
- curves: OF, PCA, EMPCA, AE, optional transformer.
- uncertainty: seed CI and bootstrap CI.

Add a small table with compute budget and train time per method.

## Experiment 07: Coverage Ablation

### Purpose

Support the slide claim that correct loss geometry is necessary but not
sufficient. The metric weights residuals; it does not create missing latent
coverage.

### Simulation

Use the current proxy coverage generator in `paper2/npml/npml_support.py`:

- amplitude `A`,
- timing `t0`,
- position-like broadening `p`,
- pulse-shape parameter `gamma`.

Training conditions:

- full coverage,
- timing restricted,
- position restricted,
- shape restricted.

Test set always has full latent variation.

### Dataset

Synthetic proxy family for immediate runnable results. For paper claims, extend
`QPSimulator` to include a physical position/channel-sharing latent and repeat.

### Models

- PCA.
- EMPCA / weighted PCA.
- AE metric x coverage matrix:
  `metric_coverage_mahalanobis_full`,
  `metric_coverage_mahalanobis_restricted`,
  `metric_coverage_mse_full`,
  `metric_coverage_mse_restricted`.

### Implementation

Immediate runnable path:

```bash
PYTHONPATH=. python -m paper2.analysis.real_metric_coverage_matrix --train --analyze
```

Synthetic support path:

- `paper2/npml/npml_support.run_coverage_ablation`

### What To Prove

Restricted training support fails on omitted latent directions even when the
Mahalanobis loss is correct. Coverage determines information; the metric cannot
recover factors that were never excited.

### Figure

Use a 2x2 metric x coverage matrix for the talk summary, but do not leave it as
an isolated heatmap. Pair it with:

- point-range plot of each metric with CI by cell,
- latent-specific RMSE curves for omitted factors.

## Experiment 08: Architecture / NFPA / Transformer Bias

### Purpose

Show that architecture selects among likelihood-compatible solutions.

### Simulation

Two branches:

1. NFPA branch:
   - use `paper2/npml/npml_support.run_nfpa_regime_sweep`,
   - separable and non-separable multichannel synthetic signals.
2. Neural branch:
   - use `data/k_alpha` for actual Paper 2 AE/transformer runs,
   - use fixed geometry `prewhitened + mahalanobis`.

### Models

NFPA branch:

- EMPCA,
- NFPA.

Neural branch:

- `experiment_d_linear_prewhite_mahalanobis`,
- `experiment_d_cnn_prewhite_mahalanobis`,
- `experiment_d_transformer_prewhite_mahalanobis`.

Current suite command:

```bash
PYTHONPATH=. python scripts/run_paper2_training_suite.py --suite architecture --dry-run
```

Run on a GPU machine:

```bash
PYTHONPATH=. python scripts/run_paper2_training_suite.py \
  --suite architecture \
  --require-cuda \
  --run-suffix architecture_npml
```

### What To Prove

Under the same likelihood geometry, NFPA should match EMPCA when the signal is
approximately channel x time separable and degrade gracefully when separability
fails. For neural models, similar training likelihood can still lead to
different generalization on held-out amplitude/timing/shape regions.

### Figure

NFPA:

- x-axis: non-separability/distortion level.
- y-axis: weighted residual or reconstruction MSE.
- curves: NFPA and EMPCA with CI.
- add subspace angle as a second panel.

Architecture:

- point-range plot of weighted residual, amplitude RMSE, timing RMSE by model.
- learning curves from `curves.csv` with seed bands.
- no standalone bar chart.

## Experiment 09: Latent Interpretability and Negative Controls

### Purpose

Show that learned coordinates track physical structure, and that performance
degrades when the physical/noise structure is deliberately broken.

### Dataset

Use the synthetic full-latent dataset first because labels are exact:

- amplitude,
- timing,
- position proxy or future physical position,
- shape.

Then repeat on `data/k_alpha` with available RQ labels:

- `A`,
- `time_shift`,
- `OF_ampl_0`,
- `OF_time_0`,
- `pca_amp`.

### Models

- EMPCA coefficients from `rankk_gls_coefficients`.
- AE latent vectors from `paper2/models/reconstruction_ae.py`.
- Transformer latents if exposed from `TransformerReconstructionModel`.

### Implementation

Interpretability:

1. Fit model without using labels.
2. Extract latent coordinates on held-out events.
3. Fit simple post-hoc linear probes on train latents only.
4. Evaluate label prediction on held-out events.
5. Compute correlation and partial correlation with bootstrap intervals.

Negative controls:

- shuffle channel labels for multichannel data,
- use wrong PSD/covariance,
- randomly permute time bins,
- replace learned basis with a random weighted-orthonormal basis,
- train on noise-only and test on signal,
- use a deliberately mismatched OF template.

### What To Prove

Latent coordinates should correlate with real physical factors when the data
contain those factors. Negative controls should degrade detector metrics and
subspace stability, showing that gains come from physical structure plus the
correct metric, not arbitrary dimensionality reduction.

### Figure

Interpretability:

- scatter or hexbin of latent coordinate vs physical label with bootstrap
  regression band,
- 2D latent embedding colored by one physical variable at a time,
- calibration curve for linear-probe prediction.

Negative controls:

- paired point-range plot of metric degradation relative to the unbroken model,
  with each seed connected by a faint line.

## Plotting Template

Every experiment should use a common summary schema:

```text
seed, method, condition, n_train, metric, value
```

Build summaries as:

```text
mean_value
seed_ci_low
seed_ci_high
bootstrap_ci_low
bootstrap_ci_high
n_seeds
n_test_events
```

Recommended Matplotlib patterns:

- Sweep:
  `ax.plot(x, mean); ax.fill_between(x, ci_low, ci_high, alpha=0.2)`.
- Categorical:
  jittered seed points plus `ax.errorbar(..., fmt="o")`.
- ROC/efficiency:
  curve per method plus bootstrap confidence band.
- Residual spectra:
  log-frequency line with percentile band across events.
- Matrix:
  heatmap plus numeric cell values and a companion point-range plot.

## Minimum Talk Package

If time is limited, implement and present these in order:

1. Experiment 01: OF recovery.
2. Experiment 02: signal-shape variation.
3. Experiment 03: noise covariance ablation.
4. Experiment 04: trigger efficiency.
5. Experiment 05: measured-PSD injection.
6. Experiment 06: sample efficiency.

This gives the clean NPML narrative:

```text
OF is recovered in the rank-1 limit.
OF fails when signals form a manifold.
Noise covariance determines the right geometry.
Detector-level trigger efficiency improves near threshold.
Measured PSD/noise realism does not erase the effect.
Noise-aware low-rank learning is sample efficient.
```

## Code Extensions Needed

The following are the concrete missing pieces before all experiments are
paper-ready:

1. Add a native position/channel-sharing latent to `QPSimulator.generate_family`.
   The current broadening-based proxy is fine for talk diagnostics, but not
   enough for a strong physical position claim.
2. Add a reusable multi-template OF helper around the existing OF/GLS utilities.
3. Add a trigger-efficiency evaluation module that produces ROC, AUC,
   efficiency-at-FPR, and threshold-at-efficiency tables.
4. Add bootstrap/seed aggregation utilities shared by all NPML plots.
5. Add real noise-only trace ingestion if a true noise-event bank becomes
   available. Until then, call the existing result "measured PSD injection."
6. Expose neural latent vectors from AE/transformer models for Experiment 09.
7. Convert the current `paper2/npml` exploratory notebook helpers into
   scriptable functions when a result becomes part of the final talk package.

## Acceptance Criteria

An experiment is ready for the NPML talk only when:

- the exact dataset and PSD source are recorded in `manifest.json`,
- every method uses the same train/test events for each seed,
- every plot includes uncertainty,
- detector-facing metrics are reported alongside reconstruction losses,
- the classical baseline is not artificially weakened,
- the figure caption states whether the data are synthetic, measured-PSD
  injected, or true measured-noise injected,
- the result directly supports one slide claim from `NPML_dwong_2026.pdf`.
