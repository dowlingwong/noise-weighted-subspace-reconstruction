# Experiment Protocols

Last consolidated: 2026-06-29.

## Environment

Use `uv` on the remote Linux environment:

```bash
uv sync --extra dev --extra gwosc
uv run --extra dev --extra gwosc pytest -q
```

If `uv` is not on `PATH`, use the installed absolute path or prepend its bin
directory, for example:

```bash
PATH=/home/dwong/.local/bin:$PATH uv sync --extra dev --extra gwosc
```

## Data Root

Default external data root:

```text
/ceph/dwong/paper1_dataset
```

Resolution order:

1. CLI `--data-root`;
2. `PAPER1_DATA_ROOT`;
3. config `data_root`;
4. `/ceph/dwong/paper1_dataset`.

Large public data should not be written into this repository unless an
experiment command explicitly allows it.

## Core Commands

Run all core synthetic experiments:

```bash
uv run python scripts/run_all_core.py
uv run python scripts/make_tables.py
uv run python scripts/make_all_figures.py
```

Run one config-driven experiment:

```bash
uv run python scripts/run_experiment.py \
  --config configs/synthetic/s5_metric_reversal.yaml
```

Run GWOSC data preparation and the baseline experiment:

```bash
uv run python scripts/download/download_gwosc.py \
  --config configs/gwosc/gw150914_smoke.yaml \
  --download \
  --timeout 900

uv run python scripts/run_experiment.py \
  --config configs/gwosc/gw150914_smoke.yaml
```

Run GWOSC follow-ups:

```bash
uv run python scripts/run_experiment.py \
  --config configs/gwosc/filter_statistic_equivalence.yaml

uv run python scripts/run_experiment.py \
  --config configs/gwosc/time_local_noise.yaml
```

Prepare CRESST:

```bash
uv run python scripts/download/download_cresst.py
uv run python scripts/run_experiment.py \
  --config configs/cresst/pulse_shape_smoke.yaml
```

## Experiment Registry

| ID | Purpose | Config | Current status |
| --- | --- | --- | --- |
| S0 | End-to-end smoke | `configs/synthetic/s0_smoke.yaml` | verified through test suite/core workflow |
| S1 | OF uncertainty and CRB | `configs/synthetic/s1_of_crb.yaml` | verified positive |
| S2 | Rank-1 EMPCA and OF direction | `configs/synthetic/s2_of_empca.yaml` | verified positive |
| S3 | Weighted tied linear AE equals EMPCA | `configs/synthetic/s3_ae_bridge.yaml` | verified positive |
| S4 | White-noise PCA/EMPCA control | `configs/synthetic/s4_white_control.yaml` | verified positive |
| S5 | Colored-noise metric reversal | `configs/synthetic/s5_metric_reversal.yaml` | verified positive |
| S6 | Timing variation requires higher rank | `configs/synthetic/s6_timing_jitter_rank_sweep.yaml` | verified positive |
| S7 | Estimated covariance approaches oracle | `configs/synthetic/s7_covariance_robustness.yaml` | verified positive |
| S8 | Whitened residual chi-square calibration | `configs/synthetic/s8_residual_calibration.yaml` | verified positive |
| S9 | Full covariance helps multichannel uncertainty | `configs/synthetic/s9_multichannel_covariance.yaml` | verified positive |
| GWOSC-A | Global PSD real-noise calibration | `configs/gwosc/gw150914_smoke.yaml` | verified negative |
| GWOSC-F1 | Shared-FIR implementation identity | `configs/gwosc/filter_statistic_equivalence.yaml` | verified positive |
| GWOSC-L1 | Time-local PSD real-noise calibration | `configs/gwosc/time_local_noise.yaml` | verified negative |
| CRESST-A | Public cryogenic pulse validation | `configs/cresst/pulse_shape_smoke.yaml` | not completed |
| TIDMAD-A | Optional denoising extension | `configs/tidmad/optional_smoke.yaml` | optional/planned |

## Preprocessing Contract

Every final result record must state:

- baseline subtraction method and interval;
- event-time alignment method or explicit absence of alignment;
- sample rate, trace length, and record duration;
- real FFT convention and one-sided PSD convention;
- PSD estimator, segment length, overlap, averaging, and floor;
- covariance regularization or shrinkage;
- whitening and unwhitening convention;
- train/validation/test split unit and random seed;
- template normalization and weighted gauge;
- handling of DC, Nyquist, and complex frequency bins;
- detector/channel selection, quality cuts, and rejected samples.

Public-data splits must prevent time-window, event, detector, duplicate-trace,
or calibration/evaluation leakage. PSD/covariance estimation data must be
independent of reported test residuals.

## Metrics

Core metrics:

- raw MSE: `mean(|r|^2)`;
- weighted residual: `r^H Sigma^{-1} r`;
- Gaussian NLL with the stated covariance normalization;
- amplitude bias and amplitude resolution;
- CRB ratio: `sigma_empirical / (s^H Sigma^{-1} s)^(-1/2)`;
- rank-1 agreement by normalized `Sigma^{-1}` inner product;
- rank-k agreement by principal angles after whitening;
- residual calibration by PSD, whitened PSD, autocorrelation, chi-square, and
  Gaussian QQ diagnostics.

Detector-facing metrics for real-data domains (GWOSC, CRESST):

- residual whitening ratio `mean(|r_f|^2 / PSD_f)` versus frequency, expected
  near 1 under correct calibration and used to localize the bands that break it;
- detection score by noise-weighted residual reduction:
  `chi2(noise-only) - chi2(signal-subspace)`;
- trigger efficiency versus injected energy/amplitude at a fixed background
  false-positive rate (for example 0.1%, 1%, 5%), with the threshold set on
  background-only data;
- 50% and 90% efficiency thresholds, plus amplitude/energy RMSE and bias.

Report detector-facing metrics alongside reconstruction residuals for any real
detector domain; do not report reconstruction loss alone.

Raw MSE is diagnostic only under structured noise. The primary success metric
must align with the declared noise likelihood.

## Reporting Standards

Every paper-facing quantitative result should:

1. show seed/split-level source rows and uncertainty;
2. state sample counts, seeds, split units, and simulation versus real data;
3. use held-out evaluation;
4. show baselines, oracle values, and null expectations;
5. use paired comparisons when methods share the same test data;
6. show failed diagnostics alongside positive metrics;
7. preserve predeclared thresholds;
8. use interpretable ratios, angles, or relative errors;
9. distinguish implementation checks from detector-performance claims;
10. archive machine-readable CSV or JSON evidence.

## Acceptance Principles

- A command exiting successfully is computational success, not scientific
  acceptance.
- Scientific acceptance requires the predeclared gate to pass.
- Failed gates remain valid evidence and must be preserved.
- Do not tune thresholds, windows, cuts, PSD radii, or model choices after
  inspecting a failed result.
- Sensitivity analyses cannot replace a failed predeclared primary result.

## CRESST Validation Protocol

CRESST is the next candidate for positive public real-data validation:
cryogenic detector-pulse reconstruction where pulses form a shape manifold
rather than a single fixed template. Predeclare the items below before running
on real data. Distilled from the NPML experiment plan; the full methodology is
preserved in `NPML/plan.md`.

Baselines, ordered classical to learned:

- single-template OF (rank-1, fixed template);
- multi-template OF: a small GLS template bank over representative decay-time
  and `t0` grid points, scored by maximum likelihood / minimum weighted
  residual. This is the strong classical baseline and is currently absent from
  the Paper 1 code surface;
- ordinary PCA and whitened/weighted PCA / EMPCA at matched rank `k`;
- the tied weighted linear autoencoder.

The Paper 1 linear claim on real pulses is: EMPCA must beat single-template OF
and stay competitive with multi-template OF as the pulse family becomes smooth
and low rank.

Noise and injection, with an explicit honesty rule:

- prefer direct injection into true noise-only detector traces when an
  event-level noise bank exists;
- otherwise use measured-PSD Gaussian injection and label it "measured-PSD
  injection", never "measured-noise-trace injection";
- match integrated variance between analytic and measured-PSD conditions and
  report degradation from analytic to measured.

Detector-facing evaluation (see Metrics): trigger efficiency versus injected
amplitude at fixed background FPR with the threshold set on background only, the
50%/90% efficiency thresholds, and amplitude/energy RMSE and bias reported
beside the weighted residual, with bootstrap 95% intervals over events in each
energy bin.

Negative controls, each of which must degrade the result:

- wrong PSD/covariance;
- permuted time bins;
- a random weighted-orthonormal basis at matched rank;
- train on noise-only, test on signal;
- a deliberately mismatched OF template.

Gains must come from physical structure plus the correct metric, not from
dimensionality reduction alone.

Predeclaration: fix train/validation/test splits per seed with no event,
time-window, detector, or calibration/evaluation leakage; declare the
template-bank grid, rank `k`, FPR points, and acceptance thresholds before
seeing real-data scores; do not tune any of these after inspecting a failed
result (see Acceptance Principles).
