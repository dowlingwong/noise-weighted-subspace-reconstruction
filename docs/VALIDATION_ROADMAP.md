# Paper 1 Validation Roadmap

Last reviewed: 2026-06-19.

This is the authoritative source for Paper 1 experiment scope, status,
acceptance criteria, dataset access, preprocessing, metrics, and reporting.
Other planning files are retained only as compatibility pointers or historical
records. The narrative and decision record behind the current status (what was
built, the phase-DOF and finite-sample findings, and the open decisions) is in
[`paper1_validation_progress.md`](paper1_validation_progress.md).

## Shared Infrastructure

- **Central rFFT↔real primitive** — `src/canonical/empca_equivalence_utils.py`
  (`rfft_to_real`, `real_weight_vector`, `complex_to_real_whitened`). All
  rFFT-domain experiments route the DC/Nyquist/weight conventions through this
  one tested function.
- **Canonical EMPCA** — `fit_empca_no_smoothing` (exact, no Savitzky–Golay) is
  the paper-grade entry point; the smoothed `EMPCA.fit()` is not. Stephen
  Bailey's `canonical/empca.py` is the independent reference oracle.
- **Validation harness** — `src/noise_geometry/validation/harness.py` and
  `scripts/sweep.py` provide seed sweeps, leakage-free held-out splits, and
  68%/95% bootstrap (and paired) confidence intervals, emitting one-row-per-seed
  CSV source data. Adopting it on an experiment is a config change.
- **Remote gate runner** — `scripts/stage0_remote_gate.py` enforces a clean
  checkout, runs the exact five Stage 0 commands, and archives environment,
  dependency, git, timing, and command-log evidence.
- **GWpy reference path** — `noise_geometry/gwosc/reference.py` compares the
  repository's one-sided PSD normalization bin-for-bin with GWpy and compares
  direct-rFFT versus inverse-spectrum-FIR whitening on a held-out, edge-trimmed
  off-source window.

## Current Stage

**Stage 0 remote reproducibility gate, followed by Stage 1 synthetic
validation hardening.**

The repository has a working local experiment framework and preliminary
S0-S9 implementations. This is not equivalent to publication-grade
experimental verification.

Honest current status:

- engineering and experiment infrastructure: substantially implemented;
- controlled synthetic demonstrations: runnable but mostly single-seed or
  compact studies;
- paper-grade synthetic validation: incomplete;
- GWOSC validation on downloaded public data: not completed;
- CRESST validation on the selected public release: not completed;
- overall experimental verification: early-to-middle stage, approximately
  one third of the full validation program.

Status terms used in this document:

- **Scaffolded:** entry point or loader exists but has not been exercised on
  the target data.
- **Preliminary:** a deterministic or small demonstration runs.
- **Validated:** the acceptance gate below has passed with held-out data,
  repeated uncertainty estimates, and archived source records.
- **Paper-ready:** validated, independently reproducible, and mapped to a
  frozen paper figure or table.

## Central Claim and Scope

For Gaussian detector noise, the likelihood-aligned reconstruction objective is

```text
(x - x_hat)^H Sigma^{-1} (x - x_hat)
```

Ordinary MSE is the corresponding likelihood only under white isotropic noise.
The estimator hierarchy tested by Paper 1 is:

```text
fixed rank-1 optimal filter
    -> learned rank-k EMPCA
    -> tied linear autoencoder with a noise-weighted loss
```

Paper 1 covers linear reconstruction, optimal filtering, weighted subspaces,
covariance/PSD estimation, and residual calibration. Nonlinear autoencoders,
transformers, NFPA, Paper 2, IceCube, NuBench, OpenFWI, and MicroBooNE are
outside the validation claim set.

## Reproducible Execution

The primary environment is a remote Linux server accessed through VS Code.
Use `uv`; conda is fallback only.

```bash
uv sync --extra dev --extra gwosc
uv run pytest -q
uv run python scripts/run_all_core.py
uv run python scripts/make_tables.py
uv run python scripts/make_all_figures.py
```

Run one experiment with:

```bash
uv run python scripts/run_experiment.py \
  --config configs/synthetic/s5_metric_reversal.yaml
```

Every run must save its config, seed, git commit, metrics, logs, preprocessing
metadata, model metadata, and dataset metadata under `results/`.

## Data Access

The default external data root is:

```text
/ceph/dwong/paper1_dataset
```

Resolution order:

1. CLI `--data-root`;
2. `PAPER1_DATA_ROOT`;
3. config `data_root`;
4. `/ceph/dwong/paper1_dataset`.

Each dataset may use `raw/`, `cache/`, and `processed/` subdirectories.
Download scripts must be idempotent and must not write large files into the
repository unless `--allow-repo-data` is passed explicitly.

### GWOSC

```bash
uv run python scripts/download/download_gwosc.py
uv run python scripts/download/download_gwosc.py --download
uv run python scripts/run_experiment.py \
  --config configs/gwosc/gw150914_smoke.yaml
```

Canonical source: https://gwosc.org/data/

Record the event, detectors, GPS windows, release/version metadata, download
URLs, checksums, and GWOSC acknowledgement. The intended scope is public
real-noise likelihood geometry and matched-filter-style inference, not
gravitational-wave parameter estimation.

### CRESST

```bash
uv run python scripts/download/download_cresst.py
uv run python scripts/run_experiment.py \
  --config configs/cresst/pulse_shape_smoke.yaml
```

Canonical source:
https://www.origins-cluster.de/odsl/dark-matter-data-center/available-datasets/cresst

When no stable direct resource URL is available, prepare the target directory
and print exact manual placement instructions. Record official filenames,
checksums, release metadata, license, and citations. The intended scope is
pulse-shape reconstruction, not a dark-matter exclusion analysis.

### TIDMAD

TIDMAD is optional and must not delay GWOSC or CRESST. Store it under
`/ceph/dwong/paper1_dataset/tidmad` and cite DOI
`10.5281/zenodo.15418539` plus the upstream software.

## Experiment Registry

| ID | Claim or purpose | Config | Current status | Required result before validation |
|---|---|---|---|---|
| S0 | End-to-end pipeline smoke | `configs/synthetic/s0_smoke.yaml` | Locally verified | Clean remote run with recorded environment |
| S1 | OF is unbiased and reaches the CRB | `configs/synthetic/s1_of_crb.yaml` | Amplitude routed through central whitened primitive (== OF to machine precision) | Multi-seed white/pink/Brownian study with CRB intervals and scaling |
| S2 | Rank-1 EMPCA recovers the OF direction | `configs/synthetic/s2_of_empca.yaml` | `amplitude_model` flag added; angle confirmed finite-sample (√N slope ≈ −0.5, tested); equivalence robust to coefficient phase | Multi-seed CRB/scaling report; extend amplitude_model to S6 jitter |
| S3 | Weighted tied linear AE equals EMPCA | `configs/synthetic/s3_ae_bridge.yaml` | Independent: gradient-trained AE vs EMPCA, optimality gap + angle + orthonormality, diagonal and full covariance | Route through central representation; multi-seed intervals |
| S4 | PCA and EMPCA agree in white noise | `configs/synthetic/s4_white_control.yaml` | Held-out, multi-seed; angle ~1e-6°, paired residual CI contains 0 (negative control holds) | Predeclare thresholds |
| S5 | Colored noise can reverse MSE and likelihood rankings | `configs/synthetic/s5_metric_reversal.yaml` | Likelihood-preserving complex representation; held-out paired 95% CIs excluding zero over 20 seeds | Predeclare condition/threshold; archive as final figure source |
| S6 | Extra rank helps only for real signal variation | `configs/synthetic/s6_timing_jitter_rank_sweep.yaml` | Held-out rank sweep, multi-seed; higher rank lowers residual (paired CI excludes 0) | Separate jitter vs shape conditions; predeclare saturation rank |
| S7 | Estimated covariance converges to oracle behavior | `configs/synthetic/s7_covariance_robustness.yaml` | Multi-seed; σ/oracle decreases toward 1 with calibration size | Floor/shrinkage sweeps with CIs |
| S8 | Matched residuals are statistically calibrated | `configs/synthetic/s8_residual_calibration.yaml` | Multi-seed; χ²/dof ≈ 1 (CI ~[0.995, 1.00]) | Residual PSD, autocorrelation, chi-square, and null intervals |
| S9 | Full covariance handles channel correlation | `configs/synthetic/s9_multichannel_covariance.yaml` | Multi-seed; full beats diagonal under correlation (paired CI excludes 0) | Correlation/channel/gain sweeps |
| GWOSC-A | Event-centered PSD/whitening/matched-filter diagnostic | `configs/gwosc/gw150914_smoke.yaml` | Real 32 s cache confirmed PSD normalization to machine precision; whitening under-dispersion exposed insufficient calibration; 256 s disjoint-window rerun prepared | Reproducible run on cached public data with reference normalization checks |
| GWOSC-B | Injection recovery in off-source real noise | same as GWOSC-A | Pipeline scaffolded | Unbiased recovery over windows/SNRs with stable residual diagnostics |
| CRESST-A | Cryogenic pulse reconstruction comparison | `configs/cresst/pulse_shape_smoke.yaml` | Loader/pipeline scaffolded | Actual-release schema validation and leakage-free detector-level evaluation |
| TIDMAD-A | Optional denoising extension | `configs/tidmad/optional_smoke.yaml` | Planned | Start only after CRESST acceptance |

S0-S9 are executed through `scripts/run_experiment.py`; the top-level
experiment scripts are compatibility entry points.

## Preprocessing Contract

Every final config and result record must state:

- baseline subtraction method and interval;
- event-time alignment method, or explicit absence of alignment;
- sample rate, trace length, and record duration;
- real FFT convention and one-sided PSD convention;
- PSD estimator, segment length, overlap, averaging, and floor;
- covariance regularization or shrinkage;
- whitening and unwhitening convention;
- train/validation/test split unit and random seed;
- template normalization and weighted gauge;
- handling of DC, Nyquist, and complex frequency bins;
- detector/channel selection, quality cuts, and rejected samples.

The repository uses NumPy `rfft` layout. Current OF-compatible inverse-PSD
weights set DC to zero, use `2 / PSD` for interior bins, and retain the legacy
OF-compatible Nyquist convention for even-length records. A study may change
these conventions only when the change is explicit and tested.

Public-data splits must prevent time-window, event, detector, or duplicate
trace leakage as appropriate. PSD/covariance estimation data must be
independent of reported test residuals.

## Required Metrics

For residual `r = x - x_hat`:

- raw MSE: `mean(|r|^2)`;
- weighted residual: `r^H Sigma^{-1} r`, with normalization stated;
- Gaussian NLL:
  `0.5 * (r^H Sigma^{-1} r + log(det(Sigma)) + d log(2 pi))`;
- amplitude bias: `mean(A_hat - A)`;
- amplitude resolution: standard deviation or RMSE of `A_hat - A`;
- CRB ratio:
  `sigma_empirical / (s^H Sigma^{-1} s)^(-1/2)`;
- rank-1 agreement: normalized `Sigma^{-1}` inner product;
- rank-k agreement: principal angles after whitening;
- residual calibration: residual PSD, whitened PSD, autocorrelation,
  normalized chi-square, and Gaussian QQ diagnostics.

Raw MSE is a diagnostic, not the sole success criterion under structured
noise. Metrics must be evaluated on held-out traces or independent noise
draws.

## Reporting and Figure Standards

Every quantitative paper-facing result must:

1. show seed/split-level points and 68%/95% uncertainty;
2. state sample counts, seeds, split units, and simulation versus real data;
3. use held-out evaluation;
4. show relevant baselines, oracle values, and null expectations;
5. use paired comparisons when methods share the same test data;
6. show failed diagnostics alongside improved metrics;
7. use predeclared thresholds or report effect sizes without pass/fail wording;
8. use interpretable units such as ratios, relative errors, angles, and
   `1 - cosine`;
9. distinguish implementation/theorem checks from detector-performance claims;
10. archive one-row-per-seed/split/condition CSV or JSON source data.

Existing generated figures are diagnostic until regenerated from a validated
multi-seed result. A script exiting successfully is a computational status,
not scientific acceptance.

## Ordered Checklist and Acceptance Gates

### Stage 0: Remote Reproducibility

- [x] Install and test locally with `uv`.
- [x] Resolve all public data outside the repository.
- [x] Run S0-S9 from configs and generate records, tables, and diagnostics.
- [x] Provide a clean-checkout gate runner and environment/log capture.
- [ ] Run the exact acceptance commands on a clean remote Linux checkout.
- [ ] Record OS, Python, CPU/GPU, dependency versions, and command logs.

**Accept Stage 0 when** all five commands in Reproducible Execution exit
successfully on the remote server without source or config edits.

### Stage 1: Core Synthetic Claims

#### S1: OF and CRB

- [ ] Use white, pink, and Brownian noise with at least 10 independent seeds.
- [ ] Report bias, empirical variance, CRB, intervals, and noise-level scaling.
- [ ] Independently verify FFT/PSD units.

Accept when mean bias is compatible with zero, every matched-condition 95%
interval for `sigma_empirical / sigma_CRB` contains 1, and the fitted
resolution/noise slope agrees with theory within its 95% interval.

#### S2: Rank-1 EMPCA and OF

- [x] Use the full complex rFFT representation.
- [x] Explain the finite-sample angle: it scales as 1/√N (log-log slope ≈ −0.5,
  tested), not a structural gap, and is robust to coefficient phase
  (`amplitude_model`).
- [ ] Match centering, PSD floor, frequency-bin handling, gauge, and scale.
- [ ] Evaluate coefficients and residuals on held-out traces over at least
  10 seeds and three noise colors.

Accept by **demonstrated convergence to 1** at the theoretical √N rate (the
single-seed `0.9999` cosine is reachable only at large N); held-out coefficient
correlation at least `0.9999`; relative weighted-residual/reconstruction
differences below `1e-3`. For the `complex` amplitude model the headline is the
template-into-EMPCA-span angle rather than the single-template cosine.

#### S3: EMPCA and Weighted Linear AE

- [x] Implement a solution independently of `fit_weighted_pca` — a gradient-
  trained tied weighted linear AE (`autoencoders/trained.py`), not the
  closed-form delegate.
- [x] Verify `P.T @ Sigma_inv @ P = I` (M-orthonormality of the learned basis).
- [x] Test diagonal and full covariance.
- [x] Compare subspaces (principal angle), reconstructions, and the optimality
  gap against the EMPCA global optimum `L*`.
- [ ] Route the experiment through the central representation primitive.
- [ ] Multiple seeds and ranks with intervals (via the harness).

Accept the **independent closed form** at maximum principal angle below `1e-5`
degrees and relative projector/reconstruction differences below `1e-8`. For the
**gradient-trained** AE use a separate, principled gate — relative optimality
gap `< 1e-6` and principal angle `< 0.05°` (full-batch L-BFGS, float64; a
first-order Adam pass is reported, not held to this bar).

#### S4-S5: White Control and Metric Reversal

- [ ] Use a likelihood-preserving complex-to-real representation.
- [ ] Use held-out splits and at least 20 paired seeds.
- [ ] Compare PCA, whitened PCA, EMPCA, MSE tied AE, and weighted tied AE.
- [ ] Report paired 68% and 95% intervals.

Accept when white-noise differences are negligible, whitened PCA agrees with
EMPCA in colored noise, and a predeclared colored-noise condition has PCA
winning raw MSE while EMPCA wins weighted residual/NLL with a paired 95%
interval excluding zero.

**Accept Stage 1 when** S1-S5 pass, tests encode the equivalences, and every
figure has seed-level source data and uncertainty.

### Stage 2: Robustness and Boundary Conditions

#### S6: Timing and Shape Variation

- [ ] Separate no-jitter, timing-jitter, and shape-variation conditions.
- [ ] Run held-out rank sweeps over at least 10 seeds.
- [ ] Compare fixed OF, time-shift OF, PCA, and EMPCA.

Accept when rank 1 agrees with OF without variation, higher rank improves only
when extra signal dimensions exist, and improvement saturates near the known
signal-family rank.

#### S7: Covariance Estimation

- [ ] Repeat independent covariance estimates at every calibration size.
- [ ] Sweep PSD floors and covariance shrinkage.
- [ ] Compare oracle, estimated, diagonal, wrong, and identity covariance.

Accept when estimated performance converges toward oracle behavior, a declared
regularization rule prevents low-sample instability, and expected approximation
failures are visible.

#### S8: Residual Calibration

- [ ] Evaluate held-out residual PSD, whitened PSD, autocorrelation,
  chi-square-per-dof, and Gaussian diagnostics.

Accept when matched synthetic residuals are compatible with the simulation
null and intentionally wrong covariance produces detectable failures.

#### S9: Multichannel Covariance

- [ ] Sweep cross-channel correlation, channel count, and gain patterns over
  at least 10 seeds.
- [ ] Compare independent-channel, diagonal-block, and full-block covariance.

Accept when methods agree at zero correlation and full covariance has a paired
95% improvement under nonzero correlation when evaluated with the true metric.

**Accept Stage 2 when** S6-S9 use held-out evaluation, repeated uncertainty,
and documented failure cases. No robustness claim may rely on one seed.

### Stage 3: GWOSC

- [x] Implement a GWpy PSD/whitening reference path and pin its normalization
  against synthetic cached fixtures.
- [x] Confirm the target directory is writable and download reproducible H1/L1
  windows (initial 32 s diagnostic; 256 s rerun required).
- [x] Freeze the event guard and deterministic disjoint calibration/evaluation
  window split in config; persist split indices and starts.
- [x] Use Hann-windowed, FINDCHIRP bias-corrected median PSDs; diagnose and
  archive calibration-window RMS, band-power, crest-factor, and glitch cuts.
- [ ] Replace the approximate chirp with a documented public waveform or a
  justified generation procedure.
- [ ] Pass held-out amplitude calibration over five deterministic split seeds:
  each `null_sigma_over_predicted` in `[0.5, 1.5]`, median in `[0.8, 1.2]`, for
  both detectors. PSD density normalization already agrees with GWpy to machine
  precision.
- [ ] Run event-centered diagnostics and off-source injections over SNRs,
  windows, and PSD choices.

Accept when an empty-cache checkout reproduces the run; checksums and metadata
are saved; injection recovery is unbiased over independent windows; and
whitening/residual conclusions are stable under reasonable PSD choices.

### Stage 4: CRESST

- [ ] Select and download the official release.
- [ ] Validate the loader against the actual schema and metadata.
- [ ] Define detector-level leakage-free splits and preprocessing.
- [ ] Estimate detector-specific noise from valid independent traces.
- [ ] Compare OF, PCA, EMPCA, MSE tied AE, and weighted tied AE with uncertainty.

Accept when a clean raw-data run reproduces one detector-level result; splits
are independent; EMPCA and the independent exact weighted AE agree; and
PCA-versus-EMPCA conclusions use held-out likelihood metrics.

### Stage 5: Paper-Ready Release

- [ ] Freeze configs, seeds, commits, and data checksums.
- [ ] Generate every final figure from archived machine-readable source rows.
- [ ] Map every claim to a script, config, result, figure, and table.
- [ ] Add `CITATION.cff` and verify all dataset citations.
- [ ] Remove private data and DELight dependencies from the public path.
- [ ] Have a second person reproduce one synthetic, one GWOSC, and one CRESST
  result.

The validation package is paper-ready only when Stages 0-4 pass, final runs
come from a clean commit, and both supported claims and observed failures are
documented.

### Optional: TIDMAD

Start only after Stage 4. Use a manageable subset, reproduce the upstream
baseline, compare it with a simple filter and noise-aware model, and do not
let this extension delay Paper 1.

## Result and Figure Mapping

| Result | Command | Primary generated record |
|---|---|---|
| S1 OF/CRB | `uv run python scripts/run_experiment.py --config configs/synthetic/s1_of_crb.yaml` | `results/metrics/S1_of_crb_seed1.json` |
| S2 OF/EMPCA | `uv run python scripts/run_experiment.py --config configs/synthetic/s2_of_empca.yaml` | `results/metrics/S2_of_empca_seed7.json` |
| S3 AE bridge | `uv run python scripts/run_experiment.py --config configs/synthetic/s3_ae_bridge.yaml` | `results/metrics/S3_ae_bridge_seed3.json` |
| S4 white control | `uv run python scripts/run_experiment.py --config configs/synthetic/s4_white_control.yaml` | `results/metrics/S4_white_control_seed4.json` |
| S5 metric reversal | `uv run python scripts/run_experiment.py --config configs/synthetic/s5_metric_reversal.yaml` | `results/metrics/S5_metric_reversal_seed19.json` |
| S6-S9 robustness | `uv run python scripts/run_all_core.py` | metrics plus diagnostic figures |
| GWOSC initial run | `uv run python scripts/run_experiment.py --config configs/gwosc/gw150914_smoke.yaml` | metrics after cache download |
| CRESST initial run | `uv run python scripts/run_experiment.py --config configs/cresst/pulse_shape_smoke.yaml` | metrics after data placement |

This mapping describes current generated diagnostics. Final paper figure
numbers must be assigned only after the corresponding validation gate passes.

## Limitations and Claim Boundaries

- Synthetic checks test controlled Gaussian assumptions and implementation
  conventions, not universal detector performance.
- The old universal rank-k whitening/debiasing claim is not established and
  should remain a boundary study unless held-out model-order experiments
  support it.
- Covariance estimation, finite calibration samples, nonstationarity, and
  non-Gaussian artifacts can materially change conclusions.
- GWOSC and CRESST pipelines are not validated merely because their loaders
  accept a file format.
- Real-data metric reversal is not yet established.
- Generated figures are currently diagnostics, not final paper evidence.

## Document Consolidation Record

Useful material was consolidated as follows:

| Previous file | Disposition |
|---|---|
| `TODO.md` | Checklist and quantitative gates moved here; retained as a pointer |
| `EXPERIMENT_PLAN.md` | 2026-06-13 DELight/K-alpha-era plan archived; durable scientific and plotting cautions moved here |
| `docs/canonical_experiment_plan.md` | Central claim, validation ladder, and public-data intent moved here |
| `docs/experiment_matrix.md` | Claims and honest statuses moved into Experiment Registry |
| `docs/experiment_registry.md` | Entry points/configs moved into Experiment Registry |
| `docs/dataset_access_notes.md` | Storage, commands, links, and scope moved into Data Access |
| `docs/dataset_links_and_access.md` | Duplicate links moved into Data Access |
| `docs/metrics.md` | Definitions moved into Required Metrics |
| `docs/preprocessing_contract.md` | Contract moved into Preprocessing Contract |
| `docs/paper_figure_mapping.md` | Commands and outputs moved into Result and Figure Mapping |
| `docs/limitations.md` | Boundaries moved into Limitations and Claim Boundaries |
| `docs/validation_plan.md` | Validation ladder moved into the staged checklist |
| `docs/rebuild_summary.md` | Retained as a dated migration snapshot |
| `docs/repo_audit.md` | Retained as repository provenance and preservation history |

When this roadmap and a historical document conflict, this roadmap governs the
current validation program.
