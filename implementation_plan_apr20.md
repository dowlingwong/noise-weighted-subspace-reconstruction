# Implementation Plan Apr 20

## Purpose

This document is the computation-support plan for Paper 1.

It is not a replacement for the manuscript-writing plan in:

- `plan.md`
- `block_01_positioning_and_intro.md`
- `block_02_unified_objective_and_hierarchy.md`
- `block_03_optimal_filter_rank1_case.md`
- `block_04_empca_rankk_case.md`
- `block_05_linear_ae_bridge.md`
- `block_06_convergence_and_practicals.md`
- `block_07_experiments_and_tables.md`
- `block_08_discussion_appendix_and_cleanup.md`

Instead, this file defines the numerical, algorithmic, and reproducibility work that must exist underneath those manuscript sections so the paper's claims are actually supportable.

## Executive framing

The paper needs three clearly separated evidence regimes:

1. `theorem regime`
   Clean matched-assumption setting for the exact OF / rank-1 EMPCA / linear-AE claims.
2. `real-data regime`
   Held-out K-alpha calibration analysis showing the theory survives realistic measured data under matched preprocessing.
3. `robustness regime`
   Non-stationary, nonlinear, correlated, and artifact-rich stress tests showing practical behavior beyond the theorem assumptions.

That separation is required by both the existing paper plan and the reviewer comments in `EMPCA_improvement.pdf`.

## Planning constraints inherited from the attached docs

From the current paper plan, the implementation must satisfy these constraints:

- keep the central theorem story mathematically clean;
- make OF, EMPCA, and the noise-aware linear AE read as one family under one Gaussian ML objective;
- add the missing CRB / variance / energy-resolution thread for OF;
- make joint-channel OF real, not a stub;
- make AE equivalence a payoff, not a detached appendix note;
- provide convergence, initialization, conditioning, and rank-selection guidance;
- populate the experiment tables with real numerical evidence;
- keep the structured-noise module out of the theorem core and use it as robustness support;
- keep nonlinear trigger ideas scoped to Paper 2.

## Core implementation principle

Every computation should be labeled by the claim it supports.

Use this discipline:

- `proof-support`: matched assumptions only, no out-of-model noise tricks.
- `real-support`: measured data, identical preprocessing across methods.
- `robustness-support`: out-of-model perturbations used to test degradation, not prove equivalence.

Do not let robustness experiments replace proof-support experiments.

## Paper-section alignment

### `02_unified_objective.tex`

Needs support from:

- a canonical preprocessing specification;
- PSD and whitening conventions;
- a single hierarchy metadata table describing estimator, constraint, degree of freedom, and solution form;
- diagonal PSD whitening and full-covariance whitening notes.

### `03_optimal_filter.tex`

Needs support from:

- a verified OF implementation under the same conventions as EMPCA;
- Fisher information, variance, CRB, and calibrated energy-resolution calculations;
- time-shift OF and, if data permit, joint-channel correlated OF;
- a clean statement of what the fixed-template model misses.

### `04_empca.tex`

Needs support from:

- strict rank-1 OF vs EMPCA verification on real K-alpha data;
- theorem-regime synthetic verification with planted signal and stationary Gaussian noise;
- gauge and amplitude-normalization checks;
- rank-`k` weighted residual improvement curves;
- finite-sample caveat diagnostics.

### `05_linear_ae.tex`

Needs support from:

- exact complex-whitened EMPCA vs linear AE equivalence;
- native weighted vs whitened representation comparison;
- one ablation proving the difference is loss geometry, not just architecture;
- a short coordinate-descent bridge between EMPCA updates and AE language.

### `06_convergence.tex`

Needs support from:

- objective-vs-iteration traces;
- seed-to-seed stability runs;
- initialization comparisons;
- rank-selection diagnostics;
- conditioning / regularization sweeps.

### `07_experiments.tex`

Needs support from:

- theorem-verification tables;
- K-alpha dataset characterization;
- rank-`k` saturation plots;
- noise-aware-vs-isotropic ablation;
- convergence figures;
- structured-noise robustness panels;
- completed Study A / Study B summary tables.

### `08_discussion.tex` and `appendix.tex`

Needs support from:

- a precise statement of theorem scope vs robustness scope;
- appendix-level simulation and noise-module details;
- reproducibility metadata for datasets, weights, seeds, and run settings.

## Main workstreams

### 1. Canonical data interface and preprocessing audit

Create one reusable analysis path for:

- baseline subtraction;
- train / validation / test split;
- FFT convention;
- PSD estimation;
- one-sided OF weights;
- whitening transform;
- template normalization;
- coefficient normalization and phase / sign gauge choices.

This is the foundation. No later result should use a different hidden convention.

### 2. OF baseline, CRB, and calibration thread

Implement and report:

- amplitude estimator;
- normalization term;
- Fisher information;
- variance / CRB comparison;
- propagation into energy-resolution units or calibrated detector metric;
- joint-channel extension if channel data are available.

This is the missing detector-facing statistical backbone highlighted in the review PDF.

### 3. Strict real-data OF vs rank-1 EMPCA verification

Use held-out K-alpha traces with matched preprocessing and matched PSD weighting.

Primary metrics:

- weighted cosine / principal angle;
- calibrated amplitude agreement;
- reconstructed-trace agreement;
- weighted residual-energy comparison;
- residual KS statistic and confidence interval if practical.

EMPCA smoothing must be disabled for the strict theorem-support run.

### 4. Synthetic theorem-regime verification

Generate signals from the available clean signal model and add only stationary Gaussian noise consistent with the measured PSD.

Show:

- planted-template recovery;
- subspace recovery as event count increases;
- amplitude bias and variance behavior;
- equality of OF and rank-1 EMPCA reconstructions under matched assumptions.

This is the cleanest empirical support for the formal theorem statements.

### 5. Gauge, amplitude, and linear-AE bridge checks

Separate three related but distinct checks:

- rank-1 gauge degeneracy and normalization matching;
- rank-`k` amplitude / coefficient comparability across rotated bases;
- EMPCA vs exact tied linear AE equivalence in whitened complex space.

Also run the native weighted vs whitened representation comparison already suggested by the repo artifacts.

### 6. Convergence, initialization, conditioning, and rank selection

Run a practical training-recipe study that answers:

- what objective decreases monotonically and under what update loop;
- whether smoothing breaks monotonicity;
- how sensitive the learned subspace is to random initialization;
- when whitening or normal equations become ill-conditioned;
- how to choose `k` in practice.

Primary selection rule:

- held-out weighted residual / `chi^2` saturation.

Secondary selection rules:

- whitened scree curve;
- subspace stability across seeds;
- energy-resolution improvement saturation.

### 7. Rank-`k` performance and noise-aware-vs-isotropic ablation

Support the paper's broader practical claim:

- rank-`k` EMPCA should reproduce OF at `k = 1`;
- improve fit only when extra physically relevant structure exists;
- outperform isotropic-loss baselines when the colored-noise model matters.

Compare against:

- OF;
- weighted EMPCA;
- plain PCA or isotropic linear AE;
- noise-aware linear AE.

### 8. PC interpretation and centered-vs-uncentered analysis

Use the existing interpretation notebooks as a starting point, but make the conclusions explicit and conservative.

Target outputs:

- `PC1` vs empirical mean / template / OF-like direction;
- `PC2` vs timing-derivative proxy;
- `PC3` vs width / shape proxy;
- coefficient correlations with amplitude, timing, and deformation proxies;
- centered vs uncentered leading-mode comparison.

Important: do not conflate "rank-1 equivalence to OF under matched assumptions" with "PC1 from general rank-`k` uncentered training equals OF."

### 9. Structured-noise robustness and multichannel stress tests

Use the nonlinear / nonstationary / correlated / artifact noise module only in the robustness regime.

Recommended uses:

- noise-aware-vs-isotropic ablation under drift / lines / glitches / sparse impulses;
- rank-`k` degradation study under piecewise-stationary noise;
- optional multichannel correlated-noise comparison;
- residual-diagnostic behavior beyond the theorem model.

Do not use these runs as the primary evidence for the equivalence theorems.

### 10. Tables, figures, and reproducibility packaging

Finish with a paper-facing artifact layer:

- theorem-verification table;
- K-alpha dataset summary table;
- rank-`k` summary table;
- Study A and Study B completed tables;
- convergence figure set;
- robustness figure set;
- run metadata and seeds;
- manifest of PSDs, templates, weight files, and splits.

## Priority order

Implement in this order:

1. preprocessing / PSD / whitening audit;
2. OF baseline with CRB / variance / resolution formulas;
3. strict real-data OF vs rank-1 EMPCA verification;
4. synthetic theorem-regime verification;
5. EMPCA vs linear-AE equivalence plus gauge handling;
6. convergence / initialization / conditioning / rank selection;
7. rank-`k` and noise-aware-vs-isotropic studies;
8. PC interpretation and centering ablation;
9. structured-noise and multichannel robustness;
10. final paper tables, figures, and reproducibility packaging.

## Folder structure for this plan

Detailed execution blocks live in:

- `plan/implementation_blocks_apr20/`

Those blocks intentionally mirror the style of the manuscript block docs, but they are narrower and computation-oriented.

## Block index

Use the following block files in order:

1. `block_01_claim_mapping_and_execution_rules.md`
2. `block_02_data_preprocessing_and_psd_audit.md`
3. `block_03_of_baseline_crb_and_real_rank1_verification.md`
4. `block_04_synthetic_theorem_regime.md`
5. `block_05_linear_ae_bridge_and_gauge.md`
6. `block_06_convergence_initialization_and_rank_selection.md`
7. `block_07_rankk_quality_and_noiseaware_ablation.md`
8. `block_08_pc_interpretation_and_centering.md`
9. `block_09_structured_noise_and_multichannel_robustness.md`
10. `block_10_tables_figures_appendix_and_reproducibility.md`

## Definition of done

This plan is done only when all of the following are true:

- each theorem claim in the manuscript has an explicit supporting computation;
- theorem-support and robustness-support regimes are clearly separated;
- the paper sections no longer contain placeholder claims without corresponding data;
- the structured-noise module appears as robustness support and appendix material, not theorem-core justification;
- there is a reproducible artifact path from raw traces / PSDs / templates to paper tables and figures.
