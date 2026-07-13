# Experiment-agent instruction: CRESST robustness and geometry contrast

Status date: 13 July 2026.

## Objective

Perform a non-tuning robustness reanalysis of the already frozen public
CRESST-A validation.  The purpose is to quantify uncertainty across the seven
run/channel clusters and to separate two empirical questions:

1. **Signal-model capacity:** fixed-template or multi-template OF versus the
   frozen rank-6 EMPCA reconstruction.
2. **Noise geometry at matched nominal rank:** ordinary rank-6 PCA versus the
   frozen rank-6 EMPCA reconstruction.

Do not rerun model selection or change the primary experiment.  Preserve the
original pooled event-bootstrap result as the predeclared primary analysis.
Label every analysis requested here as a post-freeze sensitivity analysis.

## Frozen source of truth

Use the original experiment repository at commit
`9c9f696ff6bca6e6b9729cdb90e41ee92d576787` and the configuration corresponding
to:

- experiment: `P2_CRESST_VALIDATION`;
- seed: `22`;
- rank: `6`;
- covariance model: `psd_stationary`;
- covariance shrinkage: `0.1`;
- split: per-channel chronological 70% calibration / 30% evaluation;
- channels: all seven frozen public test-release channels;
- primary event exclusions, templates, thresholds, and amplitude grid:
  unchanged;
- multi-template competitiveness tolerance: `1.1`;
- original event-bootstrap replications: `300`.

The synchronized numerical reference is:

`transfer_paper/data/cresst/current/cresst/p2_cresst_validation_seed22.json`

Do not modify this archived JSON or its checksums.

## First task: recover auditable event-level outputs

Locate or deterministically regenerate the frozen evaluation-stage per-event
residuals.  Export one row per held-out pulse with, at minimum:

- stable event identifier or original row index;
- run and channel;
- chronological split label;
- `of_weighted_residual`;
- `multi_template_of_weighted_residual`;
- `pca_weighted_residual`;
- `empca_weighted_residual`;
- weighted residuals for all five frozen negative controls;
- any event weight used by the original pooled calculation.

Before new analysis, verify that the export:

- contains exactly 19,335 evaluation pulses across seven channels;
- contains no duplicate event identifiers or trace hashes;
- reproduces every archived per-channel mean within floating-point tolerance;
- reproduces the archived pooled means and the original 300-replicate event
  bootstrap intervals;
- uses calibration-only PSDs, templates, PCA/EMPCA bases, and preprocessing;
- contains no evaluation event in any fitted calibration object.

If the event-level arrays cannot be recovered exactly, stop.  Report the
missing files or code path precisely and do not infer event distributions from
channel means, confidence intervals, or simulated values.

## Required sensitivity estimands

For channel `c` and event `i`, retain the original weighted residual `R` and
compute paired differences:

- `d_OF = R_OF - R_EMPCA`;
- `d_comp = 1.1 * R_multiOF - R_EMPCA` (the frozen gate definition);
- `d_multi_raw = R_multiOF - R_EMPCA` (interpretive only);
- `d_PCA = R_PCA - R_EMPCA` (matched-rank geometry contrast);
- `d_control_j = R_control_j - R_EMPCA` for each frozen negative control.

Report the original pooled event-weighted mean for each estimand, then report
an equal-channel mean in which each of the seven channel-specific paired means
receives weight `1/7`.

Also report scale-normalized descriptive effects by channel:

- `1 - mean(R_EMPCA) / mean(R_OF)`;
- `1 - mean(R_EMPCA) / mean(R_multiOF)`;
- `1 - mean(R_EMPCA) / mean(R_PCA)`.

Give their seven channel values, range, median, and equal-channel mean.  Do not
pool event-level ratios across channels as the primary normalized summary.

## Hierarchical bootstrap

Use a fixed, documented sensitivity-analysis seed and at least 10,000
replicates.  For each replicate:

1. sample seven channels with replacement from the seven observed channels;
2. for every sampled channel occurrence, sample that channel's evaluation
   events with replacement using its observed event count;
3. compute the mean paired difference within each sampled channel occurrence;
4. average the seven channel-occurrence means with equal weight.

Report percentile 95% intervals for `d_OF`, `d_comp`, `d_multi_raw`, `d_PCA`,
and all five negative controls.  If a channel is drawn more than once, perform
an independent within-channel event resample for each occurrence.  State
prominently that only seven clusters make the cluster-level interval coarse
and potentially unstable.

As a transparent nonparametric supplement, report the number of positive
channel effects for `d_OF`, `d_comp`, and `d_PCA`.  An exact sign test may be
reported if clearly labelled secondary and if both its null and one- or
two-sided convention are specified before calculation.  Do not substitute it
for the hierarchical interval.

## Robustness and leakage audit

Produce a concise audit confirming:

- calibration and evaluation are disjoint chronologically within channel;
- PSD estimation uses calibration noise only;
- PCA, EMPCA, linear-AE, OF templates, and multi-template banks use calibration
  data only;
- the same preprocessing and residual normalization are applied to all
  comparators;
- residual normalization and units are explicitly documented;
- no result depends on a post-result change to rank, shrinkage, templates,
  time shifts, exclusions, or thresholds.

Do not introduce alternative ranks, split fractions, covariance models,
template banks, or event cuts in this task.

## Outputs

Create a new versioned sensitivity-analysis directory without overwriting the
frozen primary run.  Archive:

1. event-level residual table in a lossless format plus a compact CSV view;
2. analysis script;
3. machine-readable configuration with source commit and seeds;
4. channel-balanced and hierarchical-bootstrap summary JSON/CSV;
5. leakage/reproduction audit;
6. console log, environment record, and SHA-256 manifest;
7. one compact paper-facing table or figure showing:
   - all seven normalized OF--EMPCA effects;
   - all seven normalized PCA--EMPCA effects;
   - pooled-primary versus channel-balanced sensitivity estimates;
   - hierarchical 95% intervals.

The figure must not portray trigger efficiency as a positive result.

## Interpretation boundaries

The acceptable conclusion, if supported, is limited to held-out
within-channel linear reconstruction on the seven released channels.  Do not
claim:

- cross-channel or cross-detector transfer;
- detector sensitivity or threshold improvement;
- independence of 19,335 events as detector-level replications;
- a universal advantage of covariance weighting if `d_PCA` is heterogeneous;
- evidence about nonlinear autoencoders;
- that a sensitivity analysis replaces the original frozen primary gate.

Report negative, null, heterogeneous, or sign-reversed results exactly as
observed.  No acceptance criterion should be changed after inspection.

## Handoff back to the writing repository

Synchronize the new outputs into `transfer_paper/` without modifying existing
archived evidence.  Provide the writing agent with:

- exact relative paths and checksums;
- the channel-balanced point estimates;
- the hierarchical intervals;
- all seven normalized channel effects;
- the PCA--EMPCA geometry result;
- reproduction discrepancies, if any;
- a one-paragraph claim recommendation and explicit claim boundaries.
