# CRESST Predeclared Validation Gates

Status: **APPROVED — frozen before Stage C5.**

v2 fixed the split design (see below). v3 addresses two sample-size
consequences of that fix that a second review pass surfaced: per-channel
covariance estimation is now in the N ~ p regime, and per-channel FPR
thresholds are set by small background pools. Both are addressed below and
frozen before any real-data score has been seen. This file freezes every
threshold, grid, rank, FPR point, split definition, and acceptance criterion
for the CRESST-A predeclared validation run
(`configs/cresst/cresst_validation.yaml`), per Stage C4 of
`docs/runbooks/CRESST_REMOTE_AGENT_RUNBOOK.md` and the CRESST Validation Protocol in
`docs/EXPERIMENT_PROTOCOLS.md`.

## Audit trail: commit `3b93cbd`

`git log` shows a commit `3b93cbda889060af986c53966bd76233a04bd51b`
("update", author `dwong <dowling.wong@kit.edu>`, 2026-07-04 02:53:31 +0200)
already in history, immediately after `6f6f6e3 added playbook` -- it was
not created by the acting agent in this session. `git show --stat` confirms
its contents are exactly the v1/v2 draft (`cresst_validation.yaml`,
`CRESST_PREDECLARED_GATES.md`, `download_cresst.py`, and the
`src/noise_geometry/cresst/{analysis,detector_metrics,injections,loader,
negative_controls,splits,templates}.py` modules plus their tests) --
i.e. an out-of-band checkpoint of the prior review round, not a scientific
run and not a CRESST score. It precedes this v3 freeze and does not affect
any value declared below. Recorded here rather than silently absorbed, so
anyone auditing this record later knows the commit exists and what it is.

## Why the split changed (v1 -> v2)

v1 split the test release by assigning whole `(run, channel)` groups
cross-channel (~5 channels calibration, ~2 held out). But Sigma (the noise
covariance/metric) is a per-detector-channel quantity: a covariance
estimated from channels A-E does not describe channel F's noise (published
OF resolutions differ by large factors between channels, as do offsets and
noise spectra). Evaluating channel F's pulses under channel A-E's metric
tests domain transfer, which Paper 1 does not claim, and would likely fail
for the same "wrong noise model" reason GWOSC did -- telling us nothing
about the linear hierarchy.

**v2**: split *within* each channel, chronologically. Within each channel,
the pulse stream and the noise stream are each sorted by `hours` and cut
*separately*: the earliest `channel_calibration_fraction` (0.7) of that
channel's pulses is pulse-calibration (template bank, subspace fits), and
independently the earliest 0.7 of that channel's noise is noise-calibration
(Sigma/PSD); the later remainder of each stream is that stream's evaluation
segment (test pulses; background-FPR noise, injection substrate). Each
stream's own split is chronological and leakage-safe (zero calibration/
evaluation row or release-trace-hash overlap, verified per channel -- see
the post-hoc correction below for the precise claim this supports), and
turns the 7 test-release channels into 7 quasi-independent replications
instead of one fragile 5-vs-2 channel split. The old "7-group granularity"
caveat from v1 no longer applies -- every channel contributes its own
held-out evaluation, not a coin-flip group assignment.

### Post-hoc correction (2026-07-13): pulse/noise cross-stream ordering

The post-freeze sensitivity analysis
(`transfer_paper/data/cresst/sensitivity/20260713_channel_robustness_v1/`,
`leakage_audit.json`) found that because the pulse and noise streams are
split *separately* (each internally chronological, per the corrected
description above), their cutoffs are not synchronized: in channels
`35_57.0` and `35_58.0`, respectively 6 and 3 evaluation pulses (of 19,335
pooled evaluation events total) have `hours` earlier than that channel's
final calibration-noise timestamp. No evaluation pulse entered any fitted
signal model (PCA/EMPCA/AE/template bank are fit on calibration pulses
only, Sigma/PSD on calibration noise only, and the row/hash overlap check
that matters for that -- calibration vs. evaluation *within* the same
stream -- remains exactly 0 in every channel), so this does not invalidate
the frozen Stage C5 result. It does mean the strongest possible verbal
claim, "all calibration data precede all evaluation data," is not literally
correct across streams. Use instead: **pulse and noise streams were each
split chronologically, separately, with zero row/hash overlap between that
stream's calibration and evaluation segments.** A future strict prospective
replication should use one shared time cutoff per channel for both streams
together, at the cost of unequal calibration/evaluation fractions between
pulses and noise. This is a documentation correction only -- the frozen
split fraction, seed, and code are unchanged, and Stage C5 was not rerun.

## Sample-size consequences of the per-channel redesign (v3)

Splitting within each channel instead of pooling ~5 channels of calibration
data means every per-channel sample is much smaller than it was in v1. Two
distinct problems follow, both frozen here rather than resolved after
seeing a failed gate.

### Covariance estimation is now N ~ p

Each channel's calibration noise segment is only ~500-540 traces
(0.7 x ~730-770 total per channel) against `max_features: 512`. The raw
observation-space sample covariance (`estimate_covariance`) has
`512 x 513 / 2 ~ 131,000` free parameters against ~500 samples -- badly
underdetermined; the v1/early-v2 declaration of `covariance_shrinkage: 0.1`
would have been carrying the entire burden of making that near-singular
matrix usable, undeclared as such.

**Frozen choice: `covariance_model: psd_stationary`.** Instead of the raw
sample covariance, estimate a one-sided PSD per channel
(`estimate_psd_ensemble`, median-averaged across that channel's calibration
traces -- exactly the GWOSC convention) and build the implied **circulant**
covariance from it (`noise.covariance_from_psd`): only `512/2 + 1 = 257`
free parameters, an N ~ 2p regime instead of N ~ p. This is not a new
modeling assumption -- the codebase's rFFT-based OF/whitening machinery
(`estimate_psd_rfft`, `inverse_psd_weights`, `filters.optimal`) already
treats the noise as stationary/circulant; this makes that assumption
explicit and applies it to covariance estimation too, matching the paper's
own frequency-domain framing. `covariance_shrinkage: 0.1` is kept on top as
an additional numerical safety floor, not a replacement for this fix.

Alternatives considered and rejected: heavier shrinkage alone (papers over
the problem rather than fixing the parameter count, and picking "how much
heavier" after seeing a result would itself be tuning); reduced
`max_features` (would also reduce timing resolution for the multi-template
bank and the physical pulse-shape information EMPCA/PCA/AE can use --
undesirable for the Paper 1 claim itself, which is precisely that shape
structure matters).

### FPR thresholds are set by small per-channel background pools

The 30% evaluation segment leaves only ~220 background (held-out noise)
traces per channel. A 1% FPR threshold is then set by the top ~2 order
statistics of that pool, and 0.1% by the single most extreme trace --
already why 0.1% was demoted to diagnostic in v2, but 1% deserved the same
scrutiny and didn't get it in v2.

**Frozen choice: three-tier FPR reporting**, all still computed for every
channel, only the *interpretation* differs:

- **5% -- primary, gate-relevant** (`fpr_primary_points: [0.05]`): set by
  ~11 order statistics per channel, the least noisy of the three.
- **1% -- reported with threshold uncertainty**
  (`fpr_threshold_ci_points: [0.01, 0.05]`): alongside the point-estimate
  threshold, a bootstrap 95% CI on the *threshold itself*
  (`background_threshold_ci95` in the run record, via
  `detector_metrics.background_threshold_ci`), not just on the resulting
  efficiency.
- **0.1% -- diagnostic only** (`fpr_diagnostic_points: [0.001]`, unchanged
  from v2): still computed and recorded, never gate-relevant.

Alternative considered and rejected: an asymmetric calibration fraction for
noise specifically (e.g. 50/50 noise vs. 70/30 pulses, to grow the
background pool). Rejected because it trades this problem for the
covariance-conditioning problem above (fewer calibration noise traces),
and because a single declared `channel_calibration_fraction` for the whole
channel is simpler to defend as not selectively tuned. The PSD-stationary
covariance model already tolerates the smaller calibration count
(N ~ 2p instead of N ~ p), so the trade isn't necessary.

## Rank

- `rank: 6`, unchanged.

## Multi-template OF bank (Stage C3 item 1) -- approved as-is

- Grid: terciles of `rise_time` x terciles of `decay_time` (up to 9
  empirical templates per channel, fit on that channel's own calibration
  pulses), `min_cell_pulses: 20`, `t0_grid_samples: [-4,-2,0,2,4]`.
- Scoring: minimum weighted residual (max GLS chi2 reduction) over the bank,
  per channel, evaluated on that channel's own held-out pulses.

## Detector-facing metrics (Stage C3 item 2)

- Residual whitening ratio and detection score
  (`chi2(noise-only) - chi2(signal-subspace)`): unchanged in definition, now
  computed per channel from that channel's own calibration noise/metric.
- **Amplitude grid, revised**: anchored to each channel's *published*
  `resolutions_test.csv` OF resolution (sigma_OF), not to a held-out-pulse
  median. Grid = `[0, 1, 2, 3, 4, 5, 6, 8, 10] x sigma_OF(channel)`
  (`amplitude_grid_multiples_of_sigma_of` in `detector_metrics` config). This
  samples the efficiency-crossing region densely (which occurs at a few
  sigma_OF, since released pulses already survived triggering and sit far
  above threshold) and makes curves comparable across channels in SNR
  units. `n_injections_per_amplitude: 200`.
- **Real-data finding**: the test split's channel `(35, 56)` has trace data
  but no row in `resolutions_test.csv` (conversely, `resolutions_test.csv`
  has a row for `(35, 59)`, which has no trace data in this file). This is a
  release gap, not a code bug. For a channel missing its published
  resolution, the fallback is the analytic CRB amplitude-estimator std,
  `sigma_OF = 1/sqrt(template^T Sigma^-1 template)`, computed from that
  channel's own calibration-segment template/metric -- the same physical
  definition the published resolution uses, computed rather than looked up.
  (An earlier version of this fallback used the standard deviation of
  reconstructed pulse amplitudes, which is the physical *signal* scale, not
  the noise-limited *resolution* scale -- those differ by ~100-1000x for
  this detector, e.g. published sigma_OF ~ 0.001-0.003 against a median
  `pulse_height` of ~0.85. That was caught and fixed before any real
  acceptance-gate number was interpreted.) `sigma_of_source` in the run
  record ("resolutions_test.csv" vs. "fallback_of_crb_no_published_resolution")
  reports which applied, per channel.
- FPR points: `[0.001, 0.01, 0.05]`, numbers unchanged, but reported in
  three tiers (5% primary, 1% with a threshold CI, 0.1% diagnostic-only) --
  see "Sample-size consequences" above for the full rationale. The one
  exception to "keep the FPR points" is in *tiering the interpretation*,
  not in changing the numbers.
- Efficiency thresholds: `[0.5, 0.9]`, unchanged.
- `bootstrap_reps: 300`, unchanged, now used for the amplitude bias/RMSE
  CIs, every paired-difference gate CI (see Acceptance gate below), and the
  background-threshold CI (see "Sample-size consequences" above).

## Injection machinery (Stage C3 item 3) -- template now specified explicitly

- **Primary — direct injection**: `amplitude * template` into real
  noise-flagged baseline traces from that channel's held-out (evaluation-
  segment) noise bank.
- **Secondary, labeled — "measured-PSD injection"**: Gaussian noise from the
  covariance estimated from that same held-out noise bank, plus
  `amplitude * template`. Never called "measured-noise-trace injection."
- **Template, now stated explicitly** (previously implicit): in both
  conditions, `template` is the per-channel single-OF template
  (`BaselineFit.of_template`), fit on that channel's own calibration
  segment -- the same template used for that channel's single-template-OF
  baseline, not a global or cross-channel template.
- Degradation between conditions: difference in the 50%-efficiency-crossing
  amplitude, per channel per FPR.

## Negative controls (Stage C3 item 4) -- approved as-is, now paired

Same 5 controls as before (`wrong_psd`, `permuted_time_bins`,
`random_weighted_basis`, `noise_only_train`, and `mismatched_of_template`
from the fuller protocol list, included for completeness). Each control now
reports a **paired bootstrap comparison** against its nominal (EMPCA for 4
of them, single-template OF for `mismatched_of_template`) on the same
held-out per-channel test events, not just point-estimate means (see
Acceptance gate).

## Acceptance gate (Stage C4 quantitative criterion) -- revised for uncertainty

> EMPCA at rank k must beat single-template OF and remain competitive with
> multi-template OF on the likelihood-aligned metrics, with all negative
> controls degrading as expected.

**v1 problem**: `empca_weighted_residual < of_weighted_residual` is a point
estimate that passes on a hair-thin win and fails on a hair-thin loss, with
no uncertainty. Same for each negative control's raw mean comparison. Per
Reporting Standard #5 (paired comparisons with uncertainty, since both
methods score the same held-out events), every gate boolean now requires a
**paired bootstrap 95% CI to exclude zero in the required direction**:

- `empca_beats_single_template_of`: bootstrap CI of
  `of_residual - empca_residual` (per event, pooled across channels) has a
  lower bound `> 0`.
- `empca_competitive_with_multi_template_of`: bootstrap CI of
  `1.1 x multi_template_of_residual - empca_residual` (pooled) has a lower
  bound `> 0` -- i.e. EMPCA is *confidently* within the 10% tolerance, not
  just on a point estimate. **The `1.1` tolerance itself is still the one
  number in this document that is a judgment call rather than dictated by
  the runbook**, now applied CI-aware per the above.
- Each negative control's `degraded`: bootstrap CI of
  `control_residual - nominal_residual` (pooled) has a lower bound `> 0`.

**Predeclared aggregation**: the primary gate pools paired per-event
differences across all used channels (concatenation, since the weighted
residual is already noise-normalized and comparable in that sense across
channels) and bootstraps that pooled array. Per-channel results (mean
difference, CI, pass/fail) are also reported under `per_channel`, and any
channel whose own sign disagrees with the pooled sign is flagged under
`pooled.*_sign_reversed_channels` -- a sign reversal does not by itself fail
the gate (the pooled CI is authoritative) but must be reported, not hidden.

`bootstrap_reps: 300` is used for every paired-CI computation (unchanged
declared value, now serving double duty).

## Other frozen values

- `covariance_model: psd_stationary` (new in v3, see above);
  `covariance_shrinkage: 0.1` (unchanged, now a safety floor on top of the
  PSD-based covariance rather than the sole regularizer of a near-singular
  empirical one).
- `baseline_fraction: 0.2`
- `max_features: 512` (native downsampled resolution)
- `split: test`; `channel_columns: [run, channel]`;
  `channel_calibration_fraction: 0.7` (uniform for both pulses and noise,
  see "Sample-size consequences" above for why an asymmetric noise fraction
  was rejected); `channel_time_column: hours`
- `fpr_primary_points: [0.05]`; `fpr_threshold_ci_points: [0.01, 0.05]`;
  `fpr_diagnostic_points: [0.001]` (new in v3)

## Acknowledging the Stage C2 smoke run

Stage C2 (`pulse_shape_smoke.yaml`) is a computational-only check and Hard
Rule 3 forbids quoting its metrics as a scientific result. Its output did,
however, print real weighted residuals for the *same quantity* the gate
compares (EMPCA 0.71 vs. single-template OF 4.74, before the per-channel
redesign; both channel-pooled means fall in a similar range after it). This
is mild contamination of exactly the number the headline gate is built
from, and a referee-proof C4 record should say so plainly rather than
assert cleanliness. It is defensible for three reasons, not because it
didn't happen: (1) the smoke config pools channels with different
rank/feature count than the frozen validation config (rank 4 / 256 features
vs. rank 6 / 512 features here), so the two are not numerically comparable;
(2) no threshold, grid, or gate value in this document was chosen or
adjusted in response to those numbers -- every value here was fixed from
the runbook, the protocol doc, or a stated engineering rationale
independent of the smoke output; (3) the acceptance gate is directional
(does EMPCA beat OF, is it competitive, do controls degrade) rather than a
fitted numeric threshold, which is less exploitable by incidental exposure
than a tuned cutoff would be. Recorded here for the record, not to
pre-argue away a conflict of interest.

## Sign-off checklist before Stage C5

- [x] Per-channel, time-disjoint split (chronological by `hours`, 0.7 calibration fraction, seed 22) replacing the flawed cross-channel group split -- reviewed and approved by design critique.
- [x] Rank `k=6` and FPR points `[0.001, 0.01, 0.05]` -- carried over unchanged.
- [x] **New in v3**: `covariance_model: psd_stationary` for the N ~ p per-channel covariance problem, with `covariance_shrinkage: 0.1` retained as a safety floor -- reviewed.
- [x] **New in v3**: three-tier FPR reporting (5% primary, 1% with threshold CI, 0.1% diagnostic) for the thin per-channel background-noise problem -- reviewed.
- [x] Template-bank grid (terciles x terciles x 5 shifts, `min_cell_pulses=20`) -- approved as-is.
- [x] Amplitude grid anchored to per-channel published sigma_OF (`[0,1,2,3,4,5,6,8,10] x sigma_OF`), with the CRB-formula fallback for the one channel `(35,56)` missing a published resolution -- reviewed.
- [x] 5 negative controls (including `mismatched_of_template`) -- approved as-is.
- [x] `multi_template_of_competitive_tolerance: 1.1`, now applied via paired-bootstrap CI -- approved as-is (CI-aware).
- [x] Paired-bootstrap gate criteria (CI excludes zero, not point estimate) with pooled-across-channels aggregation and per-channel sign-reversal reporting -- reviewed.
- [x] Injection template = per-channel single-OF template from that channel's calibration segment -- confirmed explicit.
- [x] Stage C2 smoke contamination acknowledgment above -- read.
- [x] `n_injections_per_amplitude=200`, `bootstrap_reps=300`, `baseline_fraction=0.2`, `max_features=512` -- approved as-is.
- [x] Commit this file and the final `cresst_validation.yaml` before running Stage C5; the commit hash becomes part of the evidence.
