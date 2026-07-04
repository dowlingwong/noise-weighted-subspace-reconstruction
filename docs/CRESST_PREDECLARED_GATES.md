# CRESST Predeclared Validation Gates

Status: **DRAFT v2 — pending review, not yet committed, Stage C5 not yet run.**

This supersedes the v1 draft after a design review that found the v1 split
(random cross-channel group assignment) contradicted Paper 1's own premise.
This file freezes every threshold, grid, rank, FPR point, split definition,
and acceptance criterion for the CRESST-A predeclared validation run
(`configs/cresst/cresst_validation.yaml`), per Stage C4 of
`docs/CRESST_REMOTE_AGENT_RUNBOOK.md` and the CRESST Validation Protocol in
`docs/EXPERIMENT_PROTOCOLS.md`.

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

**v2**: split *within* each channel, chronologically. Sort each channel's
own traces by `hours`; the earliest `channel_calibration_fraction` (0.7) is
calibration (Sigma, PSD, template bank, subspace all fit here), the later
remainder is evaluation (test pulses, background-FPR noise, injection
substrate). This is still leakage-safe (calibration and evaluation traces
are disjoint, PSD is independent of the test residuals it's later scored
against) and turns the 7 test-release channels into 7 quasi-independent
replications instead of one fragile 5-vs-2 channel split. The old "7-group
granularity" caveat from v1 no longer applies -- every channel contributes
its own held-out evaluation, not a coin-flip group assignment.

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
- FPR points: `[0.001, 0.01, 0.05]`, numbers unchanged, but **0.1% is
  demoted to diagnostic** (see below) -- the one exception to "keep the FPR
  points" granted for a concrete statistical reason.
  `fpr_diagnostic_only: [0.001]`.
- Efficiency thresholds: `[0.5, 0.9]`, unchanged.
- `bootstrap_reps: 300`, unchanged, now used both for the amplitude
  bias/RMSE CIs (as before) and for every paired-difference gate CI (new,
  see Acceptance gate below).

### 0.1% FPR demoted to diagnostic

Each channel's held-out (evaluation-segment) noise pool is on the order of a
few hundred to ~1,500 traces (real per-channel background counts, 0.3 x
each channel's noise total). A 0.1% threshold is then set by the top 1-2
order statistics of that pool and carries large threshold uncertainty. 1%
and 5% remain the reportable, gate-relevant points; 0.1% is still computed
and recorded (background threshold, efficiency curve) but is explicitly
labeled diagnostic, not used in any pass/fail decision.

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

## Other frozen values (unchanged from the prior declaration)

- `covariance_shrinkage: 0.1`
- `baseline_fraction: 0.2`
- `max_features: 512` (native downsampled resolution)
- `split: test`; `channel_columns: [run, channel]`;
  `channel_calibration_fraction: 0.7`; `channel_time_column: hours`

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

- [ ] Per-channel, time-disjoint split (chronological by `hours`, 0.7 calibration fraction, seed 22) replacing the flawed cross-channel group split -- reviewed and approved by design critique.
- [ ] Rank `k=6` and FPR points `[0.001, 0.01, 0.05]` -- carried over unchanged.
- [ ] 0.1% FPR demoted to diagnostic-only (the one FPR exception) -- reviewed.
- [ ] Template-bank grid (terciles x terciles x 5 shifts, `min_cell_pulses=20`) -- approved as-is.
- [ ] Amplitude grid anchored to per-channel published sigma_OF (`[0,1,2,3,4,5,6,8,10] x sigma_OF`), with the CRB-formula fallback for the one channel `(35,56)` missing a published resolution -- reviewed.
- [ ] 5 negative controls (including `mismatched_of_template`) -- approved as-is.
- [ ] `multi_template_of_competitive_tolerance: 1.1`, now applied via paired-bootstrap CI -- approved as-is (CI-aware).
- [ ] Paired-bootstrap gate criteria (CI excludes zero, not point estimate) with pooled-across-channels aggregation and per-channel sign-reversal reporting -- reviewed.
- [ ] Injection template = per-channel single-OF template from that channel's calibration segment -- confirmed explicit.
- [ ] Stage C2 smoke contamination acknowledgment above -- read.
- [ ] `n_injections_per_amplitude=200`, `bootstrap_reps=300`, `covariance_shrinkage=0.1`, `max_features=512` -- approved as-is.
- [ ] Commit this file and the final `cresst_validation.yaml` before running Stage C5; the commit hash becomes part of the evidence.
