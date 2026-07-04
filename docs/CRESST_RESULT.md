# CRESST Result And Interpretation

Last consolidated: 2026-07-04.

## Bottom Line

The predeclared CRESST-A validation run (Stage C5, commit `9c9f696ff6bc`,
`configs/cresst/cresst_validation.yaml`) **passed** its predeclared
acceptance gate on the public DMDC CRESST-II/III pulse-shape test split.
This is currently the first positive public real-data validation result for
the linear noise-geometry hierarchy (GWOSC, by contrast, is a negative
real-noise stress test; see `docs/GWOSC_RESULT.md`).

The gate was evaluated exactly as predeclared in
`docs/CRESST_PREDECLARED_GATES.md`: no threshold, grid, or tolerance was
adjusted after this run, and this report was written directly from its
output.

## What Succeeded

- All 7 detector channels in the test split were used (none skipped for
  insufficient data after the per-channel time-disjoint split).
- `n_duplicate_traces: 0` -- no duplicate-trace leakage in the raw split.
- EMPCA (rank 6) beat single-template OF in every one of the 7 channels
  individually, and in the pooled paired-bootstrap comparison
  (`of_residual - empca_residual` mean `1012.3`, 95% CI `[927.9, 1097.6]`,
  excludes zero).
- EMPCA was competitive with the multi-template OF bank (within the
  predeclared 1.1x tolerance) in every channel individually and pooled
  (`1.1 x multi_of_residual - empca_residual` mean `352.2`, 95% CI
  `[272.4, 424.0]`, excludes zero).
- All 5 negative controls degraded the result at the pooled level, each
  with a 95% CI excluding zero in the required direction: `wrong_psd`,
  `permuted_time_bins`, `random_weighted_basis`, `noise_only_train`,
  `mismatched_of_template`.
- The one real-release data gap found during Stage C3/C4 (channel
  `(35, 56)` has no row in `resolutions_test.csv`) was handled by the
  predeclared CRB-formula fallback, not silently or after the fact.

## What Did Not Cleanly Succeed

- **One negative control, one channel**: `wrong_psd` (calibrate with an
  identity metric instead of the true covariance) did not degrade the
  result on channel `35_58.0` alone (`mean_difference = -0.005`, CI
  `[-0.010, 0.0001]`, effectively zero) -- a genuine sign reversal against
  the pooled direction, and the only one across all 5 controls x 7
  channels. `wrong_psd` was also the weakest-margin control pooled (mean
  difference `1.57` vs. hundreds-to-tens-of-thousands for the other 4), so
  this is consistent with that channel's true noise being closer to
  white/uncorrelated than the others -- a plausible physical explanation,
  not an alarm, but reported because the predeclared aggregation commits
  to reporting sign reversals rather than only the pooled verdict.
- **Trigger efficiency was largely inconclusive within the predeclared
  amplitude grid.** At the primary 5% FPR point, the direct-injection
  efficiency curve stayed within noise of the ~5% background floor for 6 of
  7 channels across the full `[0, 10] x sigma_OF` grid (no 50%/90%
  crossing found, i.e. `NaN` thresholds); only channel `35_27.0` crossed
  50% (at the top of the grid) and none crossed 90%. This does not
  contradict the primary linear-hierarchy gate above, which is based on
  weighted-residual comparisons among established methods (OF/PCA/EMPCA/
  multi-template OF), not on the trigger-efficiency curves -- but it means
  the amplitude grid, anchored to `[0,1,2,3,4,5,6,8,10] x` each channel's
  published/CRB-estimated resolution, undersampled the true efficiency-rise
  region for this real dataset. Reported as a limitation, not adjusted:
  the grid was frozen in `docs/CRESST_PREDECLARED_GATES.md` before this run
  and Hard Rule 2 forbids retuning it now that a result is visible.

## Key Numbers

Pooled paired-bootstrap gate (`bootstrap_reps: 300`, `n_events: 19,335`
pooled across all 7 channels):

| Comparison | Mean difference | 95% CI | Excludes zero? |
| --- | ---: | --- | :---: |
| OF residual − EMPCA residual | `1012.27` | `[927.90, 1097.61]` | yes |
| `1.1 x` multi-template-OF residual − EMPCA residual | `352.17` | `[272.40, 424.01]` | yes |
| `wrong_psd` control − EMPCA residual | `1.57` | `[0.97, 2.08]` | yes |
| `permuted_time_bins` control − EMPCA residual | `59812.72` | `[53914.03, 65079.46]` | yes |
| `random_weighted_basis` control − EMPCA residual | `4013.47` | `[3694.77, 4304.39]` | yes |
| `noise_only_train` control − EMPCA residual | `489.40` | `[405.58, 563.90]` | yes |
| `mismatched_of_template` control − OF residual | `6911.82` | `[6566.70, 7241.21]` | yes |

Per-channel weighted residuals (lower is better; all in the declared noise
metric, native 512-feature resolution):

| Channel | n_train | n_test | n_bg | sigma_OF source | PCA | EMPCA | OF | Multi-OF |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 33_36 | 12421 | 5323 | 230 | published | 41.04 | 35.18 | 2892.04 | 833.97 |
| 33_37 | 12692 | 5440 | 230 | published | 7.94 | 7.03 | 68.41 | 35.32 |
| 35_26 | 3195 | 1369 | 222 | published | 12.66 | 4.81 | 255.28 | 89.76 |
| 35_27 | 3775 | 1618 | 222 | published | 11.85 | 10.30 | 240.46 | 54.43 |
| 35_56 | 3550 | 1522 | 219 | CRB fallback | 2.34 | 1.75 | 62.08 | 15.78 |
| 35_57 | 4506 | 1931 | 219 | published | 41.62 | 31.95 | 1641.06 | 807.46 |
| 35_58 | 4976 | 2132 | 219 | published | 2.50 | 2.46 | 57.27 | 25.37 |

EMPCA beats OF and beats-or-ties multi-template OF in every single channel,
not only pooled.

## Correct Interpretation

EMPCA at rank 6, under an empirically estimated per-channel noise metric,
reconstructs real CRESST pulses better than a single-template optimal
filter and remains competitive with a 9-cell x 5-shift multi-template OF
bank -- on held-out, time-later data from the same channel, not a
cross-channel transfer test. All 5 negative controls confirm the gain
comes from correct noise weighting plus real pulse-shape structure, not
from dimensionality reduction alone or from an artifact of the pipeline,
with one single-channel exception in the weakest-margin control
(`wrong_psd` on `35_58.0`) that is physically plausible and explicitly
reported rather than smoothed over.

The trigger-efficiency result is a separate, secondary detector-facing
metric and is inconclusive as measured -- it neither supports nor
contradicts the primary linear-hierarchy claim, but it should not be
described as validated detection sensitivity.

## Manuscript-Safe Text

Use this as the base paragraph:

```text
As a public real-data validation, we applied the noise-geometry pipeline to
the CRESST-II/III pulse-shape release (arXiv:2508.03078), splitting each of
the 7 test-release detector channels chronologically into calibration and
evaluation segments. EMPCA at the predeclared rank (k=6) beat a
single-template optimal filter and remained competitive with a
multi-template optimal-filter bank, judged by a paired bootstrap over held-
out events pooled across channels (95% CI excluding zero in both cases).
All five predeclared negative controls (wrong noise covariance, permuted
time bins, a random subspace, training on noise instead of signal, and a
mismatched template) degraded the result as required at the pooled level,
with one channel-level exception in the weakest-margin control that is
consistent with that channel's noise being close to white. A secondary
detector-facing metric, trigger efficiency versus injected amplitude at
fixed background false-positive rate, was inconclusive within the
predeclared amplitude grid for most channels and is reported as a
limitation, not as validated detection sensitivity.
```

## Wording To Avoid

- Do not claim the trigger-efficiency curves demonstrate validated
  detection sensitivity -- they were largely inconclusive at the
  predeclared grid.
- Do not claim all negative controls degraded with a clean margin in every
  single channel -- one channel-level exception exists in `wrong_psd` and
  is reported above.
- Do not claim this validates cross-channel/cross-detector transfer -- the
  design is explicitly per-channel; nothing here tests whether one
  channel's noise model works for another.
- Do not quote Stage C2 smoke metrics as part of this result (see the
  smoke-contamination acknowledgment in `docs/CRESST_PREDECLARED_GATES.md`).

## Evidence Files

Primary records:

- `results/metrics/p2_cresst_validation_seed22.json`
- `results/metrics/p2_cresst_validation_seed22.config.yaml`
- `results/metrics/p2_cresst_validation_seed22.log`
- `evidence/cresst/20260704T013758Z_9c9f696ff6bc/`
- `docs/CRESST_PREDECLARED_GATES.md` (frozen predeclaration, commit `9c9f696ff6bc`)

Derived summaries:

- `transfer_paper/data/derived/cresst_per_channel_summary.csv`
- `transfer_paper/data/derived/cresst_pooled_summary.csv`
- `transfer_paper/data/cresst/current/` (synchronized run copy)

## Diagnostic Framing

- Residual whitening ratio and detection score were computed per channel
  (`per_channel.<channel>.detector_metrics.residual_whitening_ratio`) but
  are not summarized above; consult the run JSON directly for the
  frequency-resolved view.
- Sign-reversal reporting: the predeclared aggregation checks every
  channel's own sign against the pooled sign for every gate component;
  `pooled.*_sign_reversed_channels` in the run record is the authoritative
  list (only `wrong_psd` / `35_58.0` is non-empty).

## If CRESST Work Is Continued

Do not modify this completed run to obtain a different result. Any future
CRESST work should be a new predeclared stage, for example:

- a wider or denser amplitude grid (more `sigma_OF` multiples, more grid
  points) to resolve the trigger-efficiency 50%/90% crossings that were
  `NaN` here;
- an investigation (diagnostic only, not a re-run of this gate) of why
  channel `35_58.0`'s noise appears close to white;
- extending to `X_train.npy`/the full release beyond the test split, with
  its own frozen predeclaration;
- a genuine cross-channel generalization study, explicitly framed as
  testing domain transfer rather than the linear-hierarchy claim tested
  here.
