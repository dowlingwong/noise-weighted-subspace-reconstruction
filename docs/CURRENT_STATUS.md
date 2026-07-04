# Current Status

Last consolidated: 2026-06-29.

## Central Claim

For Gaussian detector noise, the statistically aligned reconstruction geometry
is set by the inverse noise covariance:

```text
(x - x_hat)^H Sigma^{-1} (x - x_hat)
```

Under this metric, fixed-template optimal filtering, rank-k EMPCA, and tied
linear autoencoders with a noise-weighted loss are constrained maximum-
likelihood projections. Ordinary MSE is the corresponding likelihood only for
white isotropic noise.

Paper 1 is scoped to linear reconstruction, optimal filtering, weighted
subspaces, covariance/PSD estimation, and residual calibration. Nonlinear
architectures and Paper 2 material are outside the Paper 1 validation claim
set.

## Validation Summary

| Area | Current state | Paper use |
| --- | --- | --- |
| S1-S9 controlled synthetic gates | verified positive | Results and Methods |
| Stage 0 remote reproducibility | verified positive | Methods/reproducibility |
| GWOSC data integrity and official coverage | verified positive | Methods |
| GWOSC PSD comparison with GWpy | verified positive | Methods/diagnostic Results |
| GWOSC shared-FIR implementation identity | verified positive | Results/diagnostic Methods |
| GWOSC global-PSD null calibration | verified negative | Results, Discussion, Limitations |
| GWOSC time-local PSD calibration | verified negative | Results, Discussion, Limitations |
| GW150914 event significance | not validated | Diagnostics only, if mentioned |
| SNR-eight injection sensitivity on GWOSC | not validated | Diagnostics only, if mentioned |
| CRESST-A public-data validation (linear hierarchy) | verified positive | Results and Methods |
| CRESST trigger-efficiency detector metric | inconclusive | Limitations, if mentioned |

## What Is Supported

- The implementation reproduces the intended linear likelihood geometry in
  controlled synthetic settings.
- The OF/CRB, OF/EMPCA, EMPCA/linear-AE, white-noise control, metric reversal,
  timing-rank, covariance convergence, residual calibration, and multichannel
  covariance synthetic gates have archived positive evidence.
- The remote Stage 0 workflow can run from a clean Linux checkout with archived
  command logs and environment records.
- The GWOSC PSD estimator agrees with GWpy on identical calibration windows.
- The shared-FIR follow-up shows that explicit FFT convolution and GWpy
  convolution agree when both paths use identical FIR coefficients and score
  normalization.
- CRESST-A (public DMDC CRESST-II/III pulse-shape release, test split): EMPCA
  at the predeclared rank (k=6) beat single-template OF and was competitive
  with multi-template OF, and all 5 predeclared negative controls degraded,
  by a pooled paired-bootstrap 95% CI across all 7 detector channels (each
  split chronologically into its own calibration/evaluation segments). See
  `docs/CRESST_RESULT.md`.

## What Is Not Supported

- The current evidence does not support calibrated GW150914 significance.
- The current evidence does not support validated nominal SNR-eight injection
  sensitivity on GWOSC.
- The current evidence does not support a general claim that the method is
  calibrated on GWOSC data.
- The current evidence does not show that a 64-second local PSD model fixes the
  GWOSC real-noise calibration failure.
- The shared-FIR result does not prove that the original PSD-domain GLS
  statistic and a finite-duration FIR statistic are mathematically equivalent.
- The CRESST-A result does not support a cross-channel/cross-detector
  transfer claim; the design is explicitly per-channel.
- The CRESST-A trigger-efficiency detector metric (versus injected amplitude
  at fixed background FPR) was inconclusive within the predeclared amplitude
  grid for most channels and does not support a validated detection-
  sensitivity claim.

## Current GWOSC Conclusion

GWOSC is a completed negative public real-noise stress test for the current
paper revision. It is not a positive validation dataset.

The correct interpretation is:

```text
GWOSC shows the boundary of the present validation: real-noise calibration
requires more than a correct PSD implementation and a predeclared local PSD
model.
```

Keep GWOSC as a limitation and boundary result unless a new predeclared Stage
3b is designed and run.

## Current CRESST Conclusion

CRESST-A is a completed positive public real-data validation of the linear
noise-geometry hierarchy for the current paper revision, with two reported
limitations (see `docs/CRESST_RESULT.md`): a single-channel exception in the
weakest-margin negative control, and an inconclusive trigger-efficiency
detector metric at the predeclared amplitude grid.

The correct interpretation is:

```text
CRESST-A shows that, on real cryogenic detector pulses split chronologically
within each channel, the noise-weighted linear hierarchy (EMPCA vs.
single-/multi-template OF) holds with statistically supported margins. It
does not show cross-channel transfer, and it does not show validated
detection sensitivity from the trigger-efficiency metric as measured.
```

## Next Steps

1. Revise the manuscript so the positive claims rest on theory, controlled
   synthetic validation, and the CRESST-A public real-data result.
2. Include GWOSC as a negative public real-noise stress test, not as calibrated
   detection evidence.
3. ~~Move positive public real-data validation to CRESST or another
   detector-pulse dataset~~ -- done; see `docs/CRESST_RESULT.md`. If a wider
   detection-sensitivity claim is wanted, that requires a new predeclared
   stage with a revised amplitude grid (see "If CRESST Work Is Continued" in
   that document).
4. Do not tune the completed GWOSC experiment by changing thresholds, PSD
   radii, windows, quality cuts, or exclusions after seeing the result. The
   same applies to the completed CRESST-A run.
5. If GWOSC is continued, write a new predeclared Stage 3b protocol first.

## Important Evidence Locations

- `transfer_paper/data/derived/claim_status.csv`
- `docs/CRESST_RESULT.md`, `docs/CRESST_PREDECLARED_GATES.md`
- `transfer_paper/data/derived/cresst_per_channel_summary.csv`,
  `transfer_paper/data/derived/cresst_pooled_summary.csv`
- `evidence/cresst/20260704T013758Z_9c9f696ff6bc/`
- `transfer_paper/data/derived/paper_implications.csv`
- `transfer_paper/MANUSCRIPT_EVIDENCE_MAP.md`
- `transfer_paper/data/gwosc/followup/filter_equivalence.json`
- `transfer_paper/data/gwosc/followup/time_local_noise.json`
- `transfer_paper/data/derived/gwosc_filter_equivalence_summary.csv`
- `transfer_paper/data/derived/gwosc_time_local_psd_summary.csv`
