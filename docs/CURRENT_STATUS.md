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
| CRESST public-data validation | not completed | Future/next validation |

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

## Next Steps

1. Revise the manuscript so the positive claims rest on theory and controlled
   synthetic validation.
2. Include GWOSC as a negative public real-noise stress test, not as calibrated
   detection evidence.
3. Move positive public real-data validation to CRESST or another detector-pulse
   dataset if the paper requires a positive real-data result.
4. Do not tune the completed GWOSC experiment by changing thresholds, PSD
   radii, windows, quality cuts, or exclusions after seeing the result.
5. If GWOSC is continued, write a new predeclared Stage 3b protocol first.

## Important Evidence Locations

- `transfer_paper/data/derived/claim_status.csv`
- `transfer_paper/data/derived/paper_implications.csv`
- `transfer_paper/MANUSCRIPT_EVIDENCE_MAP.md`
- `transfer_paper/data/gwosc/followup/filter_equivalence.json`
- `transfer_paper/data/gwosc/followup/time_local_noise.json`
- `transfer_paper/data/derived/gwosc_filter_equivalence_summary.csv`
- `transfer_paper/data/derived/gwosc_time_local_psd_summary.csv`
