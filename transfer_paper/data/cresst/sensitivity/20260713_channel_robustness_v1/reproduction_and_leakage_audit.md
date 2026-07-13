# CRESST post-freeze reproduction and leakage audit

This document reports a post-freeze sensitivity analysis. The archived pooled event bootstrap remains primary.

- Frozen reproduction passed: **True** (232 checks; 0 failures).
- Evaluation pulses: **19335** across **7** channels.
- Full-release duplicate traces: **0**.
- All within-channel pulse/noise splits chronological: **True**.
- Calibration/evaluation source-row overlap: **0**.
- Calibration/evaluation release-trace-hash overlap: **0**.
- PSD and inverse-noise metric: calibration noise only.
- PCA, EMPCA, and tied linear-AE fits: calibration pulses only.
- Single-OF template and multi-template bank: calibration pulses only.
- All comparators receive the same baseline subtraction, retained samples, and residual normalization.
- Residual units/normalization: R = (x - xhat)^T Sigma^{-1} (x - xhat) / n_features; dimensionless covariance-normalized mean-square residual per retained sample.
- Frozen rank, shrinkage, templates, shifts, exclusions, thresholds, and split fraction were not changed.

| channel | calibration pulses | evaluation pulses | calibration noise | evaluation noise | row overlap | trace-hash overlap |
|---|---:|---:|---:|---:|---:|---:|
| 33_36.0 | 12421 | 5323 | 537 | 230 | 0 | 0 |
| 33_37.0 | 12692 | 5440 | 537 | 230 | 0 | 0 |
| 35_26.0 | 3195 | 1369 | 517 | 222 | 0 | 0 |
| 35_27.0 | 3775 | 1618 | 517 | 222 | 0 | 0 |
| 35_56.0 | 3550 | 1522 | 512 | 219 | 0 | 0 |
| 35_57.0 | 4506 | 1931 | 512 | 219 | 0 | 0 |
| 35_58.0 | 4976 | 2132 | 512 | 219 | 0 | 0 |
