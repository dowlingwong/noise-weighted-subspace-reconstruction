# Block 03: OF Baseline, CRB, and Real Rank-1 Verification

## Goal

Establish OF as the rank-1 baseline both statistically and empirically on real K-alpha data.

This block combines three things that need to land together in the paper:

- the OF estimator itself;
- its variance / CRB / resolution meaning;
- the real-data rank-1 OF vs EMPCA verification.

## Paper sections supported

- `03_optimal_filter.tex`
- `04_empca.tex`
- `07_experiments.tex`

## Required deliverables

### OF statistical backbone

Compute and document:

- OF normalization term;
- Fisher information;
- `Var(A_hat)`;
- CRB comparison;
- propagation to energy-resolution or calibrated amplitude units.

### Real-data strict rank-1 comparison

On held-out K-alpha data under matched preprocessing:

- train rank-1 EMPCA without smoothing;
- phase / sign align the learned basis to the template direction;
- compare OF and EMPCA amplitudes;
- compare reconstructed traces;
- compare weighted residual energies.

### Core metrics

At minimum report:

- weighted cosine or principal angle;
- maximum or median amplitude discrepancy after normalization matching;
- residual KS statistic and p-value;
- mean and spread of weighted residual energy;
- calibrated resolution metric if available.

## Implementation instructions

1. Keep the strict theorem-support run separate from any smoothed practical EMPCA run.
2. Use matched PSD weighting everywhere.
3. Make the gauge-fixing rule explicit before coefficient comparison.
4. If calibration to eV is not yet available, define a stable intermediate calibrated amplitude unit and note the missing final conversion.
5. If multichannel data exist, add a second OF baseline for joint-channel correlated OF; otherwise mark it as a deferred sub-block.

## Optional extension

If time permits in the same block:

- compare single-channel OF and joint-channel OF on the same split;
- quantify any gain in variance or reconstruction quality.

## Things to avoid

- Do not compare raw rank-1 EMPCA coefficients to OF amplitudes before fixing normalization.
- Do not mix centered and uncentered variants inside the strict rank-1 theorem check.
- Do not bury the CRB / resolution result in the appendix.

## Done when

- `03_optimal_filter.tex` has explicit variance / CRB support;
- `04_empca.tex` has a real-data rank-1 verification table;
- `07_experiments.tex` can lead with a clean theorem-verification result.
