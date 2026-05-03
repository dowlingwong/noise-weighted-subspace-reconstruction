# Block 07: Rank-k Quality and Noise-Aware Ablation

## Goal

Support the two practical claims that go beyond the strict theorem:

- increasing `k` helps only when there is real extra signal structure;
- using the noise model in the loss matters relative to isotropic MSE.

## Paper sections supported

- `04_empca.tex`
- `05_linear_ae.tex`
- `07_experiments.tex`

## Required deliverables

### Rank-`k` quality study

For `k = 1..K`, evaluate:

- held-out weighted residual;
- held-out `chi^2` or log-likelihood proxy;
- reconstruction gap relative to OF;
- calibrated amplitude or resolution metric;
- saturation point.

### Noise-aware vs isotropic comparison

Compare at matched capacity:

- weighted EMPCA or whitened linear AE;
- unweighted PCA or isotropic linear AE.

Evaluate under at least one colored-noise condition where the mismatch should matter.

## Implementation instructions

1. Make `k = 1` the anchor point tied to OF.
2. Use held-out metrics, not training fit, for all main comparisons.
3. If possible, include both clean theorem-regime colored noise and one practical colored-noise real-data case.
4. Use one visual centerpiece:
   - weighted reconstruction error vs `k`.
5. Tie the isotropic-vs-noise-aware result to the paper's broader design principle.

## Recommended outputs

- rank-`k` saturation curve;
- summary table of best `k` by metric;
- isotropic-vs-weighted ablation table or bar plot;
- one paragraph explaining why the loss, not the architecture, is the main difference.

## Things to avoid

- Do not compare models with different hidden preprocessing and call it a loss ablation.
- Do not let exploratory robustness runs replace the clean rank-`k` saturation plot.
- Do not oversell tiny gains without a stability check.

## Done when

- `04_empca.tex` has empirical support for the `k > 1` improvement story;
- `05_linear_ae.tex` has a meaningful noise-aware-vs-isotropic framing;
- `07_experiments.tex` has a clear rank-`k` centerpiece figure.
