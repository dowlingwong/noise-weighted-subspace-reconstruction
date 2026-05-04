# Block 07: Experiments, Verification, and Tables

## Goal

Turn `07_experiments.tex` into a benchmark section that verifies the paper's claims instead of merely collecting figures.

## Main file

- `07_experiments.tex`

Supporting sources:

- `noise.tex`
- figures already in `figures/`
- numerical tables already referenced in `05_linear_ae.tex` and the old equivalence notes

## Target structure

1. Benchmark domain
2. Evaluation metrics
3. Verification of equivalence theorems
4. Reconstruction quality versus rank `k`
5. Noise-aware loss versus isotropic MSE
6. Convergence behavior

## Required deliverables

### Fill the empty tables

Do not leave placeholder tables in the paper.

At minimum fill:

- Study A summary table;
- Study A crossover table;
- Study B scenario table;
- theorem-verification summary table;
- rank-`k` summary table if needed.

### Characterize the K-alpha data

Add enough information for a reader to interpret the real-data verification:

- what the data are;
- how many events;
- what preprocessing was applied;
- what metrics are being compared.

### Core experiments

The section should visibly support four claims:

1. rank-1 EMPCA matches OF under matched assumptions;
2. increasing `k` improves residual fit until saturation;
3. noise-aware loss outperforms isotropic MSE where the noise model matters;
4. the optimization is numerically well-behaved under the stated recipe.

## Structured-noise module usage

Use the nonlinear / nonstationary / correlated / artifact noise module here as a robustness stress test, not as the main theorem-verification dataset.

Recommended placement:

- brief mention in benchmark-domain subsection;
- direct use in the noise-aware-vs-isotropic ablation;
- optional robustness panel in rank-`k` saturation study.

## Implementation instructions

1. Keep the benchmark-domain subsection short and push long detector detail to the appendix.
2. Tie every metric to a theorem or design claim.
3. Put theorem-verification evidence before the more exploratory studies.
4. Use the structured-noise module to show practical relevance without undermining the clean Gaussian-baseline theorem tests.
5. Make one plot the visual centerpiece: whitened reconstruction error versus rank `k`.

## Things to avoid

- Do not let this section turn into a detector-noise survey.
- Do not mix proof language with empirical evidence.
- Do not let the structured-noise results replace the clean baseline equivalence tests.

## Done when

- every table referenced in the text is populated;
- the experiments directly support the paper's stated claims;
- the structured-noise module appears as a useful robustness stress test, not as a theory-breaking distraction.
