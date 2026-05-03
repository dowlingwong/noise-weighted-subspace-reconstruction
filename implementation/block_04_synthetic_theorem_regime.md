# Block 04: Synthetic Theorem-Regime Verification

## Goal

Build the cleanest empirical support for the paper's formal equivalence statements by using planted signal plus stationary Gaussian noise only.

This is where the assumptions of theorems are most directly controllable.

## Paper sections supported

- `04_empca.tex`
- `05_linear_ae.tex`
- `07_experiments.tex`
- `appendix.tex`

## Required deliverables

### Planted-data generator

Construct a generator that produces:

- clean signals from the available signal model or template family;
- amplitudes sampled over a controlled range;
- stationary Gaussian noise drawn from the measured PSD;
- optional train / test split with known ground truth.

### Verification runs

Run:

- OF on synthetic noisy traces;
- rank-1 EMPCA on the same traces;
- optionally exact whitened linear AE on the same split.

### Ground-truth comparisons

Measure:

- subspace recovery of the planted signal direction;
- amplitude bias and variance;
- equality of OF and rank-1 EMPCA reconstructed signals;
- residual-energy equality under matched weighting;
- dependence on event count and SNR.

## Implementation instructions

1. Keep this block stationary and Gaussian only.
2. Use the same PSD and whitening conventions as Block 02.
3. Sweep at least one axis that matters for finite-sample realism:
   - number of events;
   - SNR;
   - amount of timing jitter if included.
4. If timing jitter is included, label it clearly as a separate approximation study rather than the base theorem-regime run.
5. Save the planted truth alongside outputs so later diagnostics are straightforward.

## Recommended outputs

- one recovery-vs-sample-size plot;
- one amplitude-bias / variance table;
- one residual comparison plot or table;
- one short appendix note on synthetic generation assumptions.

## Things to avoid

- Do not add non-stationary drift, artifacts, or nonlinear noise here.
- Do not overcomplicate the signal family before the pure rank-1 test is complete.
- Do not present this synthetic study as a substitute for the real-data study.

## Done when

- the paper has a clean matched-assumption synthetic verification story;
- finite-sample caveats can be discussed with actual evidence rather than only words.
