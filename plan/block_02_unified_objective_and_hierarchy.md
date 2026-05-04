# Block 02: Unified Objective and Hierarchy Table

## Goal

Turn `02_unified_objective.tex` into the conceptual spine of the paper.

After this block, a reader should understand the single optimization problem and how each estimator appears as a constraint choice.

## Main file

- `02_unified_objective.tex`

## Source material to reuse

- current generative-model and whitening sections in `02_unified_objective.tex`
- joint-channel notation already present in `03_optimal_filter.tex` and `OF_EMPCA.tex`
- hierarchy advice from `EMPCA_improvement.pdf`

## Target structure inside the section

1. Generative model
2. Gaussian ML objective
3. Whitening transform
4. Three constraints, three estimators
5. Multi-channel extension

## Required deliverables

### One orbit equation

Make one equation visually central:

- `chi^2(\hat{s}) = (x - \hat{s})^\dagger Sigma^{-1} (x - \hat{s})`

Everything else should read as a consequence of this.

### One hierarchy table

The table should compare:

- basic OF
- OF with time shift
- joint-channel OF
- EMPCA
- noise-aware linear AE

For each row include:

- constraint on reconstruction;
- learnable degrees of freedom;
- solution form;
- where it is proved or used later.

### Multi-channel bridge

Explain that stacking channels with full covariance keeps the same objective and only enlarges the vector space.

## Implementation instructions

1. Keep the current math and notation wherever it is already clean.
2. Add a short practical whitening subsection that explains diagonal PSD whitening versus full covariance whitening.
3. Add a short regularization sentence here, but defer detailed thresholds and methods to `06_convergence.tex`.
4. Make the hierarchy table the visual center of this section.
5. End the section with a transition sentence into the rank-1 OF case and the rank-`k` EMPCA case.

## Things to avoid

- Do not start proving theorems here.
- Do not expand into detector-specific PSD estimation details here.
- Do not let the multi-channel extension become a stub again.

## Done when

- the reader can state the common ML objective in one sentence;
- the hierarchy table makes the entire paper legible at a glance;
- multi-channel OF is positioned as the same principle, not as a separate method family.
