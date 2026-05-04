# Block 06: Convergence, Rank Selection, and Regularization

## Goal

Make `06_convergence.tex` answer the practical questions reviewers will ask after reading the theory:

- does it converge;
- how do you initialize it;
- how do you choose `k`;
- what happens when whitening or coefficient solves are ill-conditioned.

## Main file

- `06_convergence.tex`

## Source material to reuse

- current stubs in `06_convergence.tex`
- convergence-related comments in `EMPCA_improvement.pdf`
- numerical behavior notes already mentioned in `07_experiments.tex`

## Target structure

1. Convergence guarantee
2. Practical initialization
3. Model order selection
4. Conditioning and regularization

## Required content

### Convergence claim

State only what you can support:

- monotone objective improvement per EM iteration under the stated setup;
- local/global caveat where necessary;
- stationarity and Gaussian assumptions as part of the practical setup.

### Initialization

Recommend one practical default:

- whitened SVD warm start or whitened subspace warm start;
- convergence criterion based on objective change or principal angle.

### Rank selection

Use one primary rule and one secondary rule:

- primary: weighted residual / `chi^2` improvement saturation;
- secondary: scree or information-criterion style view.

### Conditioning

Include:

- ill-conditioned covariance handling;
- ridge or shrinkage on whitening / normal equations;
- guidance on when to regularize.

## Implementation instructions

1. Keep the section practical and short.
2. Put rigorous theorem-level details in the appendix if they become too long.
3. Add direct forward references to the convergence and rank-selection plots in `07_experiments.tex`.
4. Make the model-order subsection operational: tell the reader what they would actually do.

## Things to avoid

- Do not promise a sweeping global-convergence theorem unless you really write it.
- Do not hide rank selection in the appendix; it is central for reviewers.
- Do not turn this section into a broad survey of nonconvex optimization.

## Done when

- the reviewer can see a credible training recipe;
- the choice of `k` no longer looks arbitrary;
- the paper has an explicit answer to conditioning and whitening edge cases.
