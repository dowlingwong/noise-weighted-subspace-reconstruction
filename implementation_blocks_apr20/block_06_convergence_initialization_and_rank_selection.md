# Block 06: Convergence, Initialization, and Rank Selection

## Goal

Provide a credible practical training recipe for EMPCA and related weighted subspace models.

This block should answer the reviewer questions that come immediately after the theory:

- does it converge;
- from what initialization;
- how do we choose `k`;
- what regularization is needed.

## Paper sections supported

- `06_convergence.tex`
- `07_experiments.tex`
- `appendix.tex`

## Required deliverables

### Convergence traces

For at least one canonical dataset, record:

- objective or weighted residual per iteration;
- subspace change per iteration;
- stopping criterion behavior;
- effect of smoothing vs no smoothing.

### Initialization study

Compare at least:

- random initialization;
- whitened SVD or template-informed warm start.

### Rank-selection study

Produce:

- held-out weighted loss vs `k`;
- resolution or calibrated performance vs `k`;
- subspace stability vs `k`;
- optional scree-style summary.

### Conditioning study

Probe:

- PSD flooring or shrinkage;
- ridge on normal equations;
- failure modes from near-degenerate components.

## Implementation instructions

1. Distinguish the theoretical EM-style loop from the practical smoothed implementation.
2. Only claim monotone decrease for the loop that truly has it.
3. Use multiple seeds so the stability claim is empirical, not anecdotal.
4. Choose one primary operational rule for `k`, preferably held-out weighted residual saturation.
5. Keep the regularization guidance practical: what to do, when to do it, and what artifact shows the need.

## Recommended outputs

- convergence plot for no-smoothing and smoothed variants;
- seed-stability principal-angle summary;
- rank-`k` selection plot;
- short regularization recommendation table.

## Things to avoid

- Do not promise a global-convergence theorem without writing one.
- Do not hide `k` selection in an appendix-only comment.
- Do not mix robustness-noise runs into the core convergence study.

## Done when

- `06_convergence.tex` can give a concrete recipe;
- `07_experiments.tex` has convergence and rank-selection evidence;
- the practical choice of `k` no longer looks arbitrary.
