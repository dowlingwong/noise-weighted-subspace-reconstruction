# Block 05: Linear-AE Bridge and Gauge Handling

## Goal

Turn the EMPCA / linear-AE equivalence from a notebook result into a stable paper-ready computation, and resolve the coefficient / gauge issue cleanly.

## Paper sections supported

- `04_empca.tex`
- `05_linear_ae.tex`
- `07_experiments.tex`

## Required deliverables

### Gauge and normalization note

Define explicitly:

- sign / phase convention for rank-1;
- basis-rotation convention for rank-`k`;
- how amplitudes or latent coordinates are compared across bases;
- when reconstruction equality matters more than coefficient equality.

### EMPCA vs exact linear-AE equivalence

On the canonical split:

- train or solve exact tied linear AE in whitened complex space;
- compare with EMPCA at the same rank `k`;
- use principal angles, residual statistics, and mean reconstruction gap.

### Native weighted vs whitened comparison

Check whether:

- exact whitened formulation;
- native weighted formulation;
- any real-feature proxy representation

all behave consistently enough for the paper's stated claim.

## Implementation instructions

1. Treat reconstruction equality and latent-coordinate equality as separate diagnostics.
2. Put the principal-angle comparison first; it is the cleanest subspace-level metric.
3. Keep the exact complex-whitened comparison as the primary bridge theorem support.
4. If a proxy real-feature representation underperforms, present it as a representation mismatch diagnostic, not a contradiction of the theorem.
5. Reuse the same split and preprocessing as Blocks 02-04.

## Recommended outputs

- one subspace-angle table;
- one residual-distribution comparison;
- one short note on gauge freedom and what is physically identifiable;
- one ablation summary sentence for `05_linear_ae.tex`.

## Things to avoid

- Do not compare latent coordinates from rotated rank-`k` bases as if they were uniquely identifiable.
- Do not overstate equivalence beyond the shared whitened objective.
- Do not let this section drift into nonlinear AE speculation.

## Done when

- `05_linear_ae.tex` has explicit theorem support;
- `04_empca.tex` has a clean gauge discussion;
- the AE section reads as a consequence of the same objective, not a side branch.
