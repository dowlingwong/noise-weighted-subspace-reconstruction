# Block 01: Claim Mapping and Execution Rules

## Goal

Freeze the paper's claim structure before running more computations.

This block prevents a common failure mode: collecting many plots while never proving that a given plot supports a specific theorem or section.

## Paper sections supported

- `02_unified_objective.tex`
- `03_optimal_filter.tex`
- `04_empca.tex`
- `05_linear_ae.tex`
- `06_convergence.tex`
- `07_experiments.tex`
- `08_discussion.tex`
- `appendix.tex`

## Required deliverables

### Claim-to-metric map

Create one internal table with columns:

- manuscript claim;
- section / theorem / proposition label;
- support regime;
- exact metric(s);
- acceptance threshold or qualitative criterion;
- output artifact path.

### Regime labels

Every experiment must be labeled as one of:

- `theorem-support`
- `real-support`
- `robustness-support`

### Canonical naming rules

Freeze naming for:

- preprocessing variants;
- PSD / weight variants;
- training variants;
- rank `k`;
- seeds / splits;
- gauge / normalization convention.

## Implementation instructions

1. Start from the theorem list and section goals already described in the parent `plan/` markdown files.
2. For each claim, decide whether it needs exact support, empirical support, or only positioning text.
3. Refuse vague labels like "looks similar" or "good agreement." Replace them with explicit metrics.
4. Decide early which claims are in-scope for Paper 1 and which must be deferred to Paper 2.
5. Mark all structured-noise studies as robustness-only unless there is a very specific reason not to.

## Suggested claim groups

- one-objective hierarchy support;
- OF estimator / CRB / energy-resolution support;
- rank-1 OF vs EMPCA equivalence support;
- rank-`k` improvement support;
- EMPCA vs linear-AE bridge support;
- convergence / initialization / rank-selection support;
- noise-aware-vs-isotropic support;
- PC interpretation support;
- robustness beyond theorem assumptions.

## Things to avoid

- Do not let one artifact serve multiple incompatible claims.
- Do not use out-of-model noise to justify theorem statements.
- Do not delay threshold choices until after seeing the results.

## Done when

- every planned experiment has a claim owner;
- support regimes are frozen;
- later blocks can point to explicit metrics instead of general intentions.
