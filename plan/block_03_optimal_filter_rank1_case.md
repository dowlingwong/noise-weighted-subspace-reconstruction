# Block 03: Optimal Filtering as the Rank-1 Case

## Goal

Turn `03_optimal_filter.tex` into a complete, self-sufficient rank-1 constrained-ML section.

It should not just derive OF; it should also explain why OF is the physically meaningful baseline for the rest of the paper.

## Main file

- `03_optimal_filter.tex`

## Source material to reuse

- existing OF derivations already in `03_optimal_filter.tex`
- joint-channel formula already present there
- equivalence notes in `OF_EMPCA.tex`
- experimental verification numbers later reused in `07_experiments.tex`

## Target structure

1. Rank-1 constraint and estimator
2. Time-shift OF
3. Joint-channel OF
4. Numerical verification pointer
5. Limitation: fixed-template assumption

## Required content additions

### CRB and variance

Add the missing detector-facing statistics:

- Fisher information / normalization term;
- `Var(\hat{A})`;
- connection to CRB;
- relation to energy resolution or calibrated detector metric.

### Practical assumptions

Briefly acknowledge:

- stationarity assumption;
- finite-record / PSD-estimation caveat;
- fixed-template limitation.

These should be short and controlled, not a literature dump.

## Implementation instructions

1. Preserve the current algebra for basic OF and time-shift OF.
2. Add the variance/CRB calculation immediately after the amplitude estimator so the physics relevance lands early.
3. Make joint-channel OF a full subsection with one worked explanation, not just a proposition pasted from elsewhere.
4. Add a short concluding paragraph that explains what OF cannot capture: timing manifolds, pulse-shape variation, and multi-mode structure.
5. Push long detector-specific details to the appendix.

## Things to avoid

- Do not let this section become a full review of matched filtering literature.
- Do not defer the fixed-template limitation entirely to later sections.
- Do not bury the energy-resolution connection in the discussion.

## Done when

- rank-1 OF is clearly defined as the baseline constrained estimator;
- the reader sees why OF is optimal only under the fixed-template assumption;
- the CRB / variance / resolution link is explicit enough for detector reviewers.
