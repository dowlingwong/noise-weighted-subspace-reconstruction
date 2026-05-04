# Block 01: Positioning, Abstract, and Introduction

## Goal

Write the paper opening so the reader immediately understands:

- what problem is shared across OF, EMPCA, and AE;
- why this is an MLST paper instead of a detector-specific note;
- what theorems and experiments the paper will actually deliver;
- why nonlinear models belong to Paper 2, not this manuscript.

## Main files

- `00_abstract.tex`
- `01_intro.tex`

Optional later restructure:

- add a dedicated related-work section after `01_intro.tex`

## Source material to reuse

- `EMPCA_improvement.pdf`, especially the improved structure pages
- current comments and subsection headings in `01_intro.tex`
- existing paper title and current section names in `main.tex`

## Target structure

### Abstract

Write four compact beats:

1. shared reconstruction problem under known/estimated noise covariance;
2. theoretical unification of OF, EMPCA, and noise-aware linear AE;
3. key proofs and hierarchy;
4. empirical verification and setup for Paper 2.

### Introduction

Use this structure:

1. shared physics problem first;
2. one paragraph each for signal processing, statistics, and ML communities;
3. the core gap across these communities;
4. numbered contribution list tied to later theorem labels;
5. scope paragraph that explicitly excludes nonlinear trigger models from Paper 1;
6. paper organization.

## Content that must appear

- explicit statement that the paper is about constrained solutions of one Gaussian ML objective;
- contribution bullets that map to the actual theorem sequence;
- one sentence that says the linear AE section is not a side note but the endpoint of the linear theory;
- one short paragraph that says Paper 2 will relax the linear encoder while preserving the noise-aware training principle.

## Implementation instructions

1. Draft the abstract only after the hierarchy table and theorem list are stable.
2. In the intro, open with the measurement problem, not with acronyms.
3. Give each community one paragraph and end each paragraph with the same gap.
4. Keep detector specifics light in the introduction; heavy context belongs in the appendix and benchmark section.
5. If you do not want to renumber files yet, place related-work synthesis inside the intro after the contributions.
6. If you later create a separate Related Work section, move those paragraphs almost verbatim.

## Things to avoid

- Do not discuss ResNet, transformer, RL trigger, or SBI in detail here.
- Do not lead with LAMCAL implementation detail.
- Do not promise robustness claims that the experiments do not verify.

## Done when

- the abstract reads like a finished paper abstract, not a project summary;
- the intro states the gap cleanly for all three communities;
- the contribution bullets clearly point to specific theorem-level outputs;
- the nonlinear story is explicitly postponed to Paper 2.
