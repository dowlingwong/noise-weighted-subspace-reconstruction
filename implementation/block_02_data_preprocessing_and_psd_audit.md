# Block 02: Data, Preprocessing, and PSD Audit

## Goal

Build one canonical analysis interface for real K-alpha traces, PSDs, weights, templates, and whitening.

All later numerical comparisons depend on this block being stable.

## Paper sections supported

- `02_unified_objective.tex`
- `03_optimal_filter.tex`
- `04_empca.tex`
- `05_linear_ae.tex`
- `07_experiments.tex`
- `appendix.tex`

## Required deliverables

### Canonical preprocessing specification

Freeze:

- baseline subtraction window;
- trace length and crop;
- train / validation / test split;
- per-event exclusion rules;
- frequency-domain transform convention;
- one-sided PSD convention;
- weight construction convention;
- whitening convention;
- template normalization convention.

### PSD audit

Report:

- PSD source and estimation method;
- frequency-grid consistency;
- finite / nonnegative PSD check;
- any PSD floor, clipping, ridge, or shrinkage;
- one-sided to OF-weight conversion rules;
- justification for diagonal PSD treatment or full covariance treatment.

### Split manifest

Save:

- event counts;
- split definitions;
- random seed;
- date / code version;
- paths to PSD, weights, and template files.

## Implementation instructions

1. Use the same preprocessing for OF, EMPCA, and AE theorem-support runs.
2. If any method needs a different internal representation, document the representation change explicitly while preserving the same physical input.
3. Keep a clean distinction between:
   - raw traces;
   - baseline-corrected traces;
   - Fourier-domain traces;
   - whitened traces;
   - weighted real-feature representations.
4. Record whether EMPCA is run with or without smoothing for each study.
5. Create one small smoke-test subset so future changes can be checked cheaply.

## Recommended checks

- baseline mean close to zero on the chosen pretrigger region;
- FFT length and PSD bin count match exactly;
- OF weights and whitening weights agree algebraically where expected;
- no train/test leakage through shared template estimation;
- identical event order and masking across compared methods.

## Things to avoid

- Do not let notebooks silently redefine baseline windows or FFT conventions.
- Do not compare methods that used different PSDs or different splits and call it an equivalence check.
- Do not hide regularization choices.

## Done when

- one preprocessing note can be copied into `07_experiments.tex` and `appendix.tex`;
- all later blocks can import the same split, PSD, and weight definitions without reinterpretation.
