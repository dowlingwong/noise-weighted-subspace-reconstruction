# Block 09: Structured-Noise and Multichannel Robustness

## Goal

Use the structured-noise module and, if available, multichannel data to show practical behavior beyond the clean theorem assumptions without corrupting the core theorem story.

## Paper sections supported

- `07_experiments.tex`
- `08_discussion.tex`
- `appendix.tex`

## Required deliverables

### Structured-noise robustness suite

Use the noise module to create controlled perturbations such as:

- piecewise-stationary PSD changes;
- drift;
- spectral lines;
- glitches;
- sparse impulses;
- correlated-channel backgrounds.

### Method comparisons

Evaluate degradation for:

- OF;
- rank-1 EMPCA;
- rank-`k` EMPCA;
- noise-aware linear AE or related weighted baseline;
- isotropic-loss baseline if it fits the study.

### Multichannel study

If suitable multichannel traces exist, add:

- single-channel OF baseline;
- joint-channel correlated OF;
- multichannel EMPCA or correlated-noise diagnostics.

If not, state explicitly that only synthetic multichannel robustness was run.

## Implementation instructions

1. Label every result in this block as robustness support, not theorem support.
2. Keep the baseline clean theorem-verification dataset separate.
3. Use the structured-noise module to answer practical questions:
   - which method degrades first;
   - whether weighted methods retain an advantage;
   - whether extra rank helps under specific out-of-model perturbations.
4. Summarize the noise-module configuration in an appendix-ready format.
5. If one perturbation causes catastrophic failure, preserve that result; it is valuable for the limitations section.

## Recommended outputs

- one robustness panel figure;
- one degradation summary table by perturbation type;
- one appendix subsection describing the structured-noise families used;
- optional multichannel comparison figure.

## Things to avoid

- Do not present these results as proofs of equivalence.
- Do not let the paper's main theorem table depend on these noisy stress tests.
- Do not hide the fact that these perturbations violate the stationary Gaussian assumptions.

## Done when

- `07_experiments.tex` has a practical robustness subsection;
- `08_discussion.tex` can state theorem scope versus out-of-model robustness precisely;
- `appendix.tex` has enough detail for reproducibility.
