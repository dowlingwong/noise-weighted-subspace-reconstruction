# Block 10: Tables, Figures, Appendix, and Reproducibility

## Goal

Convert the outputs of Blocks 01-09 into paper-ready artifacts and a reproducible appendix trail.

## Paper sections supported

- `07_experiments.tex`
- `08_discussion.tex`
- `appendix.tex`
- `main.tex`

## Required deliverables

### Paper tables

Populate at least:

- theorem-verification summary table;
- K-alpha dataset characterization table;
- rank-`k` performance table;
- Study A summary table;
- Study A crossover table;
- Study B scenario table;
- any AE-equivalence summary table referenced in the text.

### Paper figures

Prepare and freeze:

- one hierarchy-support figure if needed;
- one strict theorem-verification comparison plot;
- one rank-`k` saturation plot;
- one convergence plot;
- one robustness plot;
- one PC-interpretation figure set kept to the minimum useful number.

### Reproducibility manifest

Create an internal manifest containing:

- dataset version and counts;
- split seed;
- PSD and weight file paths;
- template file paths;
- model rank and seed;
- code location or notebook used;
- output artifact path.

## Implementation instructions

1. Make sure every table or figure cited in the manuscript has a real artifact behind it.
2. Prefer a small set of high-information figures over many notebook-style panels.
3. Keep appendix figures for overflow material that helps reproducibility but not the main story.
4. Verify all cross-references, filenames, and labels after the final artifact set is chosen.
5. Write a short internal note on any results intentionally excluded from the main paper and why.

## Suggested packaging

- `results/tables/` for CSV or JSON summaries;
- `results/figures/` for final plots;
- appendix text for generator details, extra diagnostics, and implementation notes.

Adjust the actual output location to the repo's current conventions if different, but keep the structure conceptually equivalent.

## Things to avoid

- Do not leave placeholder tables in the paper draft.
- Do not rely on notebook output cells as the final artifact source.
- Do not mix draft and final figure versions without a manifest.

## Done when

- all manuscript placeholders referenced by the plan are backed by real data;
- the appendix can reproduce every important preprocessing and robustness choice;
- the April 20 implementation plan has a visible path into the manuscript files.
