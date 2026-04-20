# Block 08: PC Interpretation and Centering

## Goal

Convert the existing interpretation notebooks into a cautious, paper-usable interpretation story for the leading EMPCA components.

## Paper sections supported

- `04_empca.tex`
- `07_experiments.tex`
- `08_discussion.tex`
- `appendix.tex`

## Required deliverables

### Leading-component interpretation

Quantify:

- `PC1` overlap with empirical mean;
- `PC1` overlap with template;
- `PC1` overlap with OF-like direction;
- `PC2` overlap with timing-derivative proxy;
- `PC3` overlap with width / deformation proxy.

### Coefficient correlations

Correlate component coefficients with:

- amplitude proxy;
- timing proxy;
- width or shape proxy;
- any additional physically meaningful nuisance variables available in the data.

### Centered vs uncentered ablation

Show how centering changes:

- the leading mode;
- mean-direction overlap;
- interpretability of the first coefficient.

## Implementation instructions

1. Keep the language conservative: "consistent with" is better than "proves."
2. Distinguish between the strict rank-1 theorem regime and general rank-`k` interpretation.
3. Use the same weighted cosine and alignment conventions throughout.
4. Prefer one summary table plus a small number of figures over many redundant overlays.
5. If the interpretation is unstable across seeds or preprocessing, say so explicitly.

## Recommended outputs

- one metrics table for `PC1`, `PC2`, `PC3`;
- one centered-vs-uncentered comparison figure;
- one coefficient-correlation table;
- one short interpretation note that can be quoted in `07_experiments.tex`.

## Things to avoid

- Do not claim `PC1 = OF direction` in the general uncentered rank-`k` setting unless the evidence actually supports that exact statement.
- Do not mix several different proxy definitions without documenting them.
- Do not force a physical interpretation onto a weakly identified component.

## Done when

- the paper has a defensible leading-PC interpretation story;
- the difference between mean-like, timing-like, and deformation-like directions is explicit;
- centering effects are no longer an implicit notebook-only detail.
