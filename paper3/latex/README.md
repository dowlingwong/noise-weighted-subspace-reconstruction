# Paper 3 LaTeX proposal

The source builds the three-page **ShiftScope** systems proposal. The longer
implementation plan remains in `../PAPER3_PLAN.md` as a technical appendix.

Compile from this directory so the relative pilot-figure path resolves:

```bash
pdflatex -interaction=nonstopmode -halt-on-error paper3_proposal.tex
pdflatex -interaction=nonstopmode -halt-on-error paper3_proposal.tex
```

The second pass resolves the internal figure reference and citations from the
document's self-contained bibliography. The deliverable is
`paper3_proposal.pdf`.
