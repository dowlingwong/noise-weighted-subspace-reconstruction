# Presentation figure audit

Source deck: `/Users/wongdowling/Downloads/NPML_dwong_2026_final.pptx`

All 19 slides were rendered and inspected on 2026-08-07. The following slides were the only serious candidates for the R/PS mini-paper.

| Slide | Candidate | Decision | Reason |
|---:|---|---|---|
| 5 | Likelihood contours under white-noise and detector-noise metrics | Retain asset, remove from paper | Directly illustrates covariance geometry, but its curved learned manifold conflicts with the paper's linear subspace analysis |
| 9 | Raw and detector-PSD-weighted residual spectra | Exclude | Slide labels the methodology exploratory; the plot is not part of the current audited experimental package |
| 11 | Resolution versus retained principal components | Exclude | Older method labels and configurations do not map cleanly onto the distilled paper's audited results |
| 12 | EMPCA versus NFPA architecture comparison | Remake/retain existing alternative only | The slide describes NFPA as nonlinear, which conflicts with the current validated linear Kronecker-projection claim; the mini-paper equations already state the correct restriction |

The original embedded slide-5 asset remains at `figures/likelihood_geometry_contours.png` for provenance, but is no longer included in the paper. The intended replacement is a two-panel empirical figure generated only after the multi-seed experiment in `notes/nfpa_submission_experiment_plan.md` is completed. The pipeline schematic remains in `figures/geometry_schematic.pdf` as an unused alternative.
