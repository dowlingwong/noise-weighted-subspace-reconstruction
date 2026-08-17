# Source map

| Mini-paper section | Source paper | Source section/file | Kept | Cut |
|---|---|---|---|---|
| Abstract and introduction | Papers 1 and 2 plus cited literature | Paper 1 `00_abstract.tex`, `01_introduction.tex`; Paper 2 `00_abstract.tex`, `01_intro.tex`; Noise2Self, SSDU, Noisier2Inverse, generalized PCA, MPCA | Label-free/self-supervised reconstruction, instrument-derived covariance prior, metric/architecture separation, novelty boundary | Detector-heavy motivation, transfer/sampling claims, nonlinear roadmap |
| Covariance geometry | Paper 1 | `02_unified_objective.tex`, `04_empca.tex`, `05_linear_ae.tex`, `06_loss_geometry.tex`, `appD_proofs.tex` | Gaussian objective, whitening, OF/CW-PCA/tied-AE bridge, metric reversal, centering and covariance-estimation contract | Full OF derivation, Fisher analysis, convergence and sensitivity details |
| NFPA | Paper 2 | `03_nfpa.tex` | Encoder/decoder, product Grassmannian, restricted bridge, rank-one correspondence | Long framework, proposed approximation bound, design-space roadmap |
| Evidence | Papers 1 and 2 | Paper 1 `07_experiments.tex`; Paper 2 `04_nfpa_experiments.tex` | Controlled metric reversal, NFPA metrics, principal angles, limitations | CRESST/GWOSC/TIDMAD protocols and detector-performance claims |
| Discussion | Paper 2 | `01_intro.tex`, `07_discussion.tex` | Compact metric/architecture tuple and calibrated limitation language | Coverage theorem, U-Net case study, transfer/sampling claims, VAE/Transformer extensions |

## Numerical provenance

The 30-seed colored-noise comparison, ten-seed white-noise control, and single-file recorded-truth check are copied from Paper 1 `07_experiments.tex`. The NFPA result table, subspace angles, SNR, sample sizes, ranks, and rank-one cosine are copied from Paper 2 `04_nfpa_experiments.tex`. Paper 2 explicitly labels these results as one synthetic setting, one seed, and one train/test split. No unreported uncertainty or exact sub-$10^{-5}$ raw-MSE value is inferred.

## Figure provenance

`figures/likelihood_geometry_contours.png` is the original embedded plot asset extracted from slide 5 of `/Users/wongdowling/Downloads/NPML_dwong_2026_final.pptx`. It is retained for provenance but removed from the manuscript because its curved learned manifold is inconsistent with the paper's linear subspace analysis.

`figures/geometry_schematic.pdf` is generated locally from `figures/geometry_schematic.tex` and retained as an unused alternative. `figures/metric_reversal_summary.pdf` is copied from Paper 2 `figures/nfpa_vs_empca.pdf` for provenance but is not used because it summarizes a separate archived separability sweep and could be confused with the single-setting table.

The planned empirical replacement figure and the exact experiments required to generate it are specified in `notes/nfpa_submission_experiment_plan.md`. No placeholder result or unrun confidence interval has been added to the manuscript.
