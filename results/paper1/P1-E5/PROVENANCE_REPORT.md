# S5 provenance report (2026-09-02)

Archived 30-seed confirmatory file: `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json` (sha256 `a235af4297255ef34b1baf7a307a3f30c24f90eed6eb64bcc0de14dbe6f8188f`).
Files searched: 89; relative tolerance 0.0005.

## 1. Current manuscript values — where each is found

| quantity | value | found in |
|---|---|---|
| pca_weighted_residual | 48.419 | `evidence/paper_bundle/20260726T044539Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:summary.pca_weighted_residual_to_observed.ci68[0]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row7.pca_weighted_residual_to_observed; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row14.weighted_pca_clean_mse_diagnostic; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:raw_value.pca_weighted_residual_mean (+7 more) |
| empca_weighted_residual | 47.869 | `evidence/paper_bundle/20260726T044539Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:summary.weighted_pca_weighted_residual_to_observed.ci68[0]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row3.weighted_pca_weighted_residual_to_observed; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:raw_value.empca_weighted_residual_mean; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:rows[3].weighted_pca_weighted_residual_to_observed (+6 more) |
| paired_difference | 0.55 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:absolute_difference.weighted_residual_pca_minus_empca.mean_difference |
| ci_low | 0.53 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:absolute_difference.weighted_residual_pca_minus_empca.ci95[0] |
| ci_high | 0.571 | `evidence/paper_bundle/20260726T044539Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:pairs[0].ci95[1]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:absolute_difference.weighted_residual_pca_minus_empca.ci95[1]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:pairs[0].ci95[1]; `results/sweeps/S5_sweep_10seeds.json`:pairs[0].ci95[1] (+1 more) |
| fractional_improvement_pct | 1.14 | **not found** |
| nll_difference | 11.67 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:absolute_difference.nll_pca_minus_empca.mean_difference |
| nll_ci_low | 11.24 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:absolute_difference.nll_pca_minus_empca.ci95[0] |
| nll_ci_high | 12.11 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:absolute_difference.nll_pca_minus_empca.ci95[1] |
| pca_nll_mean | 1026.92 | `evidence/paper_bundle/20260726T044539Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:summary.pca_nll_mean.ci68[0]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S4/S4_confirmatory_30seeds.csv`:row27.seed; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S4/S4_confirmatory_30seeds.json`:predeclaration.seeds[27]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S4/S4_confirmatory_30seeds.json`:rows[27].seed (+10 more) |
| pca_raw_mse | 2330.27 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:raw_value.pca_raw_residual_mean; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:summary.pca_raw_residual_to_observed.mean |
| empca_raw_mse | 7412.67 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:raw_value.empca_raw_residual_mean; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:summary.weighted_pca_raw_residual_to_observed.mean |
| max_angle_deg | 86.25 | `evidence/paper_bundle/20260726T044539Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:summary.subspace_angle_deg_max.ci68[1]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:summary.subspace_angle_deg_max.mean; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:summary.subspace_angle_deg_max.ci68[1]; `results/sweeps/S5_sweep_10seeds.json`:summary.subspace_angle_deg_max.ci68[1] (+1 more) |
| angle_ci_low | 85.16 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:summary.subspace_angle_deg_max.ci95[0] |
| angle_ci_high | 87.21 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:summary.subspace_angle_deg_max.ci95[1] |

## 2. Superseded values from OPEN_ISSUES #0 — where each is found

| quantity | value | found in |
|---|---|---|
| pca_weighted_residual | 48.418 | `evidence/paper_bundle/20260726T044539Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:summary.pca_weighted_residual_to_observed.ci68[0]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row7.pca_weighted_residual_to_observed; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row14.weighted_pca_clean_mse_diagnostic; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:raw_value.pca_weighted_residual_mean |
| empca_weighted_residual | 47.878 | `evidence/paper_bundle/20260726T044539Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:summary.weighted_pca_weighted_residual_to_observed.ci68[0]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row15.weighted_pca_weighted_residual_to_observed; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:raw_value.empca_weighted_residual_mean; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:rows[15].weighted_pca_weighted_residual_to_observed |
| paired_difference | 0.54 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:absolute_difference.weighted_residual_pca_minus_empca.ci68[0] |
| ci_low | 0.523 | **not found** |
| ci_high | 0.56 | **not found** |
| pca_raw_mse | 2367.99 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row2.pca_raw_residual_to_observed; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row20.pca_raw_residual_to_observed; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:rows[2].pca_raw_residual_to_observed; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:rows[20].pca_raw_residual_to_observed |
| empca_raw_mse | 5388.75 | **not found** |
| max_angle_deg | 86.07 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row5.subspace_angle_deg_max; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:rows[5].subspace_angle_deg_max |

## 3. S1–S9 ten-seed sweeps: bundle vs results/sweeps

| file | identical | bundle sha256 |
|---|---|---|
| `transfer_paper/data/synthetic/S1_sweep_10seeds.csv` | yes | `94edc6b10849bc6d…` |
| `transfer_paper/data/synthetic/S1_sweep_10seeds.json` | yes | `9a8591a9ea0360cb…` |
| `transfer_paper/data/synthetic/S2_sweep_10seeds.csv` | yes | `2e1b21e1b36fb06c…` |
| `transfer_paper/data/synthetic/S2_sweep_10seeds.json` | yes | `7eef9b15f37c7276…` |
| `transfer_paper/data/synthetic/S3_sweep_10seeds.csv` | yes | `a7377cd1bd848487…` |
| `transfer_paper/data/synthetic/S3_sweep_10seeds.json` | yes | `535cf157185c7968…` |
| `transfer_paper/data/synthetic/S4_sweep_10seeds.csv` | yes | `ccc8463444e0607b…` |
| `transfer_paper/data/synthetic/S4_sweep_10seeds.json` | yes | `3d36fc2bade4ac0e…` |
| `transfer_paper/data/synthetic/S5_sweep_10seeds.csv` | yes | `aec18a0508aa6421…` |
| `transfer_paper/data/synthetic/S5_sweep_10seeds.json` | yes | `cb036ac9c524658c…` |
| `transfer_paper/data/synthetic/S6_sweep_10seeds.csv` | yes | `06285178068af397…` |
| `transfer_paper/data/synthetic/S6_sweep_10seeds.json` | yes | `edb00ae0bd519d64…` |
| `transfer_paper/data/synthetic/S7_sweep_10seeds.csv` | yes | `2feccd40585678bc…` |
| `transfer_paper/data/synthetic/S7_sweep_10seeds.json` | yes | `9e0ab6bb29a867ab…` |
| `transfer_paper/data/synthetic/S8_sweep_10seeds.csv` | yes | `553fd12f66b79bcb…` |
| `transfer_paper/data/synthetic/S8_sweep_10seeds.json` | yes | `5c23f26b2d38903b…` |
| `transfer_paper/data/synthetic/S9_sweep_10seeds.csv` | yes | `ff98a71f5548afca…` |
| `transfer_paper/data/synthetic/S9_sweep_10seeds.json` | yes | `729f27a16d8e48eb…` |

## 4. Re-execution

Not run (`--rerun` not given).

## Verdict

Fill in after reading sections 1–4: (a) every current manuscript value traces to the archived 30-seed file → OPEN_ISSUES #0 closes as *substitution confirmed*; (b) the superseded values trace to another archived run → cite that run instead; (c) neither → the numbers are unrecoverable and must be regenerated from section 4.
