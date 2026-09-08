# S5 provenance report (2026-09-03)

Archived 30-seed confirmatory file: `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json` (sha256 `a235af4297255ef34b1baf7a307a3f30c24f90eed6eb64bcc0de14dbe6f8188f`).
Files searched: 90; relative tolerance 0.0005.

## 1. Current manuscript values — where each is found

| quantity | value | found in |
|---|---|---|
| pca_weighted_residual | 48.419 | `evidence/paper_bundle/20260726T044539Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:summary.pca_weighted_residual_to_observed.ci68[0]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row7.pca_weighted_residual_to_observed; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row14.weighted_pca_clean_mse_diagnostic; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:raw_value.pca_weighted_residual_mean (+11 more) |
| empca_weighted_residual | 47.869 | `evidence/paper_bundle/20260726T044539Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:summary.weighted_pca_weighted_residual_to_observed.ci68[0]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row3.weighted_pca_weighted_residual_to_observed; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:raw_value.empca_weighted_residual_mean; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:rows[3].weighted_pca_weighted_residual_to_observed (+9 more) |
| paired_difference | 0.55 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:absolute_difference.weighted_residual_pca_minus_empca.mean_difference; `results/paper1/P1-E5/rerun/S5_confirmatory_30seeds_rerun.json`:absolute_difference.weighted_residual_pca_minus_empca.mean_difference |
| ci_low | 0.53 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:absolute_difference.weighted_residual_pca_minus_empca.ci95[0]; `results/paper1/P1-E5/rerun/S5_confirmatory_30seeds_rerun.json`:absolute_difference.weighted_residual_pca_minus_empca.ci95[0] |
| ci_high | 0.571 | `evidence/paper_bundle/20260726T044539Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:pairs[0].ci95[1]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:absolute_difference.weighted_residual_pca_minus_empca.ci95[1]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:pairs[0].ci95[1]; `results/paper1/P1-E5/rerun/S5_confirmatory_30seeds_rerun.json`:absolute_difference.weighted_residual_pca_minus_empca.ci95[1] (+2 more) |
| fractional_improvement_pct | 1.14 | **not found** |
| nll_difference | 11.67 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:absolute_difference.nll_pca_minus_empca.mean_difference; `results/paper1/P1-E5/rerun/S5_confirmatory_30seeds_rerun.json`:absolute_difference.nll_pca_minus_empca.mean_difference |
| nll_ci_low | 11.24 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:absolute_difference.nll_pca_minus_empca.ci95[0]; `results/paper1/P1-E5/rerun/S5_confirmatory_30seeds_rerun.json`:absolute_difference.nll_pca_minus_empca.ci95[0] |
| nll_ci_high | 12.11 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:absolute_difference.nll_pca_minus_empca.ci95[1]; `results/paper1/P1-E5/rerun/S5_confirmatory_30seeds_rerun.json`:absolute_difference.nll_pca_minus_empca.ci95[1] |
| pca_nll_mean | 1026.92 | `evidence/paper_bundle/20260726T044539Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:summary.pca_nll_mean.ci68[0]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S4/S4_confirmatory_30seeds.csv`:row27.seed; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S4/S4_confirmatory_30seeds.json`:predeclaration.seeds[27]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S4/S4_confirmatory_30seeds.json`:rows[27].seed (+14 more) |
| pca_raw_mse | 2330.27 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:raw_value.pca_raw_residual_mean; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:summary.pca_raw_residual_to_observed.mean; `results/paper1/P1-E5/rerun/S5_confirmatory_30seeds_rerun.json`:raw_value.pca_raw_residual_mean; `results/paper1/P1-E5/rerun/S5_confirmatory_30seeds_rerun.json`:summary.pca_raw_residual_to_observed.mean |
| empca_raw_mse | 7412.67 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:raw_value.empca_raw_residual_mean; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:summary.weighted_pca_raw_residual_to_observed.mean; `results/paper1/P1-E5/rerun/S5_confirmatory_30seeds_rerun.json`:raw_value.empca_raw_residual_mean; `results/paper1/P1-E5/rerun/S5_confirmatory_30seeds_rerun.json`:summary.weighted_pca_raw_residual_to_observed.mean |
| max_angle_deg | 86.25 | `evidence/paper_bundle/20260726T044539Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:summary.subspace_angle_deg_max.ci68[1]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:summary.subspace_angle_deg_max.mean; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:summary.subspace_angle_deg_max.ci68[1]; `results/paper1/P1-E5/rerun/S5_confirmatory_30seeds_rerun.json`:summary.subspace_angle_deg_max.mean (+2 more) |
| angle_ci_low | 85.16 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:summary.subspace_angle_deg_max.ci95[0]; `results/paper1/P1-E5/rerun/S5_confirmatory_30seeds_rerun.json`:summary.subspace_angle_deg_max.ci95[0] |
| angle_ci_high | 87.21 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:summary.subspace_angle_deg_max.ci95[1]; `results/paper1/P1-E5/rerun/S5_confirmatory_30seeds_rerun.json`:summary.subspace_angle_deg_max.ci95[1] |

## 2. Superseded values from OPEN_ISSUES #0 — where each is found

| quantity | value | found in |
|---|---|---|
| pca_weighted_residual | 48.418 | `evidence/paper_bundle/20260726T044539Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:summary.pca_weighted_residual_to_observed.ci68[0]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row7.pca_weighted_residual_to_observed; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row14.weighted_pca_clean_mse_diagnostic; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:raw_value.pca_weighted_residual_mean |
| empca_weighted_residual | 47.878 | `evidence/paper_bundle/20260726T044539Z_eb4592d72ad9/synthetic/S5_sweep_10seeds.json`:summary.weighted_pca_weighted_residual_to_observed.ci68[0]; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row15.weighted_pca_weighted_residual_to_observed; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:raw_value.empca_weighted_residual_mean; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:rows[15].weighted_pca_weighted_residual_to_observed |
| paired_difference | 0.54 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:absolute_difference.weighted_residual_pca_minus_empca.ci68[0]; `results/paper1/P1-E5/rerun/S5_confirmatory_30seeds_rerun.json`:absolute_difference.weighted_residual_pca_minus_empca.ci68[0] |
| ci_low | 0.523 | **not found** |
| ci_high | 0.56 | **not found** |
| pca_raw_mse | 2367.99 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row2.pca_raw_residual_to_observed; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row20.pca_raw_residual_to_observed; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:rows[2].pca_raw_residual_to_observed; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:rows[20].pca_raw_residual_to_observed |
| empca_raw_mse | 5388.75 | **not found** |
| max_angle_deg | 86.07 | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.csv`:row5.subspace_angle_deg_max; `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.json`:rows[5].subspace_angle_deg_max; `results/paper1/P1-E5/rerun/S5_confirmatory_30seeds_rerun.json`:rows[5].subspace_angle_deg_max |

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

## 4. Re-execution of the 30-seed protocol (1 s)

| summary mean | archived | rerun | rel. diff |
|---|---|---|---|
| pca_weighted_residual_to_observed | 48.419219272785604 | 48.41921927278561 | 1.4674807781534517e-16 |
| weighted_pca_weighted_residual_to_observed | 47.86895768898272 | 47.868957688982725 | 1.484349712347372e-16 |
| pca_raw_residual_to_observed | 2330.2722866845334 | 2330.272286684534 | 1.9514773165562978e-16 |
| weighted_pca_raw_residual_to_observed | 7412.665899397949 | 7412.665899397951 | 2.453893684448389e-16 |
| pca_nll_mean | 1026.9224650062088 | 1026.9224650062088 | 0.0 |
| subspace_angle_deg_max | 86.24687806525682 | 86.24687806525685 | 3.295389939668232e-16 |
| pca_over_empca_weighted_residual_ratio | 1.011496210701969 | 1.0114962107019692 | 2.1952094587772546e-16 |

## Verdict

**(a) Substitution confirmed.** Every current manuscript value (the derived 1.14 % excepted) traces to the archived 30-seed confirmatory file, and re-executing the predeclared 30-seed protocol reproduces that file's summary to machine precision. OPEN_ISSUES #0 can be closed: §7.2 is aligned with a reproducible archived run.

Superseded values (OPEN_ISSUES #0): pca_weighted_residual: found; empca_weighted_residual: found; paired_difference: found; ci_low: not found in any archived file; ci_high: not found in any archived file; pca_raw_mse: found; empca_raw_mse: not found in any archived file; max_angle_deg: found. The superseded raw-MSE 5388.75 and the interval [0.523, 0.560] exist in no archived record, so they were never a reproducible run; the hits for the other superseded numbers are per-seed rows or 68 % bounds, not summary values.
