# Manuscript evidence map

This file maps paper claims to evidence inside `transfer_paper/`. It is a
writing-control document: use it to decide what can be stated, what must be
qualified, and what must wait for the next evidence run.

For any row whose support level is pending, diagnostic only, verified negative
when the desired claim would be positive, or not validated, the writing agent
must use a visible placeholder first. Use the exact templates in
`PENDING_RESULT_PLACEHOLDERS.md`. Do not write pending experiments into Results
as if the evidence exists.

## Claim-to-evidence map

| paper claim area | evidence inside this bundle | support level | safe manuscript implication | boundary |
| --- | --- | --- | --- | --- |
| S1 optimal-filter uncertainty matches the CRB | `data/synthetic/S1_sweep_10seeds.*`, `data/derived/synthetic_gate_summary.csv` | verified positive | The synthetic OF normalization is unbiased within the recorded multi-seed interval. | Synthetic Gaussian setting only. |
| S2 EMPCA aligns with the optimal-filter direction | `data/synthetic/S2_sweep_10seeds.*`, `data/derived/synthetic_gate_summary.csv` | verified positive | The weighted subspace geometry converges toward the optimal-filter direction under the amplitude-model test. | Does not prove real-detector filtering equivalence. |
| S3 AE/EMPCA bridge is independent | `data/synthetic/S3_sweep_10seeds.*`, `data/derived/synthetic_gate_summary.csv` | verified positive | The tied linear autoencoder reaches the EMPCA optimum to numerical precision under the independent implementation. | Only validates the tied linear case. |
| S4 white-noise control holds | `data/synthetic/S4_sweep_10seeds.*`, `data/derived/synthetic_gate_summary.csv` | verified positive/control | In white noise, PCA and EMPCA agree as expected. | This is a negative control, not a real-data claim. |
| S5 colored-noise metric reversal holds | `data/synthetic/S5_sweep_10seeds.*`, `data/derived/synthetic_gate_summary.csv` | verified positive | Weighted PCA improves the weighted residual metric in the synthetic colored-noise setting. | Does not imply the global real-data PSD is adequate. |
| S6 timing/rank effect holds | `data/synthetic/S6_sweep_10seeds.*`, `data/derived/synthetic_gate_summary.csv` | verified positive | Higher rank is needed when timing variability is present. | Synthetic timing-jitter model only. |
| S7 covariance estimation converges | `data/synthetic/S7_sweep_10seeds.*`, `data/derived/synthetic_gate_summary.csv` | verified positive | Estimated covariance approaches the oracle as calibration size grows. | Does not guarantee enough stationary calibration data in GWOSC. |
| S8 residual chi-square calibration holds | `data/synthetic/S8_sweep_10seeds.*`, `data/derived/synthetic_gate_summary.csv` | verified positive | The whitened residual chi-square is calibrated under the synthetic assumptions. | Real detector noise violates some synthetic assumptions. |
| S9 full covariance improves multichannel uncertainty | `data/synthetic/S9_sweep_10seeds.*`, `data/derived/synthetic_gate_summary.csv` | verified positive | Full covariance can improve uncertainty relative to diagonal approximation. | Synthetic multichannel covariance setting only. |
| Stage 0 remote reproducibility | `data/gwosc/runs/*/stage0/`, `data/gwosc/runs/*/manifest.json`, `data/derived/gwosc_run_history.csv`, `figures/gwosc_run_history.*` | verified positive for latest run | The workflow has been run on a clean remote Linux checkout with environment and command logs archived. | This is operational reproducibility, not scientific acceptance. |
| GWOSC data coverage and checksums | `data/gwosc/current/gwosc/raw_metadata.json`, `data/gwosc/current/SHA256SUMS`, `data/derived/gwosc_data_quality.csv` | verified positive | The selected H1/L1 GW150914 interval had official DATA coverage, valid event/off-source windows, and verified evidence checksums. | Raw strain arrays are not stored in this bundle. |
| GWpy PSD reference | `data/gwosc/current/gwosc/gwpy_reference.json`, `data/derived/gwosc_reference_summary.csv`, `figures/gwosc_reference_comparison.*` | verified positive | The repository PSD estimator matches GWpy for the same windows and estimator. | This validates the PSD estimator only, not the full likelihood calibration. |
| Global-PSD real-noise null calibration | `data/gwosc/current/gwosc/experiment.json`, `data/derived/gwosc_null_calibration.csv`, `data/derived/gwosc_primary_results.csv`, `figures/gwosc_null_calibration.*` | verified negative | The globally estimated PSD failed the held-out null calibration gate on the selected GWOSC interval. | Do not report event significance or validated sensitivity. |
| Matched statistic path comparison from baseline run | `data/derived/gwosc_reference_summary.csv`, `figures/gwosc_reference_comparison.*` | diagnostic only | The repository GLS score correlates well with repository direct whitening but not consistently with the earlier GWpy FIR path. | The paths were not mathematically identical; do not call this an equivalence failure. |
| Shared-FIR statistic equivalence | `data/gwosc/followup/filter_equivalence.json`, `data/gwosc/followup/filter_equivalence.config.yaml`, `data/derived/gwosc_filter_equivalence_summary.csv`, `figures/gwosc_filter_equivalence.*` | verified positive for shared-statistic implementation identity | The explicit FFT-convolution and GWpy-convolution paths agree under the predeclared shared FIR statistic for synthetic control and H1/L1 real windows. | This does not prove the original PSD-domain GLS statistic and finite-duration FIR statistic are mathematically equivalent; GLS-to-FIR differences remain diagnostic sensitivity. |
| Time-local PSD model | `data/gwosc/followup/time_local_noise.json`, `data/gwosc/followup/time_local_noise.config.yaml`, `data/derived/gwosc_time_local_psd_summary.csv`, `data/derived/gwosc_time_local_psd_blocks.csv`, `data/derived/gwosc_time_local_psd_spectral_summary.csv`, `figures/gwosc_time_local_psd.*` | verified negative on real H1/L1; synthetic control passed | The stationary synthetic control passed, but the predeclared primary 64-second local PSD model failed H1 and L1 real-data score-std acceptance with full coverage. | This does not invalidate the synthetic theory; it shows this GWOSC interval is not calibrated by the tested global/local PSD models. Do not retune radius, thresholds, quality cuts, or windows into a pass. |
| Confirmatory GWOSC interval | `PENDING_RESULT_PLACEHOLDERS.md` | not run | Use a placeholder for any broader GWOSC calibration claim. | Current evidence is one GW150914-centered interval only. |
| CRESST-A within-channel linear hierarchy | `data/cresst/current/cresst/p2_cresst_validation_seed22.json`, `data/cresst/current/cresst/p2_cresst_validation_seed22.config.yaml`, `data/derived/cresst_per_channel_summary.csv`, `data/derived/cresst_pooled_summary.csv` | verified positive primary analysis | Pulse and noise streams were each split chronologically, separately. Rank-6 EMPCA beat single-template OF, met the multi-template-OF competitiveness gate, and all five negative controls degraded at the pooled level across seven channels. The tied linear-AE/EMPCA maximum principal angle was about `1.91e-6` degrees. | The pooled 300-resample event bootstrap is conditional on the observed channels. Per-channel design only; one wrong-PSD control sign exception; trigger efficiency inconclusive; no cross-channel transfer or detection-sensitivity claim. |
| CRESST-A channel-balanced and hierarchical sensitivity | `data/cresst/sensitivity/20260713_channel_robustness_v1/summary.json`, `data/cresst/sensitivity/20260713_channel_robustness_v1/channel_summary.csv`, `data/cresst/sensitivity/20260713_channel_robustness_v1/estimand_summary.csv`, `data/cresst/sensitivity/20260713_channel_robustness_v1/hierarchical_bootstrap_draws.npz`, `figures/cresst_channel_robustness.*` | verified positive post-freeze sensitivity | Equal-channel means were `731.87` for OF minus EMPCA, `279.26` for the competitiveness margin, and `3.78` for PCA minus EMPCA. Channel-then-event hierarchical 95% intervals were `[110.69, 1537.48]`, `[36.52, 569.01]`, and `[1.22, 6.69]`, respectively. All seven channel means were positive for each contrast. | This analysis does not replace the frozen pooled primary gate. It uses 10,000 resamples with seed `20260713`, but only seven run/channel clusters from two runs make cluster-level percentile intervals coarse and potentially unstable. |
| Study A noise-budget band decomposition | `data/study_a/study_a_summary.json`, `data/study_a/study_a_band_fractions.csv`, `data/study_a/study_a_crossovers.csv`, `data/study_a/study_a_quadrature_diagnostics.json`, `data/study_a/source_checksums.json`, `figures/study_a_overview.*`, `figures/study_a_fractional_contributions.*`, `figures/study_a_quadrature_validation.*` | verified positive | Computed directly from the authoritative `noise_module/Al2O3_Al_athermal/{SQUID,Johnson,TD,Er,total}_noise.dat` files under their native labels. SQUID dominates every analyzed band (median fraction 0.867-0.967, 1 Hz-30 MHz) with no SQUID/Johnson, SQUID/TD, or SQUID/Er crossing; Johnson-Er crosses near 319.67 Hz, Johnson-TD near 688.58 Hz, and TD-Er crosses twice (near 11.7 Hz and 1335.3 Hz). Quadrature-sum reconstruction of the total ASD agrees with `total_noise.dat` to `3.3e-16` max fractional deviation. | Never relabel `SQUID`/`Johnson`/`TD`/`Er` (e.g. as "1/f phonon", "vibrational", or "TES Johnson") -- those labels are not supported by the source files. This supersedes the previously mislabeled `data/noise_sample/Study_A/studyA_*` export, which is now marked superseded, not deleted. |
| Study B conditional OF sensitivity attribution | `data/study_b/study_b_summary.json`, `data/study_b/study_b_scenarios.csv`, `data/study_b/source_checksums.json`, `figures/study_b_sensitivity.*` | verified positive, conditional on declared assumptions | Leave-one-out and 0.5x/0.2x scaling of each component under two explicit acquisition branches (`inband_no_alias`, `unfiltered_alias_fold`); SQUID removal improves OF amplitude sensitivity by 3.27x (in-band) / 5.13x (alias-fold), Johnson/TD/Er are each within ~1-5% of baseline. Cross-checked against an independent OF-variance implementation (`noise_module.al2o3_athermal.OptimalFilter`) to 0.014% fractional disagreement on relative sensitivity. Monotonicity gate (removing/reducing a nonnegative component cannot decrease information) holds. | This is conditional sensitivity attribution under a declared PSD and one of two non-exhaustive acquisition branches, not unique physical source decomposition and not a measured detector improvement. Always state which acquisition branch a reported ratio uses. |
| TIDMAD gate G0 (baseline reproduction) | `data/tidmad/current/G0_baseline_reproduction.json`, `data/tidmad/current/tidmad_data_contract.json` | verified positive | The release's own unmodified `benchmark.py` `none` score (1.1026) matches the published value (1.10) on the full 20-file validation set, confirming channel semantics, data integrity, sampling rate, and environment. `savgol` (-2.7708) does not match published (-2.35/-2.77 coarse/fine); diagnosed as numerical ill-conditioning in the published baseline's own hyperparameters (`savgol_coeffs(100, 11)`'s coefficient sum collapses from ~1 to 1.7e-10), not a defect in this reproduction. | Do not claim bit-exact `savgol` reproduction; it is not achievable by any implementation at these hyperparameters. See `docs/results/TIDMAD_PHASE0_RESULT.md` in the parent repository for the full coefficient-sum evidence. |
| TIDMAD Phase 1 real-noise estimator identity | `data/tidmad/current/TIDMAD_P1_real_seed20260727.json` | verified positive, all 6 gates | Amplitude bias, PSD-predicted variance bound, no-signal null, learned-subspace/analytic-quadrature alignment (2.33°), cos²-loss law, and 1/SNR subspace-error scaling law all pass on real detector noise (injected signal is synthetic, into real science-file noise). | Establishes the estimator/whitening machinery is correct on real noise; is not itself the metric-reversal claim (that is Phase 2). |
| **TIDMAD Phase 2 metric reversal against real injected truth** | `data/tidmad/current/TIDMAD_P2_real_seed20260727.json` (headline, file `abra_validation_0000.h5`); replication attempts in `TIDMAD_P2_diagnostic_0010_seed20260727.json` (file 0010) and `TIDMAD_P2_confirmatory_seed20260727.json` (file 0005) — three **distinct** files, verify `config.validation_glob` before citing any number | verified positive, first attempt, no file selection or tuning | On `abra_validation_0000.h5`'s own real injected reference channel as ground truth: paired amplitude-error difference (PCA − EMPCA) `+0.0190`, 95% CI `[0.0045, 0.0321]` window-level / `[0.0027, 0.0344]` block-level bootstrap, both excluding zero. Raw MSE favours PCA, weighted residual favours EMPCA, truth favours EMPCA — all three predicted orderings reversed. | This is the headline TIDMAD result and the strongest real-data result in this repository, because unlike CRESST it has actual injected ground truth. Report exactly as: "on real ABRACADABRA detector data, with the file's own injected reference channel as ground truth." Do not extrapolate beyond this one file/operating point without further runs. **`TIDMAD_P2_real_seed20260727.json` was once silently overwritten by the file-0010 diagnostic run because both used the same output path — fixed 2026-07-27, see OPEN_ISSUES.md item 6 — always spot-check `config.validation_glob` says `abra_validation_0000.h5` before citing this file's numbers.** |
| TIDMAD Phase 3 rank-2 quadrature manifold, real data | `data/tidmad/current/TIDMAD_P3_real_seed20260727.json` (file 0000), `TIDMAD_P3_diagnostic_0010_seed20260727.json` (file 0010), `TIDMAD_P3_confirmatory_seed20260727.json` (file 0005), `TIDMAD_P3_file_selection.json` (the selection scan) — four distinct files | **not confirmed on any of 3 files, including 1 genuinely confirmatory attempt; not a passed or failed predeclared gate** | Three files tried. `abra_validation_0000.h5`: rank1≈rank2, CI spans zero -- diagnosed: reference channel carries a genuine, persistent two-tone signal (a richer manifold than the theory's single-frequency assumption). `abra_validation_0010.h5`: flat across all ranks -- diagnosed: single-tone confirmed, but reference-to-SQUID-channel coupling is only 0.057 (vs. 3.22 for file 0000), so the calibration tone barely reaches the channel being reconstructed. `abra_validation_0005.h5`: selected by a predeclared, mechanically-applied file-selection rule (single-tone + observed-channel SNR ≥3) *before* its outcome was inspected -- genuinely confirmatory, and still flat across ranks (4.752-4.758), likely due to insufficient statistical power at 150 evaluation windows rather than a property of the file. | **Use a placeholder, per `PENDING_RESULT_PLACEHOLDERS.md`, for any real-data rank-2 confirmation claim.** Only the third attempt was genuinely confirmatory (first two files were chosen diagnostically, after seeing a prior failure); all three failed to confirm, for three different, non-implementation reasons. Theory and controlled synthetic Phase 3 results remain fully supported (see `TIDMAD_PHASES_2_TO_6_RESULT.md` in the parent repo). |
| TIDMAD Phase 5 detector-facing denoising score | `data/tidmad/current/TIDMAD_P5_file_seed20260727.json` | verified negative | The noise-aware linear Wiener estimator scored via the unmodified upstream `benchmark.py`: coarse score `-1.8025`, below the "no denoising" baseline (`1.10`) and every other published baseline except the ill-conditioned `savgol`. | Do not claim the linear estimator is competitive with deep baselines on this metric — it is not, at this operating point. Report the predeclared claim boundary honestly: this measures how much of the doing-nothing-to-deep-denoiser gap a linear, noise-aware model explains, and the honest answer here is a negative amount. |
| **TIDMAD Phase 4 calibration boundary, real data** | `data/tidmad/current/TIDMAD_P4_real_seed20260727.json` | verified positive, unexpected | On `abra_validation_0000.h5`/real science-file noise: the global Gaussian PSD model calibrates well (chi2/dof 1.0031, score dispersion 1.0142, both within the predeclared [0.9, 1.1] band and close to the synthetic control's own values); `cost_of_mismatch` is negligible (sigma inflation 1.0002). | Unlike GWOSC, this is a genuine positive finding, not a boundary lesson -- do not force it into a GWOSC-style negative narrative. Do not generalize beyond the one file/~0.2-second sample range checked. |
| TIDMAD Phase 6 negative controls, real data | `data/tidmad/current/TIDMAD_P6_real_seed20260727.json` | 3 of 5 degrade; 2 do not | `wrong_frequency` (+0.357), `permuted_bins` (+0.016), `wrong_psd` (+0.017) all degrade with CI excluding zero. `noise_only_train` is flat (CI spans zero); `random_weighted_basis` improves slightly (CI entirely negative). | Report both exceptions plainly, matching CRESST's precedent of naming its one exception. Do not claim all five controls degrade on real data. |

## Suggested Results narrative

The Results section should move from controlled validation to real-data stress
testing. First, report that all nine synthetic gates passed their multi-seed
criteria. This establishes that the implemented estimators, representations,
and covariance models reproduce the intended mathematical behavior in
controlled settings. The synthetic result should be framed as necessary
validation, not as real-data proof.

Second, report the public CRESST hierarchy. Preserve the pooled event bootstrap
as primary, then state that equal-channel weighting and the post-freeze
channel-then-event bootstrap retained positive OF, multi-template
competitiveness, and same-rank PCA contrasts. Report the exact intervals and
the seven-cluster limitation together. Describe the split only as: “pulse and
noise streams were each split chronologically, separately.”

Third, report the GWOSC PSD reference check. The repository PSD estimator
agreed with GWpy exactly at recorded precision on identical windows for both
H1 and L1. This is important because it isolates the later failure: the
global-PSD real-data gate did not fail because the median-Welch PSD estimator
was trivially inconsistent with GWpy.

Fourth, report the failed held-out null calibration. The observed held-out
score spread exceeded the PSD-predicted spread in both random and chronological
tests, with chronological blocks failing for both detectors. This result should
be written as the main real-data finding currently available. It implies that
the global PSD model is insufficient for calibrated inference on this interval.

Finally, state that event and injection scores were retained only as
diagnostics under the failed calibration. This prevents the manuscript from
making unsupported significance or sensitivity claims.

## Suggested Discussion interpretation

The Discussion should distinguish implementation correctness from statistical
adequacy. The exact PSD agreement with GWpy shows the estimator is implemented
correctly for the chosen reference path. The failed held-out null calibration
shows that a correct global PSD estimator is not enough to produce calibrated
template inference on this real-noise interval. This supports a careful
negative conclusion: the synthetic hierarchy is internally consistent, while
the first real-data stress test exposes nonstationarity or statistic-definition
effects that must be modelled before making astrophysical claims.

The shared-FIR experiment resolves the narrow implementation-identity question:
when both paths compute the same finite FIR statistic with the same
normalization, they agree. It does not retroactively make the original
PSD-domain GLS and finite-FIR statistics equivalent. The time-local PSD
experiment is a negative real-data follow-up: the synthetic control passed, but
the fixed 64-second local model did not calibrate H1 or L1 and should not be
tuned after inspection.

For CRESST, distinguish signal-model capacity from covariance weighting. The
OF contrast is large but changes both model capacity and geometry; the matched-
rank PCA contrast is the more direct weighting comparison. Its channel-aware
interval is positive, while the normalized effect remains heterogeneous across
channels. The correct conclusion is a supported within-channel reconstruction
hierarchy, not detector sensitivity or population-level channel generalization.

## Placeholder insertion rule

For a working manuscript draft, insert placeholders at the exact locations
where missing results will later go. The placeholder must name the experiment
and the required evidence file. The shared-FIR, time-local PSD, and CRESST-A
results no longer require pending-evidence placeholders because their JSON
evidence exists and has been reviewed; they require careful pass/fail wording
instead. Confirmatory GWOSC, calibrated event-significance, and injection-
sensitivity claims should remain placeholders until their archived evidence
appears inside this bundle.

## Manuscript statements to avoid

Do not state that the method detects GW150914 with calibrated significance.
Do not state that the nominal SNR-eight injection target was validated. Do not
state that the original GLS and GWpy FIR statistics are equivalent. Do not
state that the 64-second local PSD model improves or corrects calibration; it
failed the real H1/L1 acceptance gate in this evidence bundle. Do not
generalize from this one 256-second interval to all GWOSC data. Do not
generalize the CRESST-A result beyond within-channel reconstruction. Do not
describe the 19,335 pulses as independent detector-level replications or call
the post-freeze hierarchical analysis the primary analysis. Do not relabel
the Study A `SQUID`/`Johnson`/`TD`/`Er` components with any other physical
name, and do not report a Study B sensitivity ratio without naming its
acquisition branch (`inband_no_alias` or `unfiltered_alias_fold`) or call
Study B a unique physical noise decomposition. Do not claim TIDMAD confirms
the rank-2 quadrature-manifold prediction on real data (Phase 3 did not
confirm it on any of 3 files tried, including 1 genuinely confirmatory
attempt). Do not claim bit-exact reproduction of the published `savgol`
baseline score. Do not claim the noise-aware linear estimator is
competitive with trained deep denoisers on TIDMAD's detector-facing score
(Phase 5 scored below the "no denoising" baseline). Do not claim all five
TIDMAD Phase 6 negative controls degrade on real data (2 of 5 do not). Do
not generalize TIDMAD Phase 2's or Phase 4's results beyond the one
file/operating point each was run at.

## Tables and plots to use

Use `data/derived/paper_implications.csv` as the machine-readable version of
this map. Use `data/derived/method_traceability.csv` when writing Methods or a
supplemental reproducibility section. Use `data/derived/figure_index.csv` to
decide which figures are currently evidence-backed. Use `tables/figure_captions.md`
for draft captions.
