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
| Shared-FIR statistic equivalence | `data/configs/gwosc/filter_statistic_equivalence.yaml`, `data/source_documents/GWOSC_FILTERING_AND_LOCAL_PSD_PROTOCOL.md`, `data/gwosc/followup/PENDING.md` | implemented pending remote evidence | The paper can describe the predeclared test design. | No real-data result can be claimed yet. |
| Time-local PSD model | `data/configs/gwosc/time_local_noise.yaml`, `data/source_documents/GWOSC_FILTERING_AND_LOCAL_PSD_PROTOCOL.md`, `data/gwosc/followup/PENDING.md` | implemented pending remote evidence | The paper can describe local PSD radii, chronological blocks, template-projected diagnostics, and narrow-band diagnostics as the next test. | Do not claim improvement until remote evidence exists. |
| Confirmatory GWOSC interval | `PENDING_RESULT_PLACEHOLDERS.md` | not run | Use a placeholder for any broader GWOSC calibration claim. | Current evidence is one GW150914-centered interval only. |
| CRESST / SCRESST validation | `PENDING_RESULT_PLACEHOLDERS.md` | not run | Use a placeholder for any CRESST result. | No schema-validated CRESST evidence is present in this bundle. |

## Suggested Results narrative

The Results section should move from controlled validation to real-data stress
testing. First, report that all nine synthetic gates passed their multi-seed
criteria. This establishes that the implemented estimators, representations,
and covariance models reproduce the intended mathematical behavior in
controlled settings. The synthetic result should be framed as necessary
validation, not as real-data proof.

Second, report the GWOSC PSD reference check. The repository PSD estimator
agreed with GWpy exactly at recorded precision on identical windows for both
H1 and L1. This is important because it isolates the later failure: the
global-PSD real-data gate did not fail because the median-Welch PSD estimator
was trivially inconsistent with GWpy.

Third, report the failed held-out null calibration. The observed held-out
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

The pending shared-FIR experiment tests whether the earlier GLS/GWpy
score-path discrepancy is an implementation/filter-definition issue. The
pending local-PSD experiment tests whether local spectral modelling can reduce
the observed null-spread inflation. These should be presented as predeclared
next tests, not as confirmed explanations.

## Placeholder insertion rule

For a working manuscript draft, insert placeholders at the exact locations
where missing results will later go. The placeholder must name the experiment
and the required evidence file. For example, the shared-FIR Results subsection
should contain the shared-FIR placeholder until
`data/gwosc/followup/filter_equivalence.json` exists. The local-PSD Results
subsection should contain the time-local placeholder until
`data/gwosc/followup/time_local_noise.json` exists. CRESST and confirmatory
GWOSC claims should remain placeholders until their archived evidence appears
inside this bundle.

## Manuscript statements to avoid

Do not state that the method detects GW150914 with calibrated significance.
Do not state that the nominal SNR-eight injection target was validated. Do not
state that the original GLS and GWpy FIR statistics are equivalent. Do not
state that the 64-second local PSD model improves calibration. Do not generalize
from this one 256-second interval to all GWOSC data or to CRESST.

## Tables and plots to use

Use `data/derived/paper_implications.csv` as the machine-readable version of
this map. Use `data/derived/method_traceability.csv` when writing Methods or a
supplemental reproducibility section. Use `data/derived/figure_index.csv` to
decide which figures are currently evidence-backed. Use `tables/figure_captions.md`
for draft captions.
