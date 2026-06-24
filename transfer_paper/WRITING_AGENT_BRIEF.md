# Writing-agent brief for Paper 1

Status date: 24 June 2026.

This directory is intended to be a standalone paper-writing bundle. A writing
agent should be able to read this folder, inspect the copied evidence, use the
derived CSVs and figures, and draft paper updates without opening the original
experiment directories or contacting the remote server. Raw GWOSC strain arrays
are intentionally excluded because they are large and are not needed for
manuscript interpretation; the bundle carries the run logs, metadata,
checksums, configs, JSON evidence, derived tables, notebooks, and plots.

Before drafting Results, read `PENDING_RESULT_PLACEHOLDERS.md`. Any experiment
that has not been run, has not been synchronized into this bundle, or did not
solidly pass its predeclared acceptance gate must be written as an explicit
placeholder first. A placeholder is safer than speculative prose. It must be
resolved or removed before submission.

## Current evidence state

The synthetic validation ladder is the strongest positive result. Gates S1–S9
have multi-seed sweep records under `data/synthetic/` and a compact summary in
`data/derived/synthetic_gate_summary.csv`. These results support the paper's
controlled claims about optimal-filter normalization, weighted subspace
geometry, AE/EMPCA equivalence, colored-noise metric behavior, rank sensitivity,
covariance estimation, residual calibration, and multichannel covariance.

The remote reproducibility gate has passed. The archived GWOSC run
`20260622T175125Z_b169c1f595a4` passed Stage 0 on a clean remote Linux checkout
and recorded the environment, dependencies, tests, core runs, table generation,
figure generation, and checksums. The audit trail is copied under
`data/gwosc/runs/`; the selected latest run is duplicated under
`data/gwosc/current/` for plotting.

The GWOSC PSD reference check is a positive implementation result. On the
selected GW150914 H1/L1 interval, the repository Hann-windowed,
bias-corrected median Welch PSD matches GWpy's median-Welch PSD on identical
calibration windows at recorded precision. The median, fifth-percentile, and
ninety-fifth-percentile PSD ratios are all `1.0`, the relative L2 error is
`0.0`, and the maximum absolute log10 ratio is `0.0` for both detectors. This
supports a Methods/Results statement that the PSD estimator has been validated
against an independent reference implementation.

The GWOSC held-out calibration result is negative. The global-PSD model failed
the predeclared real-noise acceptance gate. In random splits, H1 had
observed/predicted null-sigma median/min/max `1.986 / 1.396 / 7.166`, and L1
had `1.263 / 1.209 / 3.093`. In chronological blocks, H1 had
`3.183 / 1.549 / 10.480`, and L1 had `2.964 / 1.973 / 7.619`. This must be
reported as a failure of global-PSD real-noise calibration, not as a successful
GWOSC detection or significance result.

The event and injection scores are diagnostics only. The archived event scores
from the old approximate-template path were `6.636` for H1 and `-0.289` for L1.
The empirical injection SNRs were `2.396` for H1 and `6.435` for L1, below the
nominal target because the null statistic was under-calibrated. These values
must not be interpreted as calibrated event significance, false-alarm evidence,
or validated sensitivity.

The filtering/statistic-equivalence and time-local PSD experiments are
implemented locally and predeclared in the bundle, but their remote evidence is
pending. Until `data/gwosc/followup/filter_equivalence.json` and
`data/gwosc/followup/time_local_noise.json` exist, these experiments can be
described in Methods as planned/predeclared follow-up protocols, but not as
Results.

The confirmatory GWOSC interval, CRESST/SCRESST validation, calibrated event
significance, and validated injection sensitivity are also pending. Drafts
should use the exact placeholders in `PENDING_RESULT_PLACEHOLDERS.md` wherever
those results would be inserted.

## How this supports the paper

The paper can make a strong controlled-methods claim: the mathematical and
software validation ladder now has seed-sweep evidence, held-out evaluation
where appropriate, and regression-test support. This is the evidence that the
core noise-weighted subspace machinery behaves as intended under controlled
conditions.

The paper can also make a narrower real-data implementation claim: the PSD
normalization path was checked against GWpy on public GWOSC data and matched
exactly for the same estimator and windows. This strengthens the Methods
section because it shows the real-data failure is not caused by a simple PSD
implementation mismatch.

The paper cannot yet claim real-detector calibration. The central real-data
finding is that the globally estimated PSD did not predict held-out
template-score spread on the 256-second GW150914-centered interval. This is
scientifically useful because it identifies a real-noise gap that synthetic
validation alone would miss. The Discussion should frame this as a boundary of
the present method and motivation for identical-statistic filtering tests and
time-local spectral modelling.

## Recommended manuscript updates

In Methods, describe the S1–S9 validation ladder, the seed-sweep/held-out
infrastructure, the remote Stage 0 reproducibility gate, GWOSC data provenance,
official DATA coverage, Hann-windowed median PSD estimation, and the GWpy PSD
reference comparison. Describe the shared-FIR and local-PSD protocols only as
predeclared follow-up methods until remote evidence arrives.

In Results, report the synthetic gate outcomes and the GWpy PSD agreement as
positive results. Then report the GWOSC global-PSD held-out calibration failure
directly, including both random splits and chronological blocks. The negative
result should be visible, not hidden in limitations.

In Discussion, state that PSD-estimator correctness is necessary but not
sufficient for calibrated likelihood inference on nonstationary real detector
noise. Treat local spectral drift, template-projected spectral variation,
narrow-band features, and finite-duration filtering differences as hypotheses
that the pending follow-up experiments are designed to test.

In Limitations, state that the current GWOSC evidence covers one 256-second
interval around one event, that split seeds over the same interval are not
independent astrophysical replications, that event/injection scores are
uncalibrated, and that confirmatory intervals are required before any general
real-noise calibration claim.

## Files a writing agent should start with

Read `PAPER_WRITING_HANDOFF.md` first for the current high-level state. Then
read `MANUSCRIPT_EVIDENCE_MAP.md` for claim-by-claim traceability and
`PENDING_RESULT_PLACEHOLDERS.md` for placeholder text. Use
`data/derived/paper_implications.csv`, `data/derived/method_traceability.csv`,
and `data/derived/claim_status.csv` as machine-readable writing constraints.
Use `tables/figure_captions.md` for draft captions and `figures/` for rendered
plots.

The available paper-facing figures are:

| figure | purpose |
| --- | --- |
| `figures/synthetic_validation_overview.*` | representative positive synthetic validation results |
| `figures/gwosc_null_calibration.*` | failed real-data global-PSD held-out calibration |
| `figures/gwosc_reference_comparison.*` | exact GWpy PSD agreement plus diagnostic score-path comparison |
| `figures/gwosc_run_history.*` | evidence audit trail across remote runs |
| `figures/paper_claim_support_matrix.*` | writing-control view of verified, negative, pending, and not-validated claims |

The pending figures `gwosc_filter_equivalence.*` and `gwosc_time_local_psd.*`
should appear only after the next remote evidence sync.
