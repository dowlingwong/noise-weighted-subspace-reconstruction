# Paper 1 revision instruction

Status date: 12 July 2026.

This is the authoritative instruction for the next paper-writing pass. It
supersedes June 2026 statements in this bundle that describe CRESST-A as
pending. Numerical claims must still be checked against the archived JSON;
derived CSV files are convenience views.

## Editorial objective

Revise the manuscript as a concise MLST research paper and a matching arXiv
preprint. MLST currently says that research papers are normally no more than
8500 words. Target 7000-8000 words in the main manuscript, excluding references
and genuinely supplementary derivations. The present PDF is 76 pages, with 44
pages before the appendices and about 20,000 extracted words; it is not ready in
its current length or evidence state.

The paper should make one central claim:

> Under a stated linear Gaussian noise model, the inverse noise covariance
> defines the reconstruction geometry, unifying optimal filtering, weighted
> low-rank reconstruction/EMPCA, and tied noise-aware linear autoencoders as
> constrained likelihood projections.

The evidence arc should be: theory, controlled synthetic validation, positive
public CRESST reconstruction validation, then the negative GWOSC calibration
stress test as a boundary result. Do not make the paper a catalogue of every
implemented diagnostic.

## Evidence that must replace the stale CRESST placeholder

The frozen CRESST-A run passed at commit
`9c9f696ff6bca6e6b9729cdb90e41ee92d576787`. It used all seven public test-
release channels, split each channel chronologically into its own calibration
and evaluation segments, found zero duplicate traces, and evaluated 19,335
held-out pulses in the pooled comparison.

Use these paper-facing results:

- Rank-6 EMPCA beat single-template OF: paired weighted-residual difference
  `OF - EMPCA = 1012.27`, pooled event-bootstrap 95% CI
  `[927.90, 1097.61]`.
- EMPCA met the predeclared multi-template-OF competitiveness gate:
  `1.1 x multi-template OF - EMPCA = 352.17`, 95% CI
  `[272.40, 424.01]`.
- All five negative controls degraded at the pooled level with the required
  CI direction. The weakest was wrong-PSD weighting: mean difference `1.57`,
  95% CI `[0.97, 2.08]`.
- One channel-level exception must be disclosed: wrong-PSD did not degrade on
  channel `35_58.0` (mean `-0.005`, CI approximately `[-0.010, 0.0001]`).
- The tied linear-AE and EMPCA subspaces agreed on the public CRESST run: the
  maximum principal angle across channels was about
  `1.91e-6` degrees. This is a real-data consistency check of the linear bridge,
  not evidence for nonlinear autoencoders.
- Trigger efficiency was inconclusive in the frozen amplitude grid for six of
  seven channels at the primary 5% FPR; no channel reached 90% efficiency.
  Therefore do not claim validated detection sensitivity.

Authoritative sources:

- `data/cresst/current/cresst/p2_cresst_validation_seed22.json`
- `data/cresst/current/cresst/p2_cresst_validation_seed22.config.yaml`
- `data/derived/cresst_per_channel_summary.csv`
- `data/derived/cresst_pooled_summary.csv`

In the parent experiment repository, the reviewed interpretation and frozen
gate record are `docs/CRESST_RESULT.md` and
`docs/CRESST_PREDECLARED_GATES.md`.

## Required manuscript changes

1. Replace every CRESST/SCRESST placeholder and every statement that CRESST is
   planned with the completed result above. Update the abstract, contributions,
   Results, Discussion, Limitations, conclusion, paper organization, claim-state
   figure, and captions consistently.
2. Remove or sharply reduce the internal cryogenic calibration sample. The
   public CRESST result is stronger and avoids confusing an internal sample
   with the public validation. Retain an internal result only if it tests a
   distinct claim unavailable in CRESST, and label it explicitly.
3. Compress the theory. Keep the unified likelihood objective, the rank-1
   OF/EMPCA result, the weighted linear-AE bridge, and the MSE/colored-noise
   consequence in the main text. Move full proofs, EM update details,
   conditioning notes, symmetry/gauge discussion, and extended OF derivations
   to supplementary material.
4. Consolidate experiments. Use one compact synthetic validation figure/table,
   one CRESST figure/table, and one compact GWOSC boundary figure. Move run
   history, most implementation-identity grids, extended rank sweeps, detector
   simulator audits, and secondary diagnostics to the supplement or repository.
5. Write GWOSC as a completed negative stress test. State that the PSD reference
   and shared-FIR identity checks passed, while held-out calibration failed for
   both global and predeclared 64-second local PSD models. Event and injection
   scores remain diagnostic only.
6. Remove all visible working-draft placeholders from a submission manuscript.
   Pending experiments should be omitted unless they are necessary to a claim,
   in which case the manuscript is not ready.
7. Add precise data/code availability and provenance statements, including the
   frozen commits, archived configs, split unit, event counts, bootstrap unit,
   and public dataset citation.

## Required analysis before external review

The completed experiment is appropriate for the narrow within-channel
reconstruction claim, but the pooled uncertainty resamples 19,335 events as if
they were the inferential units even though the seven detector channels are the
natural replication clusters and their residual scales differ substantially.
Before review, add a non-tuning robustness analysis that preserves the frozen
method outputs:

- report all seven channel effects and their signs prominently;
- add a channel-balanced summary (equal channel weight); and
- add a hierarchical or cluster-aware bootstrap that resamples channels and
  then events within channels, clearly noting that only seven clusters limit
  precision.

This is a robustness reanalysis, not permission to change rank, covariance,
templates, splits, thresholds, or exclusions. Preserve the original
predeclared pooled gate as the primary result and label the cluster-aware result
as a sensitivity analysis.

Also generate a paper-facing CRESST figure from the synchronized evidence. It
should show per-channel paired OF-versus-EMPCA effects, the multi-template
comparison, and the five negative controls with uncertainty. A separate small
panel or table may report the AE/EMPCA principal-angle agreement. Do not feature
the inconclusive trigger curves as a positive result.

## Claim boundaries

Allowed:

- controlled equivalence and robustness claims for S1-S9;
- within-channel public CRESST reconstruction validation for the linear
  hierarchy;
- public CRESST AE/EMPCA subspace consistency in the tied linear case;
- GWOSC implementation checks as positive diagnostics;
- GWOSC global/local PSD calibration as a negative boundary result.

Not allowed:

- cross-channel or cross-detector transfer;
- validated CRESST detection sensitivity;
- calibrated GW150914 significance, false-alarm probability, or SNR-eight
  sensitivity;
- general GWOSC calibration;
- equivalence between the original PSD-domain GLS statistic and a finite FIR
  statistic;
- claims about nonlinear autoencoders.

## Readiness estimate

- Experimental implementation and archived execution: **85% ready**.
- Evidence/claim alignment: **80% ready**.
- Current manuscript: **45% ready**, because it is too long and omits the
  completed CRESST result.
- Overall readiness for external peer review: **about 65%**.

After the cluster-aware CRESST robustness analysis, CRESST paper figure,
evidence-map refresh, concise rewrite, citation/provenance audit, and final
render/claim audit, expected readiness is approximately **90%**. No new primary
experiment is mandatory for the paper's narrow reconstruction claim. A new
predeclared CRESST sensitivity stage or GWOSC Stage 3b is mandatory only if the
authors want detection-sensitivity or calibrated-significance claims.
