# GWOSC negative-result revision guide

Purpose: preserve the current GWOSC conclusion as a paper-revision guideline.
This document is the short, current interpretation layer above the detailed
evidence files. Use it when revising the manuscript so the GWOSC result is not
accidentally rewritten as a positive validation claim.

## Bottom line

The GWOSC follow-up is a completed negative public real-noise stress test.

The result does **not** mean the theory or implementation failed. It means the
tested GWOSC interval is not calibrated by the current global-PSD or
predeclared local-PSD noise models.

Recommended one-sentence interpretation:

```text
GWOSC shows the boundary of the present validation: real-noise calibration
requires more than a correct PSD implementation and a predeclared local PSD
model.
```

## What succeeded

- Synthetic S1-S9 validation passed.
- Stage 0 remote reproducibility passed.
- GWOSC data integrity, checksums, and official data-quality coverage passed.
- The repository Hann/median PSD estimator matched GWpy on identical windows.
- The shared-FIR implementation-equivalence follow-up passed.
  - Explicit FFT convolution and GWpy convolution agree when both paths use the
    same FIR coefficients and score normalization.
  - This resolves the shared-statistic software-identity question.
- The time-local PSD synthetic control passed.
  - This argues against a generic normalization bug in the local-PSD diagnostic.
- Follow-up evidence, derived CSV summaries, and figures were synchronized into
  `transfer_paper/`.

## What did not succeed

- The original global-PSD GWOSC held-out null calibration failed.
- The predeclared primary 64-second local PSD model failed real-data
  acceptance for both detectors.
- Local PSD did not repair the GWOSC calibration failure in this run.
- The current evidence does not support:
  - calibrated GW150914 significance;
  - validated nominal SNR-eight injection sensitivity;
  - a general claim that the method is calibrated on GWOSC data;
  - tuning PSD radius, thresholds, windows, or cuts into a pass.

Key synchronized follow-up numbers:

| Result | Status | Notes |
| --- | --- | --- |
| Shared-FIR synthetic identity | passed | max abs difference `4.98e-15` |
| Shared-FIR H1 identity | passed | max abs difference `1.60e-12` |
| Shared-FIR L1 identity | passed | max abs difference `1.45e-12` |
| Local PSD synthetic control | passed | primary local-64 score std `1.148` |
| Local PSD H1 real data | failed | primary local-64 score std `9.424` |
| Local PSD L1 real data | failed | primary local-64 score std `8.827` |

The predeclared real-data acceptance interval was `[0.8, 1.2]`.

## Manuscript-safe wording

Use wording like this in Results or Discussion:

```text
As a public real-noise stress test, we applied the GWOSC workflow to a
GW150914-centered interval. The PSD estimator matched GWpy on identical
calibration windows, and a shared-FIR follow-up showed that explicit FFT
convolution and GWpy convolution agree when given identical FIR coefficients
and score normalization. However, the held-out real-noise calibration gate
failed under both the original global-PSD model and the predeclared primary
64-second local-PSD model. The stationary synthetic control for the local-PSD
experiment passed, so the failure is best interpreted as real-noise/model
inadequacy for this interval rather than a generic normalization error. We
therefore retain GWOSC event and injection scores as diagnostics only and make
no calibrated GWOSC significance or sensitivity claim.
```

Short abstract/conclusion-safe version:

```text
The controlled experiments validate the linear likelihood-geometry claims,
while a public GWOSC stress test exposes the limits of simple global or local
PSD calibration on real nonstationary data.
```

## Wording to avoid

Do not write:

- "The method detects GW150914 with calibrated significance."
- "The SNR-eight injection sensitivity is validated on GWOSC."
- "The local PSD model fixed the GWOSC calibration."
- "GWOSC validates the method on public real data."
- "The original PSD-domain GLS statistic is mathematically equivalent to the
  finite-FIR statistic."

The shared-FIR result is narrower: it proves identity between two
implementations of the **same shared FIR statistic**, not equivalence between
all filtering/statistic definitions.

## Decision for current experiment

Freeze the current GWOSC result. Do not continue editing this experiment to
make it pass.

Allowed next actions:

- integrate the negative GWOSC result into the manuscript as a limitation and
  boundary result;
- move positive real-data validation to a detector-pulse dataset better aligned
  with the paper, such as CRESST;
- propose a new predeclared GWOSC Stage 3b.

Not allowed for the current evidence:

- changing the primary PSD radius after seeing results;
- changing thresholds, windows, quality cuts, or exclusions to obtain a pass;
- replacing the failed primary result with a sensitivity-analysis radius;
- presenting event or injection diagnostics as calibrated inference.

## If GWOSC is continued later

Any further GWOSC work should be a new predeclared Stage 3b, for example:

- longer off-source interval with more background windows;
- stricter official veto/category handling;
- robust PSD or shrinkage model with frozen hyperparameters;
- explicit glitch or template-projected outlier model;
- larger background for false-alarm calibration;
- untouched confirmatory interval or event set.

The current negative result should remain the baseline, not be overwritten.

## Evidence locations

Primary follow-up records:

- `transfer_paper/data/gwosc/followup/filter_equivalence.json`
- `transfer_paper/data/gwosc/followup/filter_equivalence.config.yaml`
- `transfer_paper/data/gwosc/followup/time_local_noise.json`
- `transfer_paper/data/gwosc/followup/time_local_noise.config.yaml`

Derived summaries:

- `transfer_paper/data/derived/gwosc_filter_equivalence_summary.csv`
- `transfer_paper/data/derived/gwosc_time_local_psd_summary.csv`
- `transfer_paper/data/derived/gwosc_time_local_psd_blocks.csv`
- `transfer_paper/data/derived/gwosc_time_local_psd_spectral_summary.csv`
- `transfer_paper/data/derived/claim_status.csv`
- `transfer_paper/data/derived/paper_implications.csv`

Figures:

- `transfer_paper/figures/gwosc_filter_equivalence.png`
- `transfer_paper/figures/gwosc_time_local_psd.png`

## Proposed documentation structure

The repo currently mixes current guidance, historical planning, manuscript
transfer artifacts, and archived notes. Before deleting anything, separate docs
by lifecycle:

```text
docs/
  README.md                         # doc map and "read these first"
  current/
    validation_status.md            # current project status, one source
    paper_revision_guideline.md     # claim-safe writing guidance
    gwosc_negative_result.md        # this document or a moved copy
    next_steps.md                   # current action queue
  protocols/
    gwosc_filtering_local_psd.md    # predeclared GWOSC protocol
    preprocessing_contract.md
    metrics.md
    remote_runbook.md
  registries/
    experiment_registry.md
    evidence_map.md
    figure_table_map.md
  archive/
    2026-06-13_*.md
    stale_plans/
```

Keep `transfer_paper/` as a generated or semi-generated paper-revision bundle,
not as the canonical place for project status. The canonical status should live
under `docs/current/`; `transfer_paper/` should consume evidence and produce
paper-facing tables, figures, notebooks, and writing briefs.

## Cleanup rules

Use these rules before deleting or moving docs:

1. Classify each document as one of: current, protocol, registry, generated,
   historical, or obsolete duplicate.
2. If historical but potentially useful, move to `archive/` with a short header
   saying which current doc supersedes it.
3. If generated, mark it as generated and identify the script that refreshes it.
4. If two docs conflict, make one canonical and add a pointer from the other.
5. Do not delete any doc that contains unique commands, thresholds, dataset
   provenance, or interpretation rules until that content is copied into a
   canonical current/protocol/registry file.
6. Add a `docs/README.md` table listing every canonical doc and every archived
   document's replacement.

## Transfer-paper versus submodule

`transfer_paper/` is useful as a local paper-revision pipeline because it
collects evidence, derived CSVs, figures, notebooks, and writing instructions in
one place. Keep it if the paper revision is tightly coupled to this repo's
experiment outputs.

Do not make it a submodule unless there is a strong need for independent
versioning, separate access control, or reuse across multiple repositories.
Submodules add operational friction and are easy to desynchronize.

Recommended structure:

- keep `transfer_paper/` in this repo for now;
- treat `transfer_paper/data/derived/` and `transfer_paper/figures/` as
  reproducible outputs from scripts where possible;
- keep paper-source files, if any, in a separate `paper/` directory or separate
  private repository;
- export a frozen paper evidence bundle at revision milestones.

If the manuscript grows into an independent project, a better split than a
submodule is often:

- experiment repo: code, configs, raw evidence manifests;
- paper repo: manuscript source plus a pinned exported evidence bundle;
- no live submodule dependency unless automatic cross-repo synchronization is
  genuinely needed.
