<!-- GENERATED COPY - DO NOT EDIT.
     Source: docs/PAPER_REVISION_GUIDE.md @ 1a268459e52b4e22d9235cb141ce2ebd9110d579
     Generated: 2026-07-26T17:30:29Z by scripts/sync_source_documents.py
     Edit the source in the experiment repository and re-run the sync. -->

# Paper Revision Guide

Last consolidated: 2026-07-12.

## Revision Goal

Revise the paper so the claims match the evidence:

- positive claims: theory, controlled synthetic validation, and the completed
  public CRESST-A within-channel reconstruction result;
- GWOSC: negative public real-noise stress test and limitation;
- CRESST-A: positive public real-data validation of the narrow linear hierarchy,
  with no cross-channel-transfer or detection-sensitivity claim.

## Submission-Safe Claim Shape

The main positive statement should be:

```text
Under the stated linear Gaussian model, the noise covariance defines the
reconstruction metric, and OF, EMPCA, and weighted tied linear autoencoders are
constrained maximum-likelihood projections in that metric. Controlled
synthetic experiments validate the equivalences and expose the loss-geometry
failure of isotropic MSE under colored noise.
```

The GWOSC statement should be:

```text
The GWOSC stress test passed PSD/reference and shared-FIR implementation
checks, but failed held-out real-noise calibration under both global and
predeclared local PSD models. We therefore treat GWOSC event and injection
scores as diagnostics only.
```

## Abstract And Conclusion Wording

Short version:

```text
Controlled experiments validate the linear likelihood-geometry claims, and a
public CRESST-A study supports the within-channel linear hierarchy on held-out
pulses. A public GWOSC stress test separately exposes the limits of simple
global or local PSD calibration on real nonstationary data.
```

Expanded version:

```text
Controlled simulations verify the OF/EMPCA and EMPCA/linear-autoencoder
equivalences and demonstrate that MSE can select the wrong geometry under
colored noise. On public CRESST-A pulses, rank-6 EMPCA beat a single-template
optimal filter and met the predeclared multi-template competitiveness gate
under per-channel chronological splits. A public GWOSC stress test provides an
important boundary case:
although PSD/reference and shared-FIR implementation checks pass, held-out
real-noise calibration fails under both global and predeclared local PSD
models. We therefore make no calibrated GWOSC significance or sensitivity
claim.
```

## Claim Audit

| Claim | Support level | Allowed wording | Avoid |
| --- | --- | --- | --- |
| OF/EMPCA/weighted-AE unified by noise metric | supported by theory and controlled validation | "unified as constrained ML projections under the noise metric" | "validated for all detector data" |
| S1-S9 synthetic gates | verified positive | "passed multi-seed controlled criteria" | "proves real-detector calibration" |
| S5 colored-noise metric reversal | verified positive, frozen 10-seed plus confirmatory 30-seed sweep | State the weighted-residual and raw-residual PCA/EMPCA ratios only from the confirmatory sweep's JSON `ratio` block (e.g. "EMPCA's weighted residual was X times lower / PCA's raw residual was Y times lower"); report both orderings explicitly | Never derive a "x times" figure from a raw difference or a percentage; never call `48.49` (a raw weighted-residual value, not a ratio) a "times" figure |
| Remote reproducibility | verified positive | "Stage 0 passed on a clean remote checkout" | "reproducibility implies scientific acceptance" |
| GWOSC PSD implementation | verified positive | "matches GWpy on identical windows" | "PSD agreement validates likelihood calibration" |
| Shared-FIR identity | verified positive | "FFT-convolution and GWpy-convolution agree for the shared FIR statistic" | "all GWpy and repository statistics are equivalent" |
| GWOSC global PSD calibration | verified negative | "failed the predeclared held-out gate" | "event score is calibrated" |
| GWOSC local PSD calibration | verified negative | "synthetic control passed, real H1/L1 failed" | "64-second local PSD fixed calibration" |
| GW150914 significance | not validated | "diagnostic score only" | "detection/significance claim" |
| CRESST-A public validation | verified positive | "within each chronologically split channel, rank-6 EMPCA beat single-template OF and met the multi-template-OF competitiveness gate; pooled paired-bootstrap CIs excluded zero" | "cross-channel transfer" or "validated detection sensitivity" |
| Study A noise-budget decomposition | verified positive, rebuilt from authoritative component files | "SQUID, Johnson, TD, and Er component PSDs from noise_module/Al2O3_Al_athermal/ dominate/cross as reported in results/noise_budget/<run_id>/study_a/" using exactly those source labels | "1/f phonon", "vibrational", "TES Johnson", or any physical relabeling not recorded in provenance; any crossover not independently bracketed by a sign change |
| Study B sensitivity attribution | verified positive, conditional on declared PSD/acquisition assumptions | "removing/scaling component X changes OF amplitude sensitivity by factor Y under acquisition branch Z (inband_no_alias / unfiltered_alias_fold)" | stating a sensitivity ratio without naming its acquisition branch; calling this a unique physical noise decomposition or a measured detector improvement |

## Sections That Usually Need Revision

1. Abstract: replace the stale CRESST placeholder with the narrow positive
   within-channel result and its boundary.
2. Contributions: include CRESST reconstruction validation, but avoid transfer
   or detection-sensitivity claims.
3. Results: separate synthetic and CRESST positive results from the GWOSC
   negative stress test.
4. Discussion: explain GWOSC as a boundary case and motivation for richer
   real-noise modelling.
5. Limitations: state that real detector calibration may need vetoes, larger
   backgrounds, robust PSD/shrinkage, explicit outlier models, or a new
   predeclared GWOSC Stage 3b.

## Results Ordering

Use this order:

1. theory/equivalence checks;
2. controlled synthetic S1-S9 gates;
3. CRESST-A within-channel reconstruction validation;
4. GWOSC implementation/reference checks;
5. GWOSC failed global/local real-noise calibration;
6. limitations and next validation domain.

This prevents the GWOSC negative result from appearing to undermine the
synthetic theory while still reporting it honestly.

## Transfer-Paper Workflow

`transfer_paper/` is the manuscript-support bundle. It contains synchronized
evidence, derived CSVs, figures, notebooks, and writing briefs.

Keep canonical project status in `docs/`. Use `transfer_paper/` for paper
revision artifacts and generated evidence products. Do not make
`transfer_paper/` a submodule unless independent versioning or access control
becomes necessary.
