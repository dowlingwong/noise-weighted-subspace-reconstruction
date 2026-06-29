# Paper Revision Guide

Last consolidated: 2026-06-29.

## Revision Goal

Revise the paper so the claims match the evidence:

- positive claims: theory plus controlled synthetic validation;
- GWOSC: negative public real-noise stress test and limitation;
- CRESST or another detector-pulse dataset: next candidate for positive
  public real-data validation.

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
The controlled experiments validate the linear likelihood-geometry claims,
while a public GWOSC stress test exposes the limits of simple global or local
PSD calibration on real nonstationary data.
```

Expanded version:

```text
Controlled simulations verify the OF/EMPCA and EMPCA/linear-autoencoder
equivalences and demonstrate that MSE can select the wrong geometry under
colored noise. A public GWOSC stress test provides an important boundary case:
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
| Remote reproducibility | verified positive | "Stage 0 passed on a clean remote checkout" | "reproducibility implies scientific acceptance" |
| GWOSC PSD implementation | verified positive | "matches GWpy on identical windows" | "PSD agreement validates likelihood calibration" |
| Shared-FIR identity | verified positive | "FFT-convolution and GWpy-convolution agree for the shared FIR statistic" | "all GWpy and repository statistics are equivalent" |
| GWOSC global PSD calibration | verified negative | "failed the predeclared held-out gate" | "event score is calibrated" |
| GWOSC local PSD calibration | verified negative | "synthetic control passed, real H1/L1 failed" | "64-second local PSD fixed calibration" |
| GW150914 significance | not validated | "diagnostic score only" | "detection/significance claim" |
| CRESST public validation | not completed | "planned/next independent domain" | "validated on CRESST" |

## Sections That Usually Need Revision

1. Abstract: remove any implication of positive public real-data validation.
2. Contributions: keep controlled validation, avoid real-data performance
   claims.
3. Results: separate synthetic positive results from GWOSC negative stress
   test.
4. Discussion: explain GWOSC as a boundary case and motivation for richer
   real-noise modelling.
5. Limitations: state that real detector calibration may need vetoes, larger
   backgrounds, robust PSD/shrinkage, explicit outlier models, or a new
   predeclared GWOSC Stage 3b.

## Results Ordering

Use this order:

1. theory/equivalence checks;
2. controlled synthetic S1-S9 gates;
3. GWOSC implementation/reference checks;
4. GWOSC failed global/local real-noise calibration;
5. limitations and next validation domain.

This prevents the GWOSC negative result from appearing to undermine the
synthetic theory while still reporting it honestly.

## Transfer-Paper Workflow

`transfer_paper/` is the manuscript-support bundle. It contains synchronized
evidence, derived CSVs, figures, notebooks, and writing briefs.

Keep canonical project status in `docs/`. Use `transfer_paper/` for paper
revision artifacts and generated evidence products. Do not make
`transfer_paper/` a submodule unless independent versioning or access control
becomes necessary.
