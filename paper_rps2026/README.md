# Noise Covariance as Representation Geometry for Scientific Reconstruction

Target: NeurIPS 2026 Workshop on Representations for the Physical Sciences (R/PS).
Deadline **29 Aug 2026 AoE**; submit via OpenReview
(`NeurIPS.cc/2026/Workshop/RPS`), single PDF, double-blind, non-archival.

A four-page workshop paper distilled from Paper 1 (likelihood geometry,
OF/CW-PCA/tied-linear-AE bridge, metric reversal), Paper 2 (NFPA,
Kronecker-restricted representation geometry), and the rung-4 TIDMAD
factored-attention experiment. Main thesis: a reconstruction method is the pair
`(Σ⁻¹, F)` — the measurement likelihood fixes the residual geometry, the
architecture the admissible representation class — and whether the metric choice
bites is a measurable property of the instrument and the representation.

## Build and verify

```sh
make                      # latexmk; output paper.pdf
```

Gates every edit must preserve: **4 pages of main content** (references start on
p5), 0 overfull boxes, 0 undefined references, 0 `\PENDING`, empty PDF Author
field, and a clean anonymity sweep
(`pdftotext paper.pdf - | grep -iE 'dowling|wong|karlsruhe|kit.edu|delight|acknowledg'`).

Figure 1 is generated, never hand-edited:

```sh
cd .. && python scripts/make_rps_figure.py            # seed 1 (current paper)
python scripts/make_rps_figure.py --with-seed2 E7     # after the E7 sync
```

The template (`neurips_2026.sty`) is byte-identical to the official R/PS zip,
vendored under `vendor/`.

## Layout

| Path | What |
|---|---|
| `sections/`, `appendix/`, `tables/`, `figures/` | manuscript source |
| `plan/PLAN_00_MASTER_RESTRUCTURE_2026-08-08.md` | governing skeleton; round plans are in `plan/archive/` (submitted 2026-08-29) |
| `plan/archive/` | superseded plans |
| `notes/round8_contingency.md` | drop-in text for the E7 seed replication, **both outcomes pre-compiled and fitting** |
| `notes/anonymized_code_release.md` | code-link recipe (v2) and gate status |
| `notes/openreview_metadata.md` | paste-ready submission fields + submission-day checklist |
| `notes/claim_audit.md` | claim-by-claim status ledger (kept current) |

The anonymized code snapshot is built by `../scripts/build_anonymized_export.sh`
into `../export/` (gitignored); the anonymity grep runs inside the build and
fails it on any hit.

## State (2026-08-18)

Manuscript complete at 4 pages with zero placeholders. Waiting on: the remote
Task B seed replication (contingency text ready), the author's code-link go-live
(gates 1–4 pass except the full NFPA suite, which exceeds local sandbox limits —
run `PYTHONPATH=src:. pytest experiments/synthetic/nfpa_confirmatory/tests` on
the server), and a reviewer-verifiable source for the `priorThesis` citation.
