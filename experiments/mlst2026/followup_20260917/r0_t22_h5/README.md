# R0 — T22 H5 random-effects reporting overlay

Status: complete, locally reproduced; ready to transfer; zero new cells.

## Question

What are the omitted T22 H5 random-effects coverage and interval-length
summaries, beside the original two-stage and equal-file t intervals?

## Design

- Type: reporting correction. No protocol, no seed namespace, **no new cells**.
- Input: copies of the stored T22-S `cells/` records and `REDUCED.json`.
- Output: an additive overlay report and a manuscript-facing table.
- Scope: the 65 non-drift scenarios (H1/H5 scope), the four drift stress
  scenarios separately, and all 69 descriptively.
- Estimand: the declared population equal-file mean (`true_mu`), exactly as the
  frozen reducer uses it.
- Targets: random-effects (precision-weighted file-effect mean) vs two-stage
  (equal-file mean) are labelled, and any displacement is flagged as an
  exploratory finite-sample estimator displacement rather than scored as a
  better or worse estimator.

## Files

- `reducer.py` — the overlay (`build_report`, `summarise_records`, `verify`,
  `structured_diff`, `manuscript_table`).
- `evidence_export/` — the generated correction:
  `H5_RANDOM_EFFECTS_REDUCTION.json`, `MANUSCRIPT_T22_H5_TABLE.md`,
  `STRUCTURED_DIFF.json`, `PROVENANCE.json`, and `frozen/` copies.
- `MANUSCRIPT_HANDOFF_R0.md` — the handoff for the manuscript agent.

## Command

```sh
python -m experiments.mlst2026.followup_20260917.r0_t22_h5 reduce \
  --run-root <archived t22s run root> --copy-to <scratch>/r0_copy --out evidence_export
```

## Tests

`tests/test_followup_20260917_r0.py` (9 tests): Wilson/empty-denominator,
coverage-from-stored-bounds, interval-bound perturbation, wrong-contrast marker,
not-withheld denominator, target-mismatch flag, manuscript table scopes, verify
omission, and a gated integration check of the committed export.

## R0-v2 (17 September 2026)

The canonical `evidence_export/` holds the regenerated **R0-v2** overlay:
non-destructive and path-safe, inventory/hash-validated, coverage recomputed
from stored bounds, and the estimator-displacement statistic labelled
exploratory. The original **R0-v1** export is retained unchanged under
`evidence_export_v1_retained/`; see `MANUSCRIPT_HANDOFF_R0.md` for why the
manuscript must read v2's qualification of the 29/65 figure.
