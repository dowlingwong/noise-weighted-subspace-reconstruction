# R0 — T22 H5 random-effects reporting correction (handoff)

Status: **complete — locally reproduced with the R0-v2 reducer; ready to transfer
(zero new cells, no production run required).** The R0-v1 export is retained
unchanged under `evidence_export_v1_retained/`; the current export in
`evidence_export/` is R0-v2 (copy-destination guards, input inventory/hash
validation, coverage recomputed from stored interval bounds rather than trusted
flags, and the displacement statistic relabelled exploratory). Headline numbers
are identical between the two versions.

The archived T22-S reduction computed the random-effects interval inside every
Monte-Carlo record but never summarised it, so H5's second secondary interval was
absent from `REDUCED.json`. R0 is a versioned, audited reporting overlay that
recomputes the H5 summary **from copies of the stored records** and writes it
beside the frozen reduction. It does not re-run, repair or re-tune anything.

## What the manuscript agent should consume

Read only these files (they are immutable once generated):

- `evidence_export/MANUSCRIPT_T22_H5_TABLE.md` — the manuscript-facing table.
- `evidence_export/H5_RANDOM_EFFECTS_REDUCTION.json` — the machine-readable
  report (schema `mlst-followup-r0-t22-h5/1`).
- `evidence_export/PROVENANCE.json` — source hashes, the exact command, the
  verification results.
- `evidence_export/frozen/` — byte-identical copies of the frozen
  `REDUCED.json`, `protocol_snapshot.json` and `FROZEN_CALIBRATION.json`.

Do **not** edit these; a change is a new amendment version, not an edit.

## Headline numbers (recomputed, not hard-coded)

| scope | n | RE in band | RE coverage median (range) | two-stage median | t median | median RE/two-stage length |
|---|---|---|---|---|---|---|
| non-drift (H1/H5 scope) | 65 | **24** | **0.876** (0.427–0.998) | 0.959 | 0.932 | 0.636 |
| drift stress (separate) | 4 | 0 | 0.822 (0.817–0.842) | 0.878 | 0.944 | 0.702 |
| all scenarios (descriptive) | 69 | 24 | 0.871 (0.427–0.998) | 0.958 | 0.933 | 0.636 |

These reproduce the audit's supplementary values (24/65; median 0.876) exactly.
The check `scenarios_differing == 0` confirms the two-stage coverage recomputed
from the stored records agrees with the frozen `REDUCED.json` in all 65 non-drift
scenarios.

## New caveat the overlay reports: an exploratory estimator displacement

The random-effects interval is a DerSimonian–Laird inverse-variance mean of the
per-file effects. Its declared target is the **precision-weighted** file-effect
mean, whereas the primary two-stage interval targets the **equal-file mean**;
under heterogeneous file noise these targets need not coincide. The overlay
measures the displacement `re_estimate − equal_file_mean` per scenario with its
Monte-Carlo standard error and flags it when it is resolved beyond Monte-Carlo
error and exceeds 20% of the mean RE half-width. **29 of the 65 non-drift
scenarios are flagged.** Per the R0-v2 correction this statistic is an
**exploratory finite-sample estimator displacement on these replicates**: it
establishes neither a population estimand mismatch nor the cause of the RE
undercoverage, and it never declares the RE interval a better or worse estimator
from coverage alone. Any manuscript mention must carry that exploratory label.

## Exact command

```sh
python -m experiments.mlst2026.followup_20260917.r0_t22_h5 reduce \
  --run-root remote-return/mlst-successor-closeout-20260917/extracted/ceph/dwong/mlst-successor-runs-20260913/t22s \
  --copy-to <scratch>/r0_copy_t22s \
  --out experiments/mlst2026/followup_20260917/r0_t22_h5/evidence_export
```

The `--copy-to` is required for immutability: the overlay reads the copy and the
original archive is never opened for writing. Running it again on the same copy
is deterministic.

## Validation and checks

- `python -m ...r0_t22_h5 reduce` exits 0 only if all four verification checks
  pass: non-drift count 65, RE in band 24, median 0.876 (±0.001), and zero
  two-stage disagreements with the frozen reduction.
- `tests/test_followup_20260917_r0.py` includes mutation tests that fail if the
  summary is omitted, if the wrong contrast or denominator is used, or if the RE
  interval bounds are perturbed.
- Source hashes (recorded in `PROVENANCE.json`):
  - frozen `REDUCED.json` `55c96c4b40e4631c5da16650394f92a1516a3ce95f82fe5629498840424d916c`
    (equals the locally checked transfer hash in `LOCAL_RESULT_CHECKS.json`);
  - protocol snapshot `4c5c43526929db378e226f351b0ebbefff786344c7ffba396efedfdb0583aae9`;
  - `FROZEN_CALIBRATION.json` `946656fc7f191a6802909c1fb90f58e77811f21d12cefe1d289678c527f5dd79`.

## Authorized use, and non-framing (mandatory)

1. The random-effects 24/65 in-band count and median coverage ≈ 0.876 (verified
   from the stored records, never hard-coded) are **not** evidence that the
   random-effects interval is a better estimator. Coverage below nominal is a
   **failure**. The 0.876 median is a descriptive secondary-hypothesis statistic,
   not a success.
2. **Per-scenario displacement must be labelled wherever it is flagged.**
   The overlay flags 29 of the 65 non-drift scenarios where the random-effects
   (precision-weighted) estimate is displaced from the equal-file point beyond
   Monte-Carlo error. This is an exploratory finite-sample estimator
   displacement, not an established population estimand mismatch and not an
   established cause of the lower RE coverage. Any manuscript use of a
   random-effects number must carry this scope label.
3. **No random-effects success framing is authorized** in the abstract, the main
   text, or any summary claim. Only the descriptive, scope-labelled summary in
   `evidence_export/MANUSCRIPT_T22_H5_TABLE.md` may be used. The frozen H1
   verdict (13 of 65 non-drift scenarios outside the coverage band) is unchanged
   and is not repaired or softened by this correction.

## What R0 does **not** change

- The frozen `REDUCED.json` (copied byte-for-byte).
- The H1 verdict (13 of 65 non-drift scenarios outside the coverage band).
- Any archived T22 verdict (all withheld).
- Any old cell, freeze, ledger, manifest or reduction.

## Round-2 addendum (17 September 2026) — consume `R0-v2`, not `R0-v1`

The canonical `evidence_export/` now contains the regenerated **R0-v2** export
(2026-09-17, identical headline values); the original **R0-v1** export is
retained unchanged under `evidence_export_v1_retained/`. R0-v2 supersedes R0-v1
for the *interpretation* of the displacement statistic: the R0-v1 "target
mismatch" reading must **not** be forwarded to the manuscript as established
evidence.

What `R0-v2` changes:

- **Non-destructive and auditable.** The reducer copies to a fresh `--copy-to`
  destination (resolved, with equality/ancestor/descendant/existing guards) and
  never deletes; it validates the input cell inventory (unique ids, scenario set,
  freeze binding, complete reconciliation) and records input-cell, freeze and
  reducer-source hashes in `PROVENANCE.json`.
- **Coverage recomputed from bounds.** Every two-stage/t coverage flag is
  recomputed from the stored interval bounds and the declared truth, not read
  from `covers`/`t_covers`.
- **Displacement is exploratory.** The 29/65 figure is a *finite-sample
  estimator-displacement* count (`RE estimate − equal-file point`), not an
  established population estimand mismatch and not evidence of the cause of the
  coverage shortfall. `target.establishes_population_mismatch` and
  `target.establishes_cause_of_undercoverage` are both `false`; the population
  targets and assumptions are stated separately.
- The audited values are unchanged and re-derived: **24/65 non-drift in band,
  median 0.876**, zero two-stage disagreements with the frozen reduction.

Exact command:

```sh
python -m experiments.mlst2026.followup_20260917.r0_t22_h5 reduce \
  --run-root <verified t22s run root> --copy-to <fresh scratch dir> \
  --out experiments/mlst2026/followup_20260917/r0_t22_h5/evidence_export
```
