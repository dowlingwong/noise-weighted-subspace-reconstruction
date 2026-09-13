# E12 run 1 — FAILURE RECORD. NOT SCIENTIFIC EVIDENCE.

**Nothing in this directory is validated scientific evidence, and none of it may
be cited, plotted, reduced, or used to support any claim.** It is preserved only
as the artifact record of a run that failed closed, so the failure stays
auditable.

The run's own integrity gate **rejected** these rows. They were never validated,
and the sweep they belong to was never completed.

| | |
|---|---|
| Experiment | `E12_calibration_size_sweep` (post-hoc exploratory, not confirmatory) |
| Outcome | `E12_SWEEP_PARTIAL_WITH_RECORDED_FAILURES` |
| Run window | 2026-08-23T20:44:39+02:00 → 2026-08-24T03:27:48+02:00 (6 h 43 m) |
| Host | `deepthought` (40 logical CPUs / 20 physical cores) |
| Pinned tree | content `75cbc403…`, tree `a4e44220…`, freeze `0c5e3095…` |
| Conditions completed | **0 of 5** |
| `sweep_summary.json` | **never written** |

## Why these rows are not evidence

`n_cal_256` computed all ten seeds, and the condition gate then refused the
runner's own output:

```
IntegrityError: E12_calibration_size_sweep[n_cal_256]: seed 0 analysis_label mismatch
```

The rows carry `analysis_label = single_condition_covariance_estimation_sensitivity`
(E10b's label) where the sweep's gate requires
`post_hoc_exploratory_calibration_size_sweep`. The defect is described in
`docs/plans/rps2026/E12_RUN_1_FAILURE.md` and was fixed in commit `ccd329f`.

Because the gate never passed, **none** of the sweep's scientific outputs exist:
no per-condition degradation means, no NFPA-minus-Iso contrasts, no
`trend.*.monotone_decreasing` flags, and no `reversal_sign_holds_everywhere`.
Any such number attributed to this run would be fabricated.

## Handling rules

* **Do not relabel these shards.** They are preserved exactly as written; the
  wrong label is the evidence of the defect. Their digests are in
  `CHECKSUMS.sha256`.
* **Do not resume this run**, and do not resume it under a different freeze
  commit. It is pinned to a superseded tree.
* **Do not re-run the old dispatch prompt pinned to `75cbc403`.** The fix in
  `ccd329f` post-dates it.
* A future E12 run is a new run against a current freeze, not a continuation of
  this one.

## The one partial observation, stated carefully

A shard is only written after that seed's isotropic control reproduces archived
E1, so per-seed control reproduction passed at `n_cal = 256`. The
cross-condition control gate never ran. Covariance-independence is therefore
**unestablished, not disproven** — which is not a result either.

## What was not damaged

15/15 authoritative archive digests identical before and after
(`runtime/archive_before.json`, `runtime/archive_after.json`,
`runtime/archive_compare.txt` → `ARCHIVE_INTACT`). Worktree clean throughout; no
shard deleted; no tolerance altered; no `--sizes` passed; the freeze commit
always supplied.

## Contents

* `CHECKSUMS.sha256` — SHA-256 of all 55 preserved files, computed from the
  originals at `/ceph/dwong/e12_runtime_20260823_201812` and verified against
  this copy.
* `runtime/E12_calibration_size_sweep/n_cal_256/` — the 10 refused shards, their
  concatenated `raw.csv`, `manifest.json` and `shard_registry.json`.
* `runtime/Round10_smoke/` — smoke output. Smoke carries
  `evidence_class=implementation_check_not_scientific_evidence` and bypasses the
  condition gate (`if not smoke:`), which is why it passed while the real run
  could not.
* `runtime/e12.log`, `e12_resume.log`, `e12_resume_1.log` — the failure and both
  independent reproductions of it (35 s and 2 s, after reusing validated shards).
* `runtime/pytest_full.log`, `pytest_status.txt` — the complete repository suite
  that passed (450 passed, 9 skipped) while the runner could not complete.
* `runtime/machine_preflight.txt`, `chain.log`, `supervisor.log`,
  `sweep_status.txt`, `tree_state*.txt` — host state, orchestration and
  tree-cleanliness records.
* `runtime/run_*.sh` — the drivers used, preserved verbatim.

## Correction to the narrative record

`docs/plans/rps2026/E12_RUN_1_FAILURE.md` records the machine as `bms2`. The
host was `deepthought`, per `runtime/machine_preflight.txt` captured at run
time. The rest of that document matches these artifacts.
