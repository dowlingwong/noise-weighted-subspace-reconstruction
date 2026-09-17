# R6 — instrument readiness (CRESST robustness + matched TIDMAD design)

**Status: blocked on data.** This package prepares interfaces, inventory checks and
protocols. It executes no production cell, opens no real instrument file, and claims
no executed evidence and no real-data validation. Every data-dependent entry point
fails closed with a clear refusal (`DATA-BLOCKED`) when its inputs are absent or fail
provenance. The estimator wiring is exercised on a synthetic fixture and is always
labelled `synthetic-fixture, not real data, not validated on real inputs`.

* **Interface implemented and fail-closed** — the CRESST grid, the input gates, the
  matched design, the inventory and the block logic all run and refuse cleanly.
* **Not validated on data** — no CRESST raw pulse and no independent TIDMAD hold-out
  file exists in this environment; the release-layout adapters are deliberately
  unwritten because there is no raw data to test them against. No data was
  downloaded, sought or manufactured.

New protocol `PROTOCOL_R6_INSTRUMENT_v1.json` (`mlst-local-package/1`, package
`r6_instrument`) superseding `successor_20260913/cresst_s/PROTOCOL_CRESST_S_v1.json`
and `successor_20260913/t22s/PROTOCOL_T22S_v2.json`. It was designed after the
September 2026 results were observed. The old trees are immutable and are imported
unchanged.

## What it prepares

### CRESST: the archived 350-cell robustness grid

* The rank × shrinkage × time-cut grid (35 pilot + 315 confirmation = **350 cells**,
  **1050 estimator fits** if executed) is enumerated **from the immutable archived
  design** (`robustness_design.grid_rows`) and compared field by field with the R6
  protocol: channels and order, primary setting, both arms, EM restarts and fit
  accounting. Pairing is preserved exactly (28 contrast groups; every non-reference
  setting paired against its group's rank-6/shrinkage-0.1 reference; 336 paired
  contrasts total).
* Every cell is marked `executable_here: false` and names the missing input. No cell
  is fabricated or run.
* Raw-pulse provenance/split gates: a missing root, a missing/malformed index, a
  missing array, non-chronological `times`, a declared/actual count mismatch, a
  channel-set that differs from the archived seven, an absent provenance block, or a
  provenance/split value that differs from the declaration all refuse with
  `InputNotAvailable`.
* **Runs, not channels.** Seven channels from two runs are correlated clusters. Four
  runs is the existing minimum for reporting a variance component — not a guarantee
  of useful coverage or power — and more channels cannot substitute for runs. Any
  run-aware interval is a **declared, optional, separately calibrated** study and is
  not run (protocol `cresst.run_aware_interval`).

### TIDMAD: matched whole-window vs narrowband

* The matched design is declared: the **same independent, unexposed files**, the same
  calibration and evaluation units (`cal_read`, `cal_ref`, `fit_read`, `eval_read`,
  `eval_ref`), rank 2, one transfer calibration, one label array and one scorer; the
  representation is the only difference. The band is an input selected on development
  data in a separate step, not part of matched scoring.
* Two named contrasts are produced on **identical evaluation units**:
  1. the **matched weighted-residual** contrast — the held-out whitened observation
     residual `|q − x|² / N_k`;
  2. the **truth-referenced** contrast — the held-out squared error against the
     injected reference `t_eval = cal.g · eval_ref[:, k]`.
  Their sign agreement is the P3 test, reported **separately** from usefulness
  against the zero, calibration-mean and raw-bin baselines.
* Controls: the non-learned **B = 1 identity** (band-PCA = band-EMPCA = bin read,
  relative 1e-12), a **covariance-adequacy** diagnostic, and an arithmetic check of
  the identity `R_clean,Σ = R_obs + 2 tr(B) − d` for fixed equal-trace maps. The
  identity is applied only where its assumptions hold; a physical endpoint remains
  empirical.
* **Confirmatory gate:** the exposure inventory is reused from
  `matched_stage.exposure_inventory`. The archive does **not** contain the release
  file list, so the set of never-opened files cannot be resolved here and
  confirmatory execution is blocked. Missing files, an incomplete/infeasible
  simulation calibration, and an inconclusive development dependence verdict each
  block as well. No new window cut from a previously examined file may be represented
  as a new independent file.

## Subcommands

| command | what it does | opens real data |
|---|---|---|
| `validate` | load the protocol, re-enumerate the archived grid, confirm the grid matches; prints `DATA-BLOCKED` status | no |
| `enumerate` | write `R6_ENUMERATION.json` (350 CRESST cells + matched design) | no |
| `inventory` | write the TIDMAD exposure inventory from the archived run trees | no |
| `check-inputs` | per-input readiness table; exit 3 `DATA-BLOCKED` when inputs are absent | no |
| `preflight` | confirmatory readiness verdict; exit 3 `DATA-BLOCKED` when not ready | no |

Exit codes: `0` ok, `3` DATA-BLOCKED refusal, `6` protocol/usage error.

## Local verification (actually run)

```
.venv/bin/pytest tests/test_followup_20260917_r6.py -q
.venv/bin/python -m experiments.mlst2026.followup_20260917.r6_instrument validate
.venv/bin/python -m experiments.mlst2026.followup_20260917.r6_instrument enumerate --out /tmp/r6_enum
.venv/bin/python -m experiments.mlst2026.followup_20260917.r6_instrument inventory
.venv/bin/python -m experiments.mlst2026.followup_20260917.r6_instrument check-inputs    # exit 3
.venv/bin/python -m experiments.mlst2026.followup_20260917.r6_instrument preflight       # exit 3
```

Because no real data exists, `check-inputs` and `preflight` are **expected** to exit
`3` with a `DATA-BLOCKED` status.

## What would unblock it

1. **CRESST raw pulses.** The actual CRESST release pulse traces written into the
   `cresst-s-pulse-root/1` interface (see `SCHEMA` in
   `cresst_readiness.py`), with the release-layout adapter written *and validated on
   real inputs* on a host that holds the release, plus a provenance block carrying
   the release `archive_id` and the declared evaluation `split`.
2. **TIDMAD independent files.** The release file list, from which the files the
   archive never opened can be named, plus a complete, in-tolerance confirmation
   calibration and a non-inconclusive development dependence gate. A never-read
   region of an already examined file is not sufficient.
3. **Run-aware inference (optional).** If a population/new-run claim is retained, an
   independent run-aware interval study with independent runs (four is the minimum
   for a variance component) separately calibrated and frozen.

Only then does the matched CRESST / TIDMAD work leave the **blocked on data** state.
Until then every data-dependent path refuses.

## Explicitly not claimed

* no real-data result and no executed evidence;
* no validated adapter (synthetic fixtures exercise wiring only);
* no executed CRESST robustness (the grid is enumerated and costed only);
* no new-run / population coverage and no run-aware interval;
* no TIDMAD instrument verdict and no change to the archived T22 verdicts;
* no P3 confirmation (exercised on a synthetic fixture only);
* no physical-parameter ordering from the whitened-risk identity.

## A4 — committed machine-readable inventory and refusal record

- `INPUT_READINESS_INVENTORY.json` (+ `.md`): the TIDMAD exposure inventory from
  the locally archived records; every region names its source run root and file,
  and the release file list is explicitly non-derivable.
- `INPUT_GATE_RECEIPT.json`: a machine-readable refusal record
  (`status: DATA-BLOCKED`) covering the CRESST raw-pulse input check and the
  TIDMAD independent-file/calibration gate.

Regenerate with `inventory --out ...` and `check-inputs > ...`; both are
fail-closed and no real-data validation is claimed.
