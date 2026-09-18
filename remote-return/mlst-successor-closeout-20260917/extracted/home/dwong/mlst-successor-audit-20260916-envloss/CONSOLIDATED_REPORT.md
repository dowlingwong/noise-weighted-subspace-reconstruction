# MLST 13 September successor programme — consolidated report

**Host:** deepthought · **Completed:** 2026-09-17 · **Commits:** experiment live repo
`1d29658d0db908a0379b805d352a41e5c8e1f53b`; T19 amendment `56de23e0eb113eaad9f970ab7fb8e45648231d88`;
manuscript `99040f8b2ca1d9d24a4cae8466fdafae231a2d85`.

**All six authorized selections are executed and validated.** Scientific review remains a
reviewer's task; nothing here edits manuscript claims. Nothing was pushed or published.

Execution/validation status and scientific findings are reported in separate sections.
A validated run is not a positive result.

---

## 1. Execution and validation status

| Selection | Cells | Attempts (charged, none refunded) | Fits vs frozen cap | Validation |
|---|---|---|---|---|
| T19-S | 136/136 | 137: 136 completed, 1 failed | 3,872 / 4,800 | Validated on copy (13 Sep); root intact by checksum |
| T20-S | 964/964 | 965: 964 completed, 1 failed | 4,805 / 5,400 | Validated on copy (13 Sep); root intact by checksum |
| T21-S | 2,499/2,499 | 2,500: 2,499 completed, 1 failed | 2,497 / 2,750 | Validated on copy (13 Sep); root intact by checksum |
| T22-S simulation | 727/727 | 728: 727 completed, 1 interrupted, 0 failed | 0 / 64 | **Validated on copy (17 Sep), 35/35 checks** |
| CRESST archive | **21 of 395 — INTENTIONALLY PARTIAL** | 21 completed | 6 / 1,400 | Validated on copy (13 Sep); root intact by checksum |
| CRESST calibration | **25 of 395 — INTENTIONALLY PARTIAL** | 25 completed | 0 / 1,400 | Validated on copy (13 Sep); root intact by checksum |

The two CRESST selections are complete **as authorized** and intentionally partial against the
395-cell whole design. Their missing families are full robustness, which is excluded.

T22's fit cap of 64 binds only the development-only matched fixture cell (24 of 64). The
production matrix plans zero fits by design. That does not make wall time a budget.
**Measured simulation time**, reported separately and gating nothing: 80.08 worker-hours,
median 323 s per cell.

### T22 terminal status

* **Launcher:** success line `Execution/reduction finished` present; zero tracebacks. The numeric
  exit code is not observable (the launcher was not a child of this session), but `command()`
  raises on any non-zero phase and the line prints only after all phases.
* **Final reduction:** 2026-09-16 23:43:39–23:44:45. `03-reduce.log`: `partial: false`,
  reconciliation 727/727 with nothing missing, extra, duplicated, unattributed or misnamed.
  `03-REDUCED.json` is byte-identical to the root `REDUCED.json`.
* **`T22_TERMINAL_MANUAL.json`:** `VALIDATED_COPY_REDUCTION_SCIENTIFIC_REVIEW_PENDING`, 35/35
  checks, covering:
  * no live process
  * supplied manifest verified; environment equals receipt
  * 746-file byte-identical copy
  * re-reduction exit 0, reproducing exactly apart from the relocated ledger path
  * source unchanged
  * freeze schema, freeze non-development, freeze from exactly the 32 completed pilot cells
  * freeze written before the first confirmation cell
  * **all 690 confirmation cells bound to freeze `946656fc…`**
  * protocol hash `30acaa43…` identical in manifest, loaded protocol, `REDUCED.json`, freeze,
    every ledger record and every cell
  * matrix hash `ee2982e7…` identical in manifest, `REDUCED.json`, freeze and every cell;
    727 matrix rows
  * no running or failed attempts; retry limit respected (max 2 attempts on any cell);
    fit cap respected
  * all five protocol-declared controls present and passed
* **Validation run 1 did not pass** and is retained as `T22_TERMINAL_MANUAL.run1_STOP_REVIEW_REQUIRED.json`.
  Cause: a validator defect. It looked for T19's `scientific_checks` key, which T22's reduction
  does not have. The check was replaced with a protocol-declared control-set check and a
  per-cell freeze binding check, and run 2 passed. No production artifact changed between runs.

### Completed-root integrity (by checksum, not timestamps)

All six completed roots, including the original historical T19 root, were verified file-by-file
against the SHA-256 inventories recorded at their original validation: **3,768 files, 0 changed,
0 missing, 0 extra.** The final return inventory re-hashed them and agrees with every recorded
inventory: **4,485 files, 0 mismatches**, T22 included.

---

## 2. Environment rebuild — what it touched

On **2026-09-16 ~14:04–14:06 local** an unknown actor (no trace in the interactive shell history)
deleted most of the pinned interpreter environment `/home/dwong/mlst-successor-py312-20260913`:
`bin/`, `pyvenv.cfg`, all package metadata and `pytest`. It also cleared the uv cache, archived
the 13 Sep audit directory to a tar.gz, and killed the T22 monitor. Full record:
`INCIDENT_2026-09-16_env_deletion.md`.

**Rebuilt at the user's direction** at the same path, from the intact base interpreter, with the
tested pins `numpy 2.4.6, scipy 1.17.1, pytest 9.1.0, h5py 3.16.0`. Swapped in at 22:24:37 after
confirming no reduction was running. The damaged venv is quarantined as
`...py312-20260913.damaged-20260916`, not deleted.

**Stages run under the rebuilt environment**
* T22 final reduction (production), 2026-09-16 23:43:39–23:44:45.
* Post-run validation re-reductions: T22 copy, and the T19 cross-environment check.

**Stages run under the original environment**
* All 727 T22 cells. Pilot and control ran in pid 2933410 (13 Sep); all 690 confirmation cells
  ran in pid 3050746, started 13 Sep 21:11. Attribution is from ledger pid records.
* T22 partial reduction and freeze (13 Sep 21:11).
* Every stage of T19, T20, T21 and both CRESST selections.

*Limitation:* the confirmation worker lived across the deletion (14:06) and the swap (22:24).
Its modules loaded from the original files at start, and 72 cells completed after the deletion
with no failures. Any module it imported lazily after 22:24:37 would have come from the rebuilt
tree at identical versions. That cannot be checked after the fact.

**Compatibility checks**
1. `environment()` equals T22's receipt exactly, executable path included.
2. Fresh full preflight on the rebuilt environment against T22's exact source: **136 tests, 0
   failures, 0 errors, 0 skips**. Its environment dict is identical to T22's original receipt,
   which was not replaced.
3. **Cross-environment reproduction:** T19's `REDUCED.json`, produced 13 Sep under the original
   environment, re-reduced under the rebuilt environment reproduces identically.
4. T22's re-reduction recomputes the pilot freeze deterministically and refuses on any
   difference. The 13 Sep original-environment freeze was reproduced.

Disclosure: the wheels were re-downloaded and could not be byte-verified against the originals,
whose metadata was deleted.

---

## 3. Scientific findings (separate from execution success)

### T22-S simulation — synthetic calibration of the archived T22 inference pipeline

The package's own scope statement, verbatim: *"It cannot establish the adequacy of that procedure
on unexamined instrument data, whose dependence structure is not known; the archived T22 verdicts
remain withheld and nothing here changes them."*

* **Freeze.** The candidate `reduced` replicate setting (file 1000, hier 4000, curve 200) was
  indistinguishable from the reference in **8/8** pilot scenarios, so it was frozen.
* **Controls.** 5/5 passed:
  * AR(1) lag-1 0.001 and 0.6007 against 0 and 0.6 (tolerance 0.03)
  * IID bootstrap SE ratio 1.004 (tolerance 0.25)
  * dependence gate withholds every claim, and its positive control is supported when stabilised
  * Holm arithmetic exact
  * simulator estimand within tolerance
* **Type-I (negative claims on nulls): within tolerance.** 0 of 30 null scenarios fail. Maximum
  any-null-claim rate 0.024 (Wilson 0.016–0.035) against the band [0, 0.07].
* **Coverage of the primary contrast `raw-bin|empca`: NOT within tolerance.** Band
  [0.925, 0.975]; **54 of 69 scenarios pass, 15 fail** (`all_within_tolerance: false`).
  * **12 under-cover** (0.823–0.902, Wilson interval entirely below the band). They concentrate in
    small-file scenarios (`n4`, `n8`). The worst are the drift scenarios:
    `null-partial-n4-phi0.6-tau0.01-drift0.15` at 0.823 and `alt-medium-n4-phi0.6-tau0.01-drift0.15`
    at 0.835. Heavy-tailed (`t3`) and `tau0.03` scenarios also fail.
  * **3 over-cover**, i.e. conservative (0.987–0.999): `null-partial-n16-phi0.6-tau0`,
    `null-partial-n16-phi0.6-tau0.03-dominant`, `null-partial-n4-phi0.6-tau0.03-dominant`.
* **Withholding.** Median 0.595, range 0.377–0.956. Highest in strongly dependent (`phi0.85`)
  scenarios.
* **Power proxy** (any claim supported, 39 alternative scenarios). Median 0.404, range 0.044–0.583.
* **Feasibility of a matched real-instrument comparison: NOT feasible under the frozen rules at
  4, 8 or 16 independent files.** At every size:
  * coverage is out of tolerance (type-I is within)
  * minimum medium-effect power is 0.019 / 0.039 / 0.035 against the required 0.5 (0.297 /
    0.736 / 0.761 given not withheld)
  * maximum withholding is 0.941 / 0.956 / 0.956 against the allowed 0.5

  The package rule states that the output is then *"a feasibility failure with its reasons, never
  an artificial four-file significance claim."*

**Interpretation boundary:** the declared procedure controls false positive claims in these
simulations. It does not achieve nominal coverage in small-file, drifting or heavy-tailed
scenarios, and it withholds most verdicts under strong dependence. The real-data comparison is
therefore infeasible at the file counts examined. This is a negative feasibility result about the
procedure on simulated data. It makes no statement about instrument data and does not change any
archived verdict.

### T19-S (from `T19_COMPLETION.md`, 13 Sep)

* **Theorem-domain bound satisfied** in all in-domain confirmation fits: pooled 1.0, n = 788,
  zero violations. **But `S_full` is undefined** ("no selected in-domain arm", 0 scored fits):
  inconclusive, not support.
* **First-order agreement** in band [0.5, 2.0] everywhere **except `S_full/beyond` at 0.9167**
  (negative finding).
* **Residual orders** consistent with the predeclared orders in both engineered families.
* **Training-sample arm:** bound not asserted; every realisation is beyond the domain.
* Package stress label: *"eta in [0.2, 1.3] is OUTSIDE the theorem domain; nothing in that range
  validates or extends the 44 eta² bound."*

### T20-S (13 Sep coordination report)

* **Supports** the frozen c/N + d/N² model for x_U at all four theta values on the tested analytic
  Gaussian grid.
* **Paired class contrast supports three of four.** theta = 0.12 fails its frozen band (5/6 grid
  points inside; maximum studentized deviation 3.370 against 2.984).
* Does not revise the historical pulse-family verdict or certify the reference optimum.

### T21-S (13 Sep coordination report)

* **H1 does not pass universally.** Primary agreement is 296/298, with both failures in
  orthogonal/high configurations; correlated agreement is 106/106.
* The orthogonal failures limit the isolated-spike predictor at these sizes. They do not validate
  the loadings explanation.
* Historical predictions remain unmet, and no detector benefit is established.

### CRESST archive and calibration — INTENTIONALLY PARTIAL (13 Sep coordination report)

* **Archive reconstruction reproduces archived intervals bit-identically**, with its five archive
  controls passing.
* **Calibration is intentionally partial and is not raw-pulse validation.**
* These are archive sensitivity and calibration results. They are not new-run population coverage
  or real-detector performance claims.

---

## 4. Excluded — not attempted, not pending

* **Full CRESST robustness:** needs raw pulse/noise/time arrays and a validated adapter.
* **T22 real-data matched comparison:** needs a frozen matched design and suitable data. T22's
  simulation additionally finds it infeasible under the frozen rules at 4–16 files.

## 5. Incidents during the round (all recorded, all closed)

1. **13 Sep — Ceph entry-quota overrun** caused by this agent's repository clone. Reversed within
   6 minutes, with no production impact; the spanning cell was reconciled.
   `INCIDENT_2026-09-13_quota_overrun.md` (in the 13 Sep archive).
2. **16 Sep — pinned environment deletion** by an unknown actor. Rebuilt with the checks above.
   `INCIDENT_2026-09-16_env_deletion.md`.
3. **16 Sep — replacement monitor false positive** (a substring match on shell text). Detection
   fixed; the bad record is annotated.
4. **17 Sep — validator run 1 defect.** Fixed, and run 2 passed; run 1 is retained.

## 6. Return artifacts

`RETURN_INVENTORY.sha256` (sha256 `0fa65ff92d9ae76350f9d87eb7d03f3ea716a768ca6ff16a2c718bfa67c95096`) and
`RETURN_INVENTORY.json`: **4,634 files, 0.84 GB, 16 groups**. The groups cover:
* all seven run roots, with both CRESST roots labelled intentionally partial
* launcher stdouts
* three preflights: T22 original, T19 amended, rebuilt environment
* supplied and accepted manifests
* the T19 amendment patch and transition inventories
* the 13 Sep audit archive
* this records directory

Verify with `grep -v '^#' RETURN_INVENTORY.sha256 | sha256sum -c --quiet`. Files written after the
inventory (this report and the final restart note) are covered by `RETURN_INVENTORY_ADDENDUM.sha256`.

**Keep on local disk until returned:** `/var/tmp/mlst-t19-amendment-20260913/` and
`/var/tmp/mlst-t22-envrebuild-preflight-20260916/` hold inventoried preflights, the accepted
manifest and the T19 amendment snapshot. `/var/tmp/mlst-t22-terminal-copy-20260916/` and
`/var/tmp/mlst-crossenv-t19-20260916/` are validation copies and may be removed after review.

## 7. Open items (none block completion)

* Scientific review of all six selections by a reviewer.
* Identify what ran the 16 Sep home-directory cleanup.
* About 190 unexplained extra Ceph entries appeared on 16 Sep from another writer. The quota is
  not at risk.
