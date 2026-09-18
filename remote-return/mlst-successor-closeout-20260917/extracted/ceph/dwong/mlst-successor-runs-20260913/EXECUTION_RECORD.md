# MLST successor round — remote execution record

Execution host: **deepthought** (40 cores, 503 GB RAM, Linux 4.18.0-553, no batch scheduler).
Run base: `/ceph/dwong/mlst-successor-runs-20260913` (outside both repositories).
Date: 2026-09-13. Operator: automated run under the `remote/README.md` protocol.

## 1. Repositories and manifest

| | path | HEAD at execution | worktree clean |
|---|---|---|---|
| experiment | `/ceph/dwong/noise-weighted-subspace-reconstruction` | `078b9ee5c8b5af8ec53a994bcc918e8d19c2d863` | yes |
| manuscript | `/ceph/dwong/arXiv-v1-polished-extended-preprint` | `99040f8b2ca1d9d24a4cae8466fdafae231a2d85` | yes |

Manifest `remote/MANIFEST.json`, sha256 `3c22548d06192585ad419829cb006eaaef84df8ee318c2b2ebb03998b445fd9c`
— identical to the digest recorded in `LOCAL_ACCEPTANCE.json`.

Freeze-time commits recorded in the manifest (historical, per its own note):
experiment `4d475694…`, manuscript `fba4375d…`. These differ from the HEADs above; the manifest
states content hashes are authoritative, and **all 357 frozen files verified by hash**
(344 experiment + 13 manuscript). Both ignored historical trees were present
(`results/mlst_2026-09-10`, 33 files; `results/t22_2026-09-11`, 24 files) as was the manuscript
`evidence/` tree (272 files).

**The manifest was not regenerated at any point.** `prepare.py verify` was run before and after
the environment change and passed both times, including the five production matrix identities:

| package | cells | fits |
|---|---:|---:|
| t19s | 136 | 3,840 |
| t20s | 964 | 4,800 |
| t21s | 2,499 | 2,496 |
| t22s | 727 | 0 |
| cresst_s | 395 | 1,056 |

These match the README table exactly.

## 2. Environment and preflight

Python 3.12.13 virtualenv with the exact tested pins (`numpy==2.4.6`, `scipy==1.17.1`,
`pytest==9.1.0`, `h5py==3.16.0`). BLAS threading forced to 1 by the launcher.

Preflight was run **twice on this host**, both green:

| run | venv | tests | failures | errors | skipped | dev-test wall |
|---|---|---:|---:|---:|---:|---:|
| initial | `/ceph/dwong/mlst-successor-py312-20260913` | 136 | 0 | 0 | 0 | 93.92 s |
| after env change | `/home/dwong/mlst-successor-py312-20260913` | 136 | 0 | 0 | 0 | 93.13 s |

136 tests with zero skips, matching local acceptance. All five matrices enumerated without
execution. The **second** receipt (`preflight/PREFLIGHT.json`) is the one that authorised every
execution reported below; the first was superseded by the environment change in §4 and re-run as
the README requires. No execution used a receipt from another environment.

Diagnostic only: dev-test wall time is 93.9 s here against 43.9 s in local acceptance (~2.1×).
Per the README this is **not** a production scaling guarantee, but it is the only cross-host
timing evidence available and it bears on the T22 projection (§6).

## 3. Results by package

| selection | matrix cells | distinct completed | cell files | fits charged / cap | REDUCED | status |
|---|---:|---:|---:|---:|---|---|
| t19s | 136 | 20 | 20 | 672 / 4,800 | no | **STOPPED — launcher defect (§5)** |
| t20s | 964 | **964 (all)** | 964 | 4,805 / 5,400 | yes, final | **complete** |
| t21s | 2,499 | **2,499 (all)** | 2,499 | 2,497 / 2,750 | yes, final | **complete** |
| t22s | 727 | 15 and rising | = completed | 0 / 64 | no | running (see §6) |
| cresst-archive | 395 | 21 | 21 | 6 / 1,400 | yes | **complete for its selection (partial by design)** |
| cresst-calibration | 395 | 25 | 25 | 0 / 1,400 | yes | **complete for its selection (partial by design)** |

Cell-file counts equal distinct completed cells in every root, with no duplicates, no extra files
and no misnamed files reported by any reducer, and no `.tmp` residue anywhere under the run base.
No package exceeded its frozen fit cap and no cap was raised. No `cells_with_repeat_completion`
were recorded.

### CRESST-S archive (`cresst-archive`)

`partial: true`, `complete: false`, `development_only: false`. 21 cells of the 395-cell whole
design; the 374 un-run cells are the robustness families that require raw pulse data. This is the
intended partial result, not a shortfall.

- All six archived input files hash-match their protocol declarations (`matches_protocol: true`).
- Archive interval reconstruction is **bit-identical** to the archived values at 10,000
  replicates (`max_abs_deviation: 0.0`, `bit_identical_to_archive: true`) across the
  block16/block256 channel-fixed and hierarchical constructions.
- Leave-one-channel-out deviations are 0.0 and within tolerance. The package carries its own
  scope label: *"descriptive sensitivity of the archive — not population inference; seven
  run/channel clusters from two runs cannot establish coverage for a new run or a new detector."*
  That label is retained here unchanged.
- All five scientific controls pass: `block1_equals_iid`, `empca_uniform_weight_limit`
  (uniform weights reduce EmPCA to PCA, relative difference 4.33e-05 against 1e-03 tolerance),
  `event_weight_constant`, `null_dependence`, `zero_effect_identity`
  (`max_abs_fraction: 0.0`).

### CRESST-S calibration (`cresst-calibration`)

`partial: true`, `development_only: false`. 25 cells (`calib_pilot` 12, `calib_confirm` 12,
`calib_control` 1). Coverage rises with cluster count — 0.8525 (n=2 runs, 4 clusters),
0.915 (n=4, 8 clusters) — against a declared band of [0.85, 0.99]. The `null_dependence` control
passes: the null arm covers near nominal (0.955, 95% interval [0.930, 0.971], n=400) and
between-run variance degrades coverage as predicted (drop 0.155).

These are calibration diagnostics on the archived/simulated design. They are **not** validated on
raw pulses and establish nothing about real-detector performance.

### T20-S — complete

`partial: false`, `development_only: false`. All five launcher phases ran (algebraic, pilot,
partial reduce, confirmation, final reduce), with both the pilot report
(`logs/20260913T133412-2933398/02-REDUCED.json`) and the final report (`04-REDUCED.json`)
preserved. `reconciliation.complete: true`, expected 964, 0 missing, 0 duplicates, 0 extra,
0 misnamed, `cells_with_repeat_completion: []`.

Fit accounting: frozen matrix 4,800 · distinct completed cells 964 · **executed/charged 4,805** ·
cap 5,400 (595 remaining). The 5-fit excess is the quota-failed attempt on `3b38a988d77676c0`
(§4.1), charged and not refunded. No cap was raised.

All seven controls carry explicit `true` flags — `bound_order`, `identity`,
`reference_consistency`, `spectrum_fixed`, `theta0_a_zero`, `training_loss_selection`,
`x_U_nonnegative`. The reduction notes that a control failure raises `ScienceError` (exit 4)
instead of reducing, so these are genuine passes rather than unevaluated fields. Invariants report
no `invariant_problems`; numerical rank 256 throughout and eigenvalue deviations at 1e−14/1e−15.

**Outcome.** Both estimands froze to the `c_over_N_d_over_N2` model; no thetas were excluded.

| estimand | supported at grid | detail |
|---|---|---|
| `x_U` (excess risk of PCA; exact, ≥ 0) | **4/4 thetas** | 6/6 grid points inside band at every theta |
| `d_minus_a_ref` (paired class contrast) | **3/4 thetas** | fails at θ=0.12: 5/6 inside, max studentized deviation 3.370 against critical value 2.984 |

The package's own vocabulary records that the `a` interval is *"[partial-trace lower bound,
best-found upper bound]; **not certified**"*, and its `historical_note` scopes the result: *"new
analytic Gaussian family; says nothing about the archived pulse-family proxy (c/N refuted; N* ~ 960
interpolation; 800/1600 reversal stand)."* That scoping is retained unchanged — this run neither
revisits nor repairs the archived pulse-family verdicts.

### T21-S — complete

`partial: false`, `development_only: false`. Every phase finished and reconciled exactly:
algebraic 3/3, pilot 416/416, confirmation 2080/2080; `reconciliation.complete: true`, 0 missing,
0 duplicates, 0 extra, 0 misnamed, and `cells_with_repeat_completion: []`.

Fit accounting, distinct against executed:

| | fits |
|---|---:|
| frozen matrix (manifest) | 2,496 |
| distinct completed cells | 2,496 |
| **executed / charged** (incl. the one quota-failed attempt of §4.1, later retried) | **2,497** |
| cap | 2,750 |

The one-fit excess over the matrix is the failed attempt on cell `d545591be9b6945c`; it is charged,
not refunded, and the package still finished 253 fits under cap. No cap was raised.

The pilot froze before confirmation, as required: `FROZEN_PREDICTOR.json` sha256
`50ff515c3a1c363d9faa1597ed3a7b83c69dc4ea27ec3451eda5a11af567ff3d`, with the pilot report preserved
separately at `logs/20260913T133412-2933399/01-REDUCED.json`.

**Outcome — H1 did not pass everywhere.** Per-arm, primary/near-edge/unique agreement:

| arm | primary | primary all pass | near-edge | unique | ci95 coverage |
|---|---:|---|---:|---:|---:|
| correlated \| hi | 53/53 | true | 51/51 | 96/104 | 0.8173 |
| correlated \| lo | 53/53 | true | 51/51 | 104/104 | 0.7019 |
| orthogonal \| hi | 94/96 | **false** | 4/8 | 98/104 | 0.7692 |
| orthogonal \| lo | 96/96 | true | 5/8 | 101/104 | 0.8462 |
| **overall** | 296/298 | **false** | 111/118 | 399/416 | 0.7837 |

Both primary disagreements sit in the orthogonal arm; the correlated arm passes primary throughout.
The package ships its own `interpretation_rule` block, which distinguishes
`predictor_fails_on_orthogonal_only` ("the isolated-spike overlap itself does not describe this
learner at these sizes; the loadings step is then untested") from `near_edge_disagreement`
("reported as the expected finite-d limitation of an asymptotic formula, **not as support and not
as refutation**"). Selecting the applicable branch is a scientific-review judgement and is **not**
made here.

Controls: `all_zero` carries explicit flags and passes 4/4 configurations. `all_strong`,
`correlated_unique_differs` and `orthogonal_unique_equals_ordinary` report per-configuration
numeric comparisons with **no** pass/fail flag; they are diagnostics for review, and are not
reported here as passing.

The reduction labels its own output `"executed (not yet validated)"` and records that **all five
registered T21 predictions (archive, 11 Sep 2026) remain NOT met**, and that this package cannot
repair those verdicts. Nothing in this run changes that, and no benefit has been established.

## 4. Incidents

Three incidents occurred. All are recorded in full; nothing was refunded or rewritten.

### 4.1 `/ceph/dwong` entry quota exhausted (infrastructure)

Mid-round, writes began failing with `OSError: [Errno 122] Disk quota exceeded`. Cause: CephFS
`ceph.quota.max_files` counts **entries**, not files. `/ceph/dwong` stood at
`rfiles 452,688 + rsubdirs 47,313 = rentries 500,001` against `max_files = 500,000` — one entry
over, and all writes were hard-blocked (`max_bytes` is 0, i.e. unlimited; 1.47 TB used).

Charged consequences, all preserved:
- `t20s` cell `3b38a988d77676c0` — 1 failed attempt (5 fits planned, 5 observed).
- `t21s` cell `d545591be9b6945c` — 1 failed attempt (1 fit planned, 1 observed).
- `cresst-archive` — reduce succeeded and `REDUCED.json` was written; only the launcher's copy of
  it into the invocation log failed. 0 failed attempts.

Resolution taken: the Python virtualenv **created by this run** on `/ceph`
(6,259 entries) was removed and rebuilt at `/home/dwong/mlst-successor-py312-20260913`. That
returned the filesystem to 493,742 / 500,000 and restored writes. **No user data was deleted and
the quota was not altered.** Because `PREFLIGHT.json` records the interpreter path, this counted
as an environment change, so preflight was re-run (§2) before any execution resumed — as the
README requires. All affected selections were then resumed with the identical command and root.

Before this round the tree already stood at ~493,273 / 500,000 (98.7%); this run's artefacts
pushed it over.

### 4.2 Orphaned worker outside the launcher lock (operator error)

While stopping the T22-S launcher, a `pkill` pattern matched its own shell and killed it before
the signal reached the intended target. The launcher process died, **releasing its `flock` on the
run root, while its compute child survived, reparented to init**, and kept running against
`t22s`. When T22-S was later resumed, a second, correctly-locked worker started against the same
root — briefly violating the "one launcher per root" rule the README sets.

Detected by process audit, not by the launcher: the per-root lock is held by the launcher, not by
its worker, so an orphaned worker escapes it. The orphan (PID 2928670) was terminated by explicit
PID.

Damage assessment: **none to the data.** The orphan wrote 0 cell files and consumed 0 fits
(`fits_planned: 0` for the T22-S cell it held); there were no `.tmp` leftovers and no duplicate
completions. The audit trail is preserved: `t22s/EXECUTION_LEDGER.jsonl` retains two `running`
records for cell `1e985e44ee28b665`, from PIDs 2928670 (orphan) and 2933410 (live worker).

**Accounting consequence that must not be lost:** that cell now carries 2 charged attempts of the
3 the launcher allows. One further interruption of that cell will trip `check_retry_limit` and
refuse the next T22-S `run` phase. That is a real narrowing of the retry budget caused by this
incident, not by the science.

### 4.3 T19-S — see §5.

## 5. STOP: T19-S cannot complete under the documented launcher

T19-S failed on its first confirmation cell:

```
RuntimeError: confirmation cell refused: .../t19s/PILOT_SELECTION.json is absent;
reduce the pilot cells first (the frozen selection must exist before any confirmation
cell is drawn)
```

This is a scientific guard working correctly (`t19s/runner.py:230`). The defect is in the
launcher's phase plan. `prepare.py::phase_plan` gives T19-S:

```python
if selection == 't19s':
    return 't19s', [('run',[]),('reduce',[])]
```

a single **unfiltered** `run` followed by one `reduce`. T19-S's matrix interleaves `pilot` and
`confirmation` cells (`t19s/runner.py:106,109`), so the unfiltered run reaches a confirmation cell
before any `reduce` has written `PILOT_SELECTION.json`. Every other package is staged correctly —
t20s, t21s and t22s each interpose `('reduce',['--allow-partial'])` before their confirmation run.
**T19-S under this launcher is structurally unable to finish, deterministically, on any host.**

Why local acceptance did not catch it — the guard test excludes T19-S:

```python
# tests/test_successor_remote.py:8
def test_pilot_reduce_precedes_confirmation():
    for name in ('t20s','t21s','t22s'):
```

`tests/test_successor_20260913_t19s.py:359` (`test_pilot_freeze_confirmation_flow`) proves T19-S
*does* require the staged sequence — run pilot, reduce, run, reduce — but it drives the package
CLI directly and never the launcher. So 136/136 passes while production T19-S cannot complete.

**No remedy was applied here, by design.** `prepare.py` is inside the manifest
(`experiments/**/*.py`), so correcting `phase_plan` is a source amendment, which the README says
"require[s] a reviewed new manifest, not a silent remote edit". Editing it would also invalidate
manifest verification. Driving the T19-S module directly with hand-built phases would bypass the
documented launcher and its receipt checks. Both were rejected.

**T19-S was not retried.** The failure is deterministic, so the two permitted retries would have
been spent for nothing and each identical resume charges another 32 planned fits. The run root is
intact and resumable: 20 completed pilot cells are preserved and will be skipped, and the failed
confirmation cell `5493e66358a9c0cf` has used 1 of 3 attempts.

Suggested amendment for review (not applied):

```python
if selection == 't19s':
    return 't19s', [('run',['--roles','algebraic','control','pilot']),
                    ('reduce',['--allow-partial']),
                    ('run',[]), ('reduce',[])]
```

and extending `test_pilot_reduce_precedes_confirmation` to cover `t19s`.

## 6. T22-S resource position

T22-S has completed **15 of 727** cells, charging 0 fits throughout, exactly as the frozen matrix
predicts for a package with no estimator fits. Measured per-cell wall time over those 15 cells:

| min | median | mean | max |
|---:|---:|---:|---:|
| 259.2 s | 555.5 s | 671.3 s (11.19 min) | 1,467.1 s |

At the observed mean, 727 cells is **~135.6 h ≈ 5.6 days** single-worker, which the elapsed clock
corroborates (15 cells in 2 h 50 m). That supersedes the estimate taken from the first cell alone:
that cell ran 1,402.7 s and was near the top of the observed spread, and extrapolating it gave a
figure roughly twice too large. The revised number sits close to the README's ~105.23 h projection
rather than far above it, so the ~2.1× dev-test slowdown of §2 did **not** carry over to this
workload.

Caveats: 15 cells is still a small sample, all from the `sim_pilot`/`control` phase; the
`sim_confirm` family has not been timed at all and need not cost the same. The README's warning
that zero fits does not mean zero simulation cost is borne out either way.

This needs an owner's decision, not an operator's, for three reasons:
1. **Duration.** ~5.6 days of wall time on a shared node with no scheduler holding the
   allocation — better than first estimated, but still days.
2. **Headroom.** `/ceph/dwong` stood at ~496,200 / 500,000 entries when this record was last
   updated, i.e. ~3,800 spare, and it falls as every cell file lands. With t20s and t21s now finished, only t22s
   remains, needing ~712 more cell files plus logs, and T19-S another 116 if §5 is
   resolved. It fits with roughly 2,000 entries of margin — but that margin is shared with all
   other activity on this tree, and exhausting it again reproduces exactly the charged failures
   of §4.1. A quota increase would remove the risk outright.
3. **Retry budget.** Per §4.2, T22-S cell `1e985e44ee28b665` is one interruption away from
   exhausting its retries.

T22-S was left **running** rather than cancelled, since execution was explicitly requested and the
projection was stated up front. It is detached (`setsid`) and survives the controlling session.

## 7. Not executed, by instruction

- **CRESST robustness** — blocked. Requires raw pulse/noise/time arrays in `cresst-s-pulse-root/1`
  format; the repositories carry archived residuals only, and the release-to-schema adapter is
  unvalidated on raw data. The launcher offers no full-robustness mode. Not attempted; no
  development fixture was substituted for real pulses.
- **T22 real instrument comparison** — blocked. Needs a frozen matched design, feasible
  independent unexposed units and raw data. The simulation run in §3 is **not** an instrument
  verdict and must not be read as one.

## 8. Standing caveats

`REDUCED.json` and a zero exit do not establish any claim. Nothing here has been reconciled into
manuscript tables, and the reducer should be re-run on a copy before any table or claim is
updated. Negative and inconclusive results above are retained verbatim. The accounting limitation
in the README stands: directly instrumented estimators are guarded before invocation, but audited
legacy ALS and T22 estimator batches report counts only after returning, so this is not an
absolute CPU or internal-batch fit cap. No overrun was observed, and none would have been
refunded.
