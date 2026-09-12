# B0 — integrated execution accounting: status

Branch `codex/t19-t22-successor-20260912`, worktree `/ceph/dwong/nwsr-review-final`.
Written 2026-09-12, after the review that found B0 **partial, not complete**.

**Verdict: B0 is PARTIAL.** All six required items are implemented and each is
discharged by executed tests. B0 is not called complete for one named reason,
stated in full in §3 and §8: the concurrency guarantee rests on an advisory
`flock`, and the probe that measures whether `flock` actually excludes on the
execution filesystem failed once, unexplained, and did not reproduce in five
subsequent full-suite runs. Until that is explained, **the serial/manual
fallback stands: run one T22 stage at a time.**

The review's earlier finding was an overstatement. This document is written to
be falsifiable: every claim below names the test that establishes it, and every
limit is stated as a limit rather than as a caveat at the end.

---

## Requirement-by-requirement

| # | Requirement | Status |
|---|---|---|
| 1 | Reserve planned fits before data access or estimator calls | **Discharged** |
| 2 | State machine wired into the runner; uuid in provenance | **Discharged** |
| 3 | Serialised completion-ledger updates | **Discharged, conditional on `flock`** |
| 4 | Concurrent launches and crash/retry cannot bypass the cap | **Discharged, conditional on `flock`** |
| 5 | Distinct execution identity under byte-identical outputs | **Discharged** |
| 6 | Exposure records: enforce or retain as unimplemented | **Discharged — enforced** |

### 1. Reserve before the data is opened — discharged

`t22_tidmad_narrowband.main` mints the reservation after the freeze binding and
the budget pre-flight and before the first instrument byte. Nothing above that
line opens a data file: the science-file digests, `noise_bin_power`,
`load_stretch`, `stage_evaluate`, `score_cell` and every `h5py` call are below
it. `reserve_stage_fits` sweeps abandoned reservations, then re-tests the cap
**under the reservation store's exclusive lock** against the deduplicated union
of completed executions plus every open charge.

A stage that executes no fits also reserves now. `characterize` opens every
allocated stretch, so it takes a zero-fit reservation: it charges nothing, can
never be refused (a charge of nothing cannot cross a cap, even a spent one), and
its data access is bounded and recorded like any other. `freeze` and
`recombine` read archived JSON only and reserve nothing.

Established by `test_the_reservation_is_running_before_the_first_data_access`
(a spy inside `_science_paths` — the runner's first touch of the data root —
reads the store at that instant and finds one `running` reservation),
`test_the_estimator_runs_inside_a_running_reservation`, and
`test_characterisation_reads_under_a_zero_fit_reservation`.

### 2. The state machine, and the uuid as provenance — discharged

`pending` → `running` for the whole execution; `failed` on any exception, which
is charged and never refunded; `completed` once the output is on disk, carrying
the fits that actually ran and that output's digest; `interrupted` by the next
launch's sweep when the owning process is gone. The uuid is written into the
stage output's own `provenance["fit_reservation"]` and into the ledger entry.

Established by `test_a_completed_stage_carries_its_uuid_into_the_output_and_the_ledger`
(state history `pending → running → completed`, digest and output ref recorded,
uuid in the output and in the ledger entry, and the three records counting as
*one* execution), the failure-path tests in §4, and
`test_a_process_death_mid_run_stays_charged_and_a_retry_accumulates`.

### 3. The ledger append is serialised — discharged, conditional on `flock`

`record_in_ledger` was `load → append → os.replace` with nothing between the
read and the write: atomic per write, but two completions interleaving there
lose an entry and every fit it recorded. The whole read-modify-write now runs
inside `fit_accounting.locked`, the same lock the reservation store takes.

Established by `test_concurrent_ledger_appends_lose_nothing`: eight real
processes append twenty entries each to one ledger on CephFS, and all 160 must
survive. **Negative control: with the lock removed, this test fails, losing 140
of 160 entries.** It is a check of the lock, not of the weather.

Two further bugs the contended tests found and fixed: `write_json` now writes
through a temporary file and `os.replace` (the budget scan reads every stage
output under the run tree and could read a half-written one), and
`_write_ledger`'s temporary file carries the pid (two writers shared one
`FIT_LEDGER.json.tmp` name and raced to rename it; one died with
`FileNotFoundError`).

What this does **not** give: the lock binds only processes that take it. A
hand-edit of the ledger while a stage is completing is still a lost update, and
so is any other tool that writes it.

### 4. Concurrent launches and crash/retry — discharged, conditional on `flock`

* Concurrent launches: six launches into a run tree with room for two get
  exactly two, and the cap is not crossed
  (`test_concurrent_launches_cannot_oversubscribe_the_cap`, real forked
  processes, on CephFS). The unlocked pre-flight can pass for all six at once;
  the reservation, taken under the store's lock, is what decides.
* Concurrent completions: eight stages completing at once land eight
  reservations and eight ledger entries, none lost
  (`test_concurrent_completions_lose_no_ledger_entry`).
* A second ledger does not buy a second budget. The store a run mints into
  belongs to the **run tree** (`--out`'s parent), not to wherever `--ledger`
  points, and every store the scan directories expose is charged and swept. Two
  launches into one tree therefore contend even before either has written an
  output for the expenditure scan to find
  (`test_a_second_ledger_does_not_open_a_second_budget`).
* Crash and retry accumulate, never reset. `SIGKILL` mid-stage leaves the
  reservation `running` — which is charged — and the next launch sweeps it to
  `interrupted`, keeping the charge, while its retry adds its own
  (`test_a_process_death_mid_run_stays_charged_and_a_retry_accumulates`). Three
  failed attempts charge three times (`test_a_retry_never_resets_the_expenditure`).

What this does **not** give:

* A reservation owned by **another host** cannot be probed from here. It is left
  `running` — still charged — and reported as unverifiable. Conservative, but it
  will not be reclassified.
* Two runs in genuinely separate trees (different `--out` parent *and* different
  `--ledger`) share no store and no ledger, and account separately.
* A process that does not use this runner is not bound by any of it.
* Everything above assumes `flock` excludes on the filesystem in use. See §8.

### 5. Distinct execution identity under byte-identical outputs — discharged

`union_executed` merged any two records sharing a digest, so two distinct
reservations whose outputs happened to be byte-identical collapsed into one
execution and half their compute vanished from the cap. A digest claimed by more
than one reservation uuid is now *contested*: records carrying a uuid merge on
the uuid alone, and a record of those bytes naming no reservation is counted on
its own rather than attributed to whichever execution came first — over-counting,
the safe direction for a cap.

Established by
`test_byte_identical_outputs_from_distinct_reservations_stay_distinct` (unit),
`test_an_unclaimed_record_of_contested_bytes_is_counted_on_its_own` (unit) and
`test_byte_identical_outputs_from_distinct_reservations_stay_two_executions`
(integration: the twins count 48, and a third unattributable record of those
bytes takes it to 72 — over-charged, never hidden).

Residual limit, unchanged and documented in `fit_accounting`'s module docstring:
two executions that carry **no reservation at all** and produced identical bytes
still merge into one. Reserving before executing is what removes that case, so
only an execution that ran without a reservation keeps it.

### 6. Exposure records — discharged, as enforcement

Taken as enforcement, not as a documented gap. Every instrument open in this
runner now goes through `open_instrument_file` — `load_stretch`,
`noise_bin_power` and `stage_characterize` each called `h5py.File` directly, and
there were exactly those three — and `main` makes the reservation's declared
exposure binding for the duration of the stage via `exposure_allowance`. A file
the allocation does not name, or a sample region outside the declared stretch, is
refused at the open.

Established by `test_an_undeclared_file_is_refused_at_the_open`,
`test_a_region_outside_the_allocation_is_refused_at_the_open`,
`test_without_an_allowance_nothing_is_restricted`,
`test_a_validation_stage_cannot_open_a_final_test_file` (the property the
allocation exists for: a validation stage is refused a final-test stretch and
refused its own file past its own window) and
`test_characterisation_reads_under_a_zero_fit_reservation`.

What this is **not**, stated in `exposure_allowance`'s own docstring: a guard
inside this process, not a sandbox. It cannot constrain another program, a
hand-written script, or a future code path in this file that opens `h5py`
directly instead of going through the opener, and it checks the region once, at
the open, not on each read. **It has been exercised on synthetic fixtures only.**

---

## 7. What has *not* been demonstrated

Stated plainly, because the previous review caught an overstatement:

1. **No part of this has run a real stage.** The mission compute budget is spent.
   Every test replaces the estimator and the data layer; `--data-root` points at
   a directory that does not exist and every route to instrument data raises. The
   accounting integration is demonstrated; a real 12-hour fit stage surviving
   under it is not.
2. **The exposure guard has never refused, or admitted, a real instrument file.**
3. **Crash-safety is not "no work is lost".** A stage that dies mid-fit loses its
   fits. What is guaranteed is that the compute it spent stays charged, so a
   retry adds rather than resets.
4. The one unexplained `flock` probe failure in §8 is not explained.

---

## 8. `flock` on CephFS — the measurement, including the failure

`/ceph` is a kernel CephFS mount
(`[2a00:1398:5:fe01:...],[129.13.101.127]:/ceph_data`, fstype `ceph`). `flock` is
advisory and is a silent no-op on some network filesystems, so it is measured
rather than assumed: `fit_accounting.lock_is_effective` forks contending
processes that each do locked read-modify-write increments of a counter and
looks for a lost update. The concurrency tests build their fixture tree under
`/ceph/dwong/.b0_lock_fixture_<pid>/` and remove it afterwards.

**Result: `flock` held in every reproducible measurement.** 150 of 150 contended
increments survived on CephFS (and on local `/tmp`), repeatedly; the eight-process
ledger-append test loses nothing with the lock and loses 140 of 160 entries
without it.

**The failure: one full-suite run failed
`test_flock_actually_excludes_on_the_execution_filesystem`.** It was not
reproduced by five consecutive full-suite re-runs afterwards, nor by repeated
runs of that test alone, and the diagnostics at the time did not record whether
it was a lost update or a worker that failed for an unrelated reason. The probe
now reports per-worker exit status, the child's own exception text, and the lost
update count separately, so a recurrence will say which.

This matters because the failure direction is *unsafe*: if `flock` silently does
not exclude, two runners can both reserve the same room and the cap is
oversubscribed — a bypass, not a conservative over-count. That single
unexplained observation is why B0 is called partial and why the fallback stands.

**Fallback, in force until that is explained: run T22 stages serially, one at a
time, on one host.** The accounting is conservative under serial execution
without relying on the lock at all.

---

## 9. Regression bar

All run at `80fab79`, with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`, from
`/ceph/dwong/nwsr-review-final`. `$PROTOCOL` is
`/ceph/dwong/arXiv-v1-polished-extended-preprint/plan/mlst_protocol_2026-09-11-t22-stage0.json`.

| Command | Result |
|---|---|
| `python3 -m pytest tests/test_fit_accounting.py -q` | 28 passed (was 25) |
| `T22_PROTOCOL=$PROTOCOL python3 -m pytest tests/test_mlst_t22_narrowband.py -q` | 51 passed (unchanged) |
| `T22_PROTOCOL=$PROTOCOL python3 -m pytest tests/test_accounting_integration.py -q` | 17 passed (new) |
| `T22_PROTOCOL=$PROTOCOL python3 -m pytest tests/ -q --ignore=tests/test_public_data_smoke.py` | 369 passed, 11 skipped |
| `python3 scripts/verify_t22_enforcement.py --protocol $PROTOCOL` | `ALL PROBES PASSED`, 21 `PASS` lines |

The union against the real archive is unchanged: **416 executed / 240 unique
scientific, cap 250**, 4 duplicate records merged.

`results/` was not modified: `git status --porcelain results/` is empty and
`results/t22_2026-09-11/FIT_LEDGER.json` is still
`847048b4aecbc835059fc5b7f3b8f84c2f3fd6645a40c8042635161038a64c45`.

---

## 10. Commits

| Commit | |
|---|---|
| `596ab2c` | accounting: a reservation uuid is an identity, not a hint |
| `b4c01e9` | t22: reserve before the data is opened, and serialise the ledger append |
| `f5f0324` | tests: the integrated execution path, contended on the real filesystem |
| `890ce2e` | t22: make the declared exposure binding, at the one place files are opened |
| `cc67e2a` | t22: the reservations belong to the run tree, not to `--ledger` |
| `80fab79` | t22: characterisation reads under a zero-fit reservation too |

## 11. What would make B0 complete

1. Explain or reproduce the single `flock` probe failure of §8. If it was a lost
   update, `flock` is not trustworthy on this mount and the lock must be replaced
   (an `O_EXCL` lock file with owner liveness, or a single-writer daemon); if it
   was a worker failing for another reason, say so and the fallback can be
   dropped.
2. Exercise the integrated path once against real data — the first stage of any
   future campaign, under a budget that permits it.
