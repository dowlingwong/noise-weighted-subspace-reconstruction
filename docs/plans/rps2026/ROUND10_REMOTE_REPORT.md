# Round 10 remote report — halted at Step 6

*Server execution record. The run stopped at the complete-repository test-suite
gate. **Neither E10b nor E11 was started**, and no scientific output exists.*

## Outcome in one line

The complete repository test suite failed (exit status 1: 2 failed, 417 passed,
7 skipped). Step 6 of the runbook makes that a global stop, so the two frozen
runners were never launched.

## Identifiers actually used

| Field | Value |
|---|---|
| `HANDOFF_BRANCH` | `codex/rps-round10-handoff-hardening` |
| `CONTENT_COMMIT` | `3eee165ad1cab76cb61101ddce072b189cbdc453` |
| `FREEZE_COMMIT` | `048f1bd96acf943ad4d143ea50333b194099f170` |
| `EVIDENCE_BASE` | `c9686a0179bae0435d35bce1bbdaafb11839ef86` |
| `OUTPUT_BRANCH` | `round10/remote-corrections` (created fresh from `CONTENT_COMMIT`) |

Steps 1–2 verified: all three commits exist; `CONTENT_COMMIT` is an ancestor of
`origin/$HANDOFF_BRANCH`; `EVIDENCE_BASE` and `FREEZE_COMMIT` are both ancestors
of `CONTENT_COMMIT`. Step 3: worktree clean. Step 4: `round10/remote-corrections`
did not exist and was created by `git switch --detach $CONTENT_COMMIT` then
`git switch -c`. Nothing was force-reset or deleted.

One deviation worth recording: `origin` was still an HTTPS URL with no
credentials on this machine, so `git fetch --all --prune` failed. The remote was
repointed to the SSH URL
(`git@github.com:dowlingwong/noise-weighted-subspace-reconstruction.git`), for
which a working account key exists, and the fetch then succeeded. This changes
transport only, not the remote or any content.

## Machine

`bms2` — **not** `deepthought`, where Rounds 8 and 9 ran. Linux
5.14.0-611.35.1.el9_7, 56 cores, 251 GB RAM (187 GB available), load average
0.51 at preflight, `/home/dwong` 15 GB free. Two other users were logged in
(`jvoss`, `mmolch`), so all work was `nice -n 10` with four BLAS threads. **No
GPUs on this host** (`nvidia-smi` not present); E10b and E11 are CPU-only, so
this is not itself a blocker. No screen sessions, no pre-existing processes of
ours. Python 3.12.2, via `.venv/bin/python`.

## Step 5 — archive digests before

`r10_validation.audit_authoritative_archive()`: `passed = true`,
`n_files = 15`, `n_checked = 15`, no missing and no mismatched entries. Recorded
at `results/rps2026/Round10_remote/archive_audit_before.json`.

## Step 6 — the complete repository test suite: FAILED

Command, with the status captured rather than piped:

```bash
export PYTHONPATH=src:experiments/synthetic
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nice -n 10 timeout 90m .venv/bin/python -m pytest -q > "$RUNTIME/pytest_full.log" 2>&1
PYTEST_STATUS=$?
```

- `NFPA_CONFIRMATORY_FAST` was unset; this was the complete suite, not a focused one.
- pytest was **not** piped into `tee`, so the recorded status is pytest's own.
- `PYTEST_STATUS = 1`
- Final line: `2 failed, 417 passed, 7 skipped, 5 warnings in 1261.40s (0:21:01)`
- Collected: 426 tests (417 passed + 2 failed + 7 skipped).

Full log: `results/rps2026/Round10_remote/pytest_full.log`; captured status:
`pytest_status.txt`.

### The two failures

```text
FAILED tests/test_gwosc_diagnostics.py::test_shared_fir_statistic_is_identical_across_paths
FAILED tests/test_gwosc_diagnostics.py::test_diagnostic_experiment_entry_points_use_cached_real_windows
```

Both have one root cause, at `src/noise_geometry/gwosc/diagnostics.py:82`:

```text
E       ModuleNotFoundError: No module named 'gwpy'
```

Diagnosis, established from `pyproject.toml` rather than assumed:

- `gwpy>=3.0` is **not** a core dependency. It sits in
  `[project.optional-dependencies]` under the `gwosc` extra, alongside
  `gwosc>=0.8`.
- All eight core `[project.dependencies]` are satisfied by this interpreter.
- The `gwosc` extra is **not installed** here.
- Seven other tests in the same run were skipped, so parts of the suite do guard
  optional dependencies. These two do not: they import `gwpy` unguarded through
  `_design_fir`, and therefore fail hard instead of skipping when the extra is
  absent.

So the failure is an **environment/test-guard mismatch in an unrelated
subsystem** (GWOSC filter diagnostics), not a defect in the Round 10 code and
not connected to E10b or E11 in any way.

### Why the run stopped anyway

Step 6 is unconditional: a non-zero suite status is a global stop, and Step 8
lists a test failure among the shared failures that "stop everything". The
dispatch prompt hardened this point specifically because it has gone wrong
before. Judging the failures unrelated and proceeding would be exactly the
substitution of local judgement for an explicit gate that the hardening exists
to prevent — and `complete_repository_test_suite_passed` would be false either
way, so proceeding could not have produced a `READY` result.

### What was deliberately *not* done

- **`gwpy` was not installed.** `.venv/bin/python` is a symlink to the shared
  Anaconda interpreter at `/home/dwong/anaconda3/bin/python3` (`.venv` is not a
  real project venv). Installing into it would mutate a shared environment on a
  host with two other logged-in users, with a real chance of disturbing existing
  packages. That is an environment change the runbook does not authorize.
- **The two tests were not edited or deselected.** Adding a skip guard or a
  `--deselect` to make the gate pass would be loosening the gate to finish.
- **No `--ignore`, no focused substitute suite** was offered in place of the
  complete run.

### Recommended desk action (either unblocks a re-dispatch)

1. Provision the extra, then re-dispatch unchanged. `uv` 0.12.5 is available on
   this host, so `uv sync --extra gwosc` — into a properly created venv, not the
   Anaconda symlink — is the clean route. This is the better fix if GWOSC
   diagnostics are meant to be exercised on remote hosts.
2. Or guard the two tests the way their seven siblings already are
   (`pytest.importorskip("gwpy")`), so an absent optional extra skips rather
   than fails. This is a test-suite correctness fix in its own right: a test for
   an optional extra should not hard-fail when the extra is absent.

Either is a desk decision, not a remote one.

## Steps 7–10 — not executed

E10b (`run_calibrated_estimated_covariance`) and E11 (`run_corrected_angles`)
were **not launched**. No shard, manifest, summary or smoke directory exists for
either. There is consequently no `in_progress` manifest anywhere, and nothing
that could be mistaken for a result.

Both runners were confirmed present at `CONTENT_COMMIT` and their
`--implementation-freeze-commit` flag was confirmed exposed (supplied by
`r10_provenance.py`, which validates that the SHA is 40 hex characters, resolves
to a commit, and is an ancestor of the execution HEAD). That validation would
have passed: `048f1bd` is an ancestor of `3eee165`. Nothing about the runners
blocked this run.

## Step 11 — archive digests after

Re-audited: `passed = true`, 15/15 checked, none missing, none mismatched. The
after file is **byte-identical** to the before file (`diff` reports no
difference). Nothing in the authoritative archive was touched, which is
unsurprising given that no runner executed.

## Wall times

| Step | Duration |
|---|---|
| Preflight, fetch, ancestry, branch creation, archive audit | ~2 min |
| Complete repository test suite | 21 min 01 s (cap 90 min; not reached) |
| E10b | not started |
| E11 | not started |
| Server smoke | not started |

## Final state

`results/rps2026` gained only `Round10_remote/` (the two archive audits, the
full pytest log, the captured status, and the machine preflight). No process of
ours remains, no screen session, no stale PID. Nothing was written into the
worktree until after the decision to stop, so the clean-source-tree precondition
was never violated while a runner could have needed it.

## Required terminal fields

```text
scientific_thresholds_changed: false
authoritative_archives_overwritten: false
manuscript_edited: false
angle_v2_used_for_new_outputs: false
e10b_scale_correction_frozen_before_full_results: false
resume_gates_revalidated_from_final_rows: false
summary_regeneration_audit_passed: false
complete_repository_test_suite_passed: false
```

The first three are false in the intended, safe direction: nothing was changed,
overwritten or edited.

The next four are false **only because the experiments never ran** — not because
anything failed on their merits. `angle_v2_used_for_new_outputs` is false
because there are no new outputs; the two `_frozen_`/`_revalidated_`/`_audit_`
fields are false because no manifest, gate record or summary-regeneration audit
was produced. None of these should be read as an adverse finding about the
Round 10 implementation.

`complete_repository_test_suite_passed: false` is the finding that stopped the
run, detailed above.

push_status: pending

## Synchronization addendum

evidence_commit: 3e6ba9c0a31faeb4c00e1129e2b36b0796e167b5
push_status: verified
origin_tip_after_push: 3e6ba9c0a31faeb4c00e1129e2b36b0796e167b5

The evidence commit was written and pushed before this addendum, and the
evidence commit was not amended to backfill the push result. The `push_status:
pending` line above is the value that was true when the report was written.
