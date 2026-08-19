# Round 10 remote report — E10b and E11 complete

*Server execution record. Both frozen runners completed; every gate passed.
Scientific interpretation and all manuscript edits remain desk work.*

## Identifiers actually used

| Field | Value |
|---|---|
| `HANDOFF_BRANCH` | `codex/rps-round10-handoff-hardening` |
| `CONTENT_COMMIT` | `76aaff601df5f974395f85c2d5dc7a95bcab6763` |
| `FREEZE_COMMIT` | `048f1bd96acf943ad4d143ea50333b194099f170` |
| `EVIDENCE_BASE` | `c9686a0179bae0435d35bce1bbdaafb11839ef86` |
| `OUTPUT_BRANCH` | `round10/remote-corrections` |

Ancestry verified: `CONTENT_COMMIT` is an ancestor of `origin/$HANDOFF_BRANCH`;
`EVIDENCE_BASE` and `FREEZE_COMMIT` are both ancestors of `CONTENT_COMMIT`.
`FREEZE_COMMIT` is unchanged from the halted 2026-08-19 attempt, as the dispatch
stated: the fix touched one test file only.

The runbook (`ROUND10_REMOTE_RERUN_RUNBOOK.md`) is byte-identical to the version
executed in the halted attempt — verified by `git diff` between the two content
commits — so the same seventeen steps applied.

## Step 0 — proof the fix is present

A SHA match alone is not proof, so the working tree was checked directly:

```text
HEAD_PINNED_OK
grep -c 'importorskip("gwpy")' tests/test_gwosc_diagnostics.py   ->  2
PYTHONPATH=. pytest tests/test_gwosc_diagnostics.py -q           ->  2 passed, 2 skipped
```

The two guards sit at lines 91 and 210. This host does not have the optional
`gwosc` extra, so the two `gwpy`-dependent tests skip rather than fail — the
intended behaviour.

## Step 0b — the stale output branch

`round10/remote-corrections` existed at `41cb5fb` from the halted run and
descends from the old content commit, so Step 4's ancestry check would correctly
have refused it. It was **renamed**, never force-reset or deleted:

```text
round10/remote-corrections  ->  round10/remote-corrections-halted-20260819
```

The published remote branch `origin/round10/remote-corrections` still carries
`41cb5fb`, the honest record of the halted attempt. The new local branch was
then created fresh from `CONTENT_COMMIT`. **This push therefore advances that
remote branch from the halted record to the completed run**; the halted commits
remain in its history and are also preserved locally under the renamed branch.

## Machine

`bms2` — again not `deepthought`. Linux 5.14.0-611.35.1.el9_7, 56 cores, 251 GB
RAM (188 GB available), load average 0.50 at preflight, `/home/dwong` with
15 GB free. Two other users logged in (`jvoss`, `mmolch`), so every job ran
`nice -n 10` with four BLAS threads. No GPUs on this host; both experiments are
CPU-only. No screen sessions and no pre-existing processes of ours. Python
3.12.2, NumPy/SciPy via `.venv/bin/python`.

## Step 5 — archive digests before

`r10_validation.audit_authoritative_archive()`: `passed = true`, 15/15 checked,
nothing missing, nothing mismatched.
(`results/rps2026/Round10_remote/archive_audit_before.json`)

## Step 6 — complete repository test suite: PASSED

```bash
export PYTHONPATH=src:experiments/synthetic
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nice -n 10 timeout 90m .venv/bin/python -m pytest -q > "$RUNTIME/pytest_full.log" 2>&1
PYTEST_STATUS=$?
```

- `NFPA_CONFIRMATORY_FAST` unset — the complete suite, not a focused one.
- Not piped into `tee`; the captured status is pytest's own.
- **`PYTEST_STATUS = 0`**
- Final line: `417 passed, 9 skipped, 5 warnings in 1155.04s (0:19:15)`
- **Collected: 426** (417 passed + 9 skipped).

The halted run recorded `2 failed, 417 passed, 7 skipped`. The two failures are
now skips — 7 → 9 — with the passing count unchanged at 417, which is exactly
the signature of the guard fix and of nothing else changing.

## Step 7 — the two scientific runs

Sequential, out of tree, four BLAS threads, `nice -n 10`, each with the freeze
SHA passed explicitly. No seed was parallelized and no cap was raised.

| Experiment | Seeds | Wall | Cap | Status |
|---|---:|---:|---:|:--:|
| E10b `run_calibrated_estimated_covariance` | 10 (0–9) | 3584.3 s = 1.00 h | 3 h | rc 0 |
| E11 `run_corrected_angles` | 30 (0–29) | 11944.3 s = 3.32 h | 4 h | rc 0 |

E11 covers all thirty seeds — the frozen cohort 0–9 and the replication cohort
10–29 — not ten; the 4 h cap held with ~40 min to spare.

## Step 8 — resume

Not needed. Neither run was interrupted or hit a cap, so no `--resume` was used
and no `in_progress` manifest was ever created. Nothing in this report is drawn
from an incomplete directory.

## Step 9 — server smoke (after the science)

Both smoke runs returned 0. Their manifests carry
`provenance.evidence_class = implementation_check_not_scientific_evidence` and
`implementation_freeze_commit = null`, correctly distinguishing them from the
two scientific runs, whose manifests carry
`evidence_class = scientific_evidence` and the real freeze SHA. Smoke output is
an implementation check and is not cited as evidence anywhere below.

## Step 10 — completed manifests verified

For both experiments: `status = complete`,
`summary_regeneration_audit.passed = true`,
`provenance.implementation_freeze_commit = 048f1bd96acf943ad4d143ea50333b194099f170`,
`provenance.source_tree_dirty_at_start = false`.

```text
E10b summary_digest  1846242438133ce52c1dd4adf2d78f34cfc34570c4748ac3e71f50f26b0d496b
E11  summary_digest  d4f9d004430f5bc9cf55a347c33bced1448190d1e1c9cc8678b1841c22bd6170
```

### Gate records — recomputed, not read back

E10b `gate_record`: `control_gate_passed = true`, 20 rows validated, tolerance
1e-9, worst control reproduction relative error **6.02e-15**,
`worst_stored_vs_recomputed_disagreement = 0.0`, recomputed from *"final
concatenated rows … with every archived value, reproduction error and delta
recomputed from the authoritative E1 table rather than trusted from the stored
columns"*.

E11 `gate_record`: `reproduction_gate_passed = true`, 90 rows validated,
tolerance 1e-9, worst reproduction relative error **2.80e-14**,
`worst_stored_vs_recomputed_disagreement = 0.0`, recomputed from the
authoritative E1 (seeds 0–9) and E9 (seeds 10–29) tables. E11's
`reproduction_gate` block names those two sources explicitly.

A `worst_stored_vs_recomputed_disagreement` of exactly 0.0 in both is the
positive evidence that the gates were re-derived rather than read from stored
columns.

## Step 11 — archive digests after

`passed = true`, 15/15, and the after file is **byte-identical** to the before
file. Nothing in the authoritative archive was touched.

## Results as measured

### E10b — calibrated estimated covariance

`analysis_label = single_condition_covariance_estimation_sensitivity`.

The calibration does what it was frozen to do: the diagnostics become
interpretable as estimation accuracy instead of being swamped by an
uncalibrated global scale. Against Round 9's E10 on the same condition:

| Diagnostic | E10 (Round 9) | E10b (calibrated) |
|---|---:|---:|
| product scale ratio vs oracle | 0.0103 | **0.999058** |
| Kronecker relative Frobenius error | 0.9897 | **0.036454** |
| oracle-whitening residual | 2.197 | **0.003399** |
| channel shape error | 0.028163 | 0.028163 |
| temporal shape error | 0.019673 | 0.019673 |

The two shape errors are **unchanged to all printed digits**, exactly as
expected: they were already trace-normalized and therefore already scale-free.
The calibrated estimates `v_hat = 0.0102917` sit between `v_c = 0.0102913` and
`v_t = 0.0102921`.

Held-out contrasts under estimated whitening, NFPA − Iso-MPCA, ten seeds:

| Metric | Mean | 95% CI | Excludes 0 |
|---|---:|---|:--:|
| `test_normalized_chi2` | −0.01912209 | [−0.0193400, −0.0189140] | yes |
| `test_raw_mse` | +0.00341781 | [+0.0033963, +0.0034393] | yes |

These match Round 9's E10 values (−0.0191221 and +0.00341781) to the printed
digits. That is the predicted invariance: NFPA's fit and reconstruction are
unchanged by a global rescaling of both whitening factors, so the calibration
corrects the *diagnostics* without moving the *contrasts*. The metric reversal
— better normalized chi-square, worse raw MSE — survives estimated whitening.

Degradation, estimated minus archived oracle NFPA: `normalized_chi2`
+7.905e-05 [+6.301e-05, +9.635e-05]; `raw_mse` +3.548e-06
[+1.154e-06, +6.692e-06]. Both resolved and both small.

No success threshold is imposed on either quantity; the numbers are recorded as
measured.

### E11 — corrected angles

`analysis_label = post_hoc_diagnostic_correction`. Every row carries both the
`_legacy` and the `_v2` angle columns, and the summary states that scientific
interpretation uses **only** the `_v2` columns; the legacy columns exist to
reproduce the archived diagnostic and are invalid as subspace angles for
Iso-MPCA.

Mean angle to CW-PCA, degrees:

| Cohort | n | NFPA legacy | NFPA v2 | Iso-MPCA legacy | Iso-MPCA v2 |
|---|---:|---:|---:|---:|---:|
| `frozen_0_9` | 10 | 62.8898 | **62.8898** | 87.5113 | **86.7054** |
| `replication_10_29` | 20 | 63.0267 | **63.0267** | 87.4646 | **86.7345** |
| `post_hoc_pooled_0_29` | 30 | 62.9811 | **62.9811** | 87.4801 | **86.7248** |

**NFPA's legacy and v2 angles are identical in every cohort; only Iso-MPCA
moves.** That is precisely the blast radius recorded in the Round 9 report: the
defective branch of `methods.basis_whitened` was reachable only by Iso-MPCA,
and the covariance-aware methods took a different branch. The correction shifts
Iso-MPCA by about 0.8° and leaves everything else untouched.

Corrected contrast, v2 mean angle, NFPA − Iso-MPCA:

| Cohort | Mean | 95% CI | Excludes 0 |
|---|---:|---|:--:|
| `frozen_0_9` | −23.8156 | [−24.2204, −23.3930] | yes |
| `replication_10_29` | −23.7078 | [−24.0765, −23.3396] | yes |
| `post_hoc_pooled_0_29` | −23.7437 | [−24.0264, −23.4590] | yes |

Round 9 reported this contrast as −24.62 (frozen) and −24.44 (replication) from
the legacy convention. The corrected values are about 0.8–0.9° smaller in
magnitude; direction, significance and cohort ordering are unchanged. The
pooled cohort remains labelled `post_hoc_pooled_0_29` and is not a thirty-seed
confirmatory experiment.

## Commands

```bash
nice -n 10 timeout 3h .venv/bin/python -m nfpa_confirmatory.run_calibrated_estimated_covariance \
  --implementation-freeze-commit 048f1bd96acf943ad4d143ea50333b194099f170 \
  --out "$RUNTIME/E10b_calibrated_estimated_covariance"

nice -n 10 timeout 4h .venv/bin/python -m nfpa_confirmatory.run_corrected_angles \
  --implementation-freeze-commit 048f1bd96acf943ad4d143ea50333b194099f170 \
  --out "$RUNTIME/E11_corrected_angles"
```

Both ran with `$RUNTIME` outside the worktree, so the source tree stayed clean
throughout and `source_tree_dirty_at_start` is `false` in both manifests.
Nothing entered the repository until after both runs finished.

## Deviations

One, transport only: `origin` was an HTTPS URL with no credentials on this
machine, so `git fetch --all --prune` failed. The remote was repointed to the
SSH URL, for which a working account key exists. Same remote, same content, no
history rewritten.

## Final state

`results/rps2026` is 1.3 MB. No process of ours remains, no screen session, no
stale PID file. Nothing under any authoritative archive was modified;
`archive_audit_after.json` is byte-identical to `archive_audit_before.json`.

## Required terminal fields

```text
scientific_thresholds_changed: false
authoritative_archives_overwritten: false
manuscript_edited: false
angle_v2_used_for_new_outputs: true
e10b_scale_correction_frozen_before_full_results: true
resume_gates_revalidated_from_final_rows: true
summary_regeneration_audit_passed: true
complete_repository_test_suite_passed: true
```

`e10b_scale_correction_frozen_before_full_results` is true because the
`covariance_estimation_v2` product-scale calibration lives in `FREEZE_COMMIT`
`048f1bd`, an ancestor of the executed `CONTENT_COMMIT`, and the runner
validated that ancestry before fitting anything.

`resume_gates_revalidated_from_final_rows` is true on the strict reading: both
`gate_record` blocks state the reproduction errors were recomputed from the
authoritative E1/E9 tables, and both report
`worst_stored_vs_recomputed_disagreement = 0.0`.

push_status: pending

## Synchronization addendum

evidence_commit: 64df3c7b5da61f48aaaebbf31c0e8daf963d7c67
push_status: FAILED
origin_tip_after_push: 41cb5fbe1f5f159838b23124deb352f24f7011ac (unchanged — the halted run)

### Exact error

```text
$ git push -u origin round10/remote-corrections
 ! [rejected]        round10/remote-corrections -> round10/remote-corrections (non-fast-forward)
error: failed to push some refs to 'github.com:dowlingwong/noise-weighted-subspace-reconstruction.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart.
exit 1
```

### Why, and why it was not forced

The remote `round10/remote-corrections` still carries `41cb5fb`, the halted
2026-08-19 attempt, which was built on the **old** content commit `3eee165`.
This run was built on the new `76aaff6`, which reaches `3eee165` only through
the desk commit `ef1f785`. Their merge base is `3eee165`, and `41cb5fb` is
**not** an ancestor of `64df3c7b5da61f48aaaebbf31c0e8daf963d7c67`, so the histories have genuinely
diverged and the push cannot fast-forward.

Advancing the branch therefore requires a force-push, which the dispatch
forbids ("Never force-reset `$OUTPUT_BRANCH`") and which the runbook's Step 4
lists among the operations never to perform on this branch. It is also an
irreversible, outward-facing rewrite of a published branch. It was not
performed, and synchronization is not claimed.

### What was done instead

The halted record was pushed to its own remote branch, purely additively:

```text
round10/remote-corrections-halted-20260819  ->  41cb5fbe1f5f159838b23124deb352f24f7011ac  (new remote branch)
round10/remote-corrections                  ->  41cb5fbe1f5f159838b23124deb352f24f7011ac  (unchanged)
```

Nothing was overwritten or deleted. The halted attempt is now preserved in three
places: the local branch `round10/remote-corrections-halted-20260819`, the new
remote branch of the same name, and the still-unchanged remote
`round10/remote-corrections`.

### To complete synchronization (a desk decision)

The evidence commit `64df3c7b5da61f48aaaebbf31c0e8daf963d7c67` is complete and validated locally. Since
the halted record is now independently preserved on the remote, advancing the
branch destroys nothing:

```bash
git push --force-with-lease=round10/remote-corrections:41cb5fbe1f5f159838b23124deb352f24f7011ac \
    origin round10/remote-corrections
```

The `--force-with-lease` guard makes the push refuse if the remote has moved
since this run observed it. Alternatively, publish under a fresh branch name and
leave `round10/remote-corrections` as the halted record permanently. Either is
the desk's call, not the remote agent's.
