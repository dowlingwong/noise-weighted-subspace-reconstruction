# E12 — calibration-size sweep: remote runbook

**Experiment:** `E12_calibration_size_sweep`
**Evidence class:** `post_hoc_exploratory_not_confirmatory`
**Estimated cost:** ~5 wall-hours on 56 cores (5 conditions × 10 seeds × 2 methods,
~358 s per row, ~2 CPU-hours per condition with the observed ~2× parallel speed-up)

## What this is, and what it is not

E10b showed the rung-3 reversal survives a covariance estimated from
`N_cal = 2048` noise-only events, at one calibration size. E12 turns that single
point into a curve over `N_cal ∈ {256, 512, 1024, 2048, 8192}`, changing nothing
else.

**It is post-hoc exploratory.** It was designed after the confirmatory result
existed. It supports the estimated-covariance limitation with more evidence; it
does not remove it. §5 of the paper must continue to say that no confirmatory
robustness boundary was established, and this sweep must never be described as
a robustness result.

There is no threshold, no predicted direction and no success criterion. A curve
that degrades badly at small `N_cal`, or one that is non-monotone, is a result
to record. **Do not retune anything to rescue an unfavourable curve.** If the
reversal fails at some size, that is the most interesting outcome available and
it goes in the report unedited.

## The two gates that make this checkable

1. **The anchor.** `config_hash` covers the whole frozen configuration, so the
   `N_cal = 2048` condition necessarily has E10b's hash `98001a62855cb9a5`. The
   runner requires that condition's NFPA rows to reproduce archived E10b to
   `1e-9` relative. If the code has drifted from the runner that produced the
   archived evidence, this fails and nothing is written.
2. **The control.** Iso-MPCA never sees the estimated covariance, so it must be
   identical at every calibration size and equal to archived E1. A control that
   moves with `N_cal` means estimation is leaking into the arm that defines the
   contrast, and every number in the sweep is void.

Both are enforced by the runner. Neither is advisory.

## Pinned identifiers

Supplied by the dispatch prompt, not hard-coded here — a commit cannot contain
its own SHA.

```text
HANDOFF_BRANCH   = <supplied>
CONTENT_COMMIT   = <supplied>
FREEZE_COMMIT    = <supplied>
OUTPUT_BRANCH    = <supplied>
```

## Step 0 — run outside the worktree

The non-smoke runner requires a clean source tree, so nothing may be written
into the repository until the sweep has finished.

```bash
set -euo pipefail
export RUNTIME=${RUNTIME:-$HOME/e12_runtime_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RUNTIME"
export R10_RUNTIME_ROOT="$RUNTIME"
export PYTHONPATH=src:experiments/synthetic
```

## Step 1 — checkout and verify

```bash
cd ~/noise-weighted-subspace-reconstruction
git fetch origin
git checkout "$CONTENT_COMMIT"
test "$(git rev-parse HEAD)" = "$CONTENT_COMMIT" && echo CHECKOUT_VERIFIED
git merge-base --is-ancestor "$FREEZE_COMMIT" HEAD && echo FREEZE_IS_ANCESTOR
git status --porcelain | tee "$RUNTIME/tree_state.txt"   # must be empty
```

## Step 2 — machine preflight

```bash
{ hostname; uname -a; nproc; date -Is; uptime; free -g; df -h /home /ceph; } \
  > "$RUNTIME/machine_preflight.txt" 2>&1
```

## Step 3 — archive audit BEFORE anything runs

```bash
python -c "
from nfpa_confirmatory import r10_validation as V
import json; json.dump(V.audit_authoritative_archive(), open('$RUNTIME/archive_before.json','w'), indent=1)
print('archive files:', V.audit_authoritative_archive()['n_checked'])"
```

Expect `15`.

## Step 4 — the complete test suite

Never pipe pytest into `tee`: the pipeline returns `tee`'s status and hides a
failed or timed-out suite.

```bash
python -m pytest -q > "$RUNTIME/pytest_full.log" 2>&1
PYTEST_STATUS=$?
echo "PYTEST_STATUS=$PYTEST_STATUS" | tee "$RUNTIME/pytest_status.txt"
test "$PYTEST_STATUS" -eq 0
tail -3 "$RUNTIME/pytest_full.log"
```

The Round 12 suite is `experiments/synthetic/nfpa_confirmatory/tests/test_round12_sweep.py`
(26 tests). A focused suite is not the suite.

## Step 5 — smoke, out of tree

```bash
python -m nfpa_confirmatory.run_calibration_sweep \
  --smoke --sizes 64 128 \
  --out "$RUNTIME/Round10_smoke/E12" \
  > "$RUNTIME/e12_smoke.log" 2>&1
echo "SMOKE_STATUS=$?"
```

Smoke carries `evidence_class=implementation_check_not_scientific_evidence` and
seeds 910000/910001. It is never evidence.

## Step 6 — the sweep

```bash
python -m nfpa_confirmatory.run_calibration_sweep \
  --implementation-freeze-commit "$FREEZE_COMMIT" \
  --out "$RUNTIME/E12_calibration_size_sweep" \
  > "$RUNTIME/e12.log" 2>&1
echo "E12_STATUS=$?"
tail -20 "$RUNTIME/e12.log"
```

Notes:

* Do **not** pass `--sizes`. The full frozen set is the sweep; a subset writes
  no `sweep_summary.json` by design.
* If it dies part-way, rerun the identical command with `--resume`. Resume
  rejects unregistered shards, digest mismatches, and shards from another
  freeze/execution identity. **Never delete a shard to make resume proceed** —
  if resume refuses, report the refusal.
* The runner fails closed without `--implementation-freeze-commit`. Do not work
  around that.

## Step 7 — verify the two gates fired

```bash
python - <<'PY'
import json, pathlib, os
root = pathlib.Path(os.environ["RUNTIME"]) / "E12_calibration_size_sweep"
sm = json.load(open(root / "sweep_manifest.json"))
print("status:", sm["status"])
print("control gate:", sm["control_gate"])
print("summary audit:", sm["summary_audit"])
anchor = json.load(open(root / "n_cal_2048" / "manifest.json"))["anchor_reproduction"]
print("anchor reproduction:", anchor)
assert sm["status"] == "complete"
assert sm["control_gate"]["passed"] is True
assert anchor["passed"] is True and anchor["comparisons"] == 20
print("GATES_OK")
PY
```

## Step 8 — archive audit AFTER

```bash
python -c "
from nfpa_confirmatory import r10_validation as V
import json; json.dump(V.audit_authoritative_archive(), open('$RUNTIME/archive_after.json','w'), indent=1)"
python -c "
import json
a=json.load(open('$RUNTIME/archive_before.json'))['digests']
b=json.load(open('$RUNTIME/archive_after.json'))['digests']
print('archive unchanged:', a==b)
assert a==b"
```

## Step 9 — copy results in, then commit

Only now does anything enter the worktree.

```bash
git checkout -B "$OUTPUT_BRANCH" "$CONTENT_COMMIT"
mkdir -p results/rps2026
cp -r "$RUNTIME/E12_calibration_size_sweep" results/rps2026/
mkdir -p results/rps2026/Round12_remote
cp "$RUNTIME"/{machine_preflight.txt,archive_before.json,archive_after.json,pytest_full.log,pytest_status.txt,e12.log,e12_smoke.log} \
   results/rps2026/Round12_remote/
git add results/rps2026/E12_calibration_size_sweep results/rps2026/Round12_remote
```

Write `docs/plans/rps2026/E12_REMOTE_REPORT.md` **before** the evidence commit,
with `push_status: pending`. Then commit, push, and add a synchronization
addendum carrying the verified SHA. Never amend the evidence commit to backfill
a push result.

```bash
git commit -m "round12 remote: E12 calibration-size sweep (post-hoc exploratory)"
git push -u origin "$OUTPUT_BRANCH"
```

## Terminal output

End the report with the true value of every field:

```text
scientific_thresholds_changed: false
authoritative_archives_overwritten: false
manuscript_edited: false
evidence_class_is_post_hoc_exploratory: true
anchor_reproduces_archived_e10b: <true|false>
control_is_covariance_independent: <true|false>
sweep_summary_regenerates: <true|false>
complete_repository_test_suite_passed: <true|false>
all_five_frozen_sizes_completed: <true|false>
```

Then exactly one terminal line — `E12_SWEEP_READY` only if every field is true
and the push succeeded; otherwise `E12_SWEEP_PARTIAL_WITH_RECORDED_FAILURES`.

## What must not happen

* Do not edit `config.py`, any archived result, the manuscript, or any
  threshold. E12 adds files; it changes nothing.
* Do not remove `2048` from the frozen sizes. It is the only tie to archived
  evidence.
* Do not describe the output as robustness, as confirmatory, or as removing a
  limitation.
* Do not claim a push that did not occur.
