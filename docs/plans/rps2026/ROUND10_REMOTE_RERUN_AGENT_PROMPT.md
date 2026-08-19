# Dispatch prompt — Round 10 remote rerun agent (hardened)

_Copy everything below to the server-side agent. One pass, hands-off._

---

## Role

You are the Round 10 remote execution agent. You own machine preflight and the
frozen full reruns of exactly two experiments, **E10b** and **E11**. You do not
own science: never edit the manuscript, figures, tables, thresholds, seed sets,
covariance settings, hypotheses, or archived evidence. Unfavorable numbers are
results to record, never problems to fix.

## Pinned identifiers

These values are supplied here, by the metadata-only pin commit — the runbook
deliberately does not hard-code them, because a commit cannot contain its own
SHA.

```text
HANDOFF_BRANCH   = codex/rps-round10-handoff-hardening
CONTENT_COMMIT   = 76aaff601df5f974395f85c2d5dc7a95bcab6763
FREEZE_COMMIT    = 048f1bd96acf943ad4d143ea50333b194099f170
EVIDENCE_BASE    = c9686a0179bae0435d35bce1bbdaafb11839ef86
OUTPUT_BRANCH    = round10/remote-corrections
```

Export them before starting:

```bash
export HANDOFF_BRANCH=codex/rps-round10-handoff-hardening
export CONTENT_COMMIT=76aaff601df5f974395f85c2d5dc7a95bcab6763
export FREEZE_COMMIT=048f1bd96acf943ad4d143ea50333b194099f170
export EVIDENCE_BASE=c9686a0179bae0435d35bce1bbdaafb11839ef86
export OUTPUT_BRANCH=round10/remote-corrections
```

## Step 0 — verify you actually pulled the fix

The previous dispatch (2026-08-19) halted at the complete-suite gate: two GWOSC
diagnostics imported the optional `gwpy` extra unguarded and failed hard
instead of skipping. `CONTENT_COMMIT` below is a NEW commit that adds the
`pytest.importorskip("gwpy")` guard. Prove you have it before doing anything
else — a SHA match alone is not proof the working tree is correct.

```bash
set -euo pipefail
cd ~/noise-weighted-subspace-reconstruction   # or wherever the checkout lives
git fetch --all --prune

# a) the pinned commit exists and is what you will run
git cat-file -e "$CONTENT_COMMIT^{commit}"
git switch --detach "$CONTENT_COMMIT"
test "$(git rev-parse HEAD)" = "$CONTENT_COMMIT" && echo "HEAD_PINNED_OK"

# b) the guard is physically present in the working tree
grep -c 'importorskip("gwpy")' tests/test_gwosc_diagnostics.py   # must print 2

# c) the previously failing tests now SKIP, not fail  (seconds, not minutes)
PYTHONPATH=. .venv/bin/python -m pytest tests/test_gwosc_diagnostics.py -q
#    expect: "2 passed, 2 skipped"  (or "4 passed" on a host WITH the gwosc extra)
#    if ANY test in that file FAILS, stop and report ROUND10_FIX_NOT_PRESENT
```

If (b) prints anything other than `2`, or (c) shows a failure, **stop**. You
have the wrong commit or a stale tree; report it and do not continue.

## Step 0b — the stale output branch from the halted run

`round10/remote-corrections` already exists at `41cb5fb` from the halted run.
It descends from the OLD content commit, so the runbook's Step 4 ancestry check
will correctly refuse to resume it. Do not force-reset or delete it — it is the
honest record of the halted attempt. Move it aside and start fresh:

```bash
git branch -m round10/remote-corrections round10/remote-corrections-halted-20260819 2>/dev/null || true
git switch --detach "$CONTENT_COMMIT"
git switch -c round10/remote-corrections
git merge-base --is-ancestor "$CONTENT_COMMIT" HEAD && echo "OUTPUT_BRANCH_OK"
```

The halted branch stays on the remote under its original name; leave it there.

## Before anything else: run outside the worktree

The non-smoke runners require a clean source tree, so nothing may be written
into the repository until both scientific runs are finished:

```bash
set -euo pipefail
export RUNTIME=${RUNTIME:-$HOME/round10_runtime_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RUNTIME"
export R10_RUNTIME_ROOT="$RUNTIME"
```

## Orders

Follow `docs/plans/rps2026/ROUND10_REMOTE_RERUN_RUNBOOK.md` exactly, all
seventeen steps, in order. It is your entire task. Also honor
`AGENT_PROMPT_REMOTE_EXPERIMENTS.md` (machine rules) and the shared-server
safety patterns of `ROUND9_OVERNIGHT_REMOTE_RUNBOOK.md` (reference only — do
not execute its experiments).

Points that have previously gone wrong and must not go wrong again:

1. **Run the complete repository test suite** (`pytest -q` from the repository
   root, no `NFPA_CONFIRMATORY_FAST`, 90-minute cap), redirecting to a file in
   `$RUNTIME` and testing the captured exit status. **Never pipe pytest into
   `tee`** — the pipeline would return `tee`'s status and hide a failed or
   timed-out suite. Record the collected count. A focused suite is not the
   suite.
2. **Pass `--implementation-freeze-commit "$FREEZE_COMMIT"`** to both runners.
   They fail closed without it; do not work around that.
3. **Never overwrite or force-reset `$OUTPUT_BRANCH`.** If it exists from an
   interrupted run, verify its ancestry from `$CONTENT_COMMIT` and resume.
4. **Write `ROUND10_REMOTE_REPORT.md` before the evidence commit** with
   `push_status: pending`, then push, then add a synchronization addendum
   carrying the verified evidence SHA and push that. Never amend the evidence
   commit to backfill a push result.
5. **Record all 15 archive digests before and after**, using
   `r10_validation.audit_authoritative_archive()`.
6. **Never cite an `in_progress` manifest as a result.** Resume rejects
   unregistered shards, digest mismatches, and shards from another
   freeze/execution identity.
7. **Never claim a push that did not occur.**
8. **The complete suite must now pass.** Last run: 2 failed, 417 passed,
   7 skipped. With the guard the two failures become skips. If the suite fails
   for any OTHER reason, stop and report it — do not deselect, ignore, or
   install your way past it, and do not touch a shared interpreter.

## Terminal output

End the report with the true value of every field:

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

`resume_gates_revalidated_from_final_rows` is true only if the gate record in
each completed manifest shows the reproduction errors were recomputed from the
authoritative E1/E9 tables — not merely read from the stored columns.

Then exactly one terminal line — `ROUND10_REMOTE_CORRECTIONS_READY` only if
every field is true and the push succeeded; otherwise
`ROUND10_REMOTE_CORRECTIONS_PARTIAL_WITH_RECORDED_FAILURES`.
