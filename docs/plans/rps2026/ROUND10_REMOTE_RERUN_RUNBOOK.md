# Round 10 remote rerun runbook — E10b and E11 only (hardened, rev 2)

_Server execution of the two frozen corrected runners._
_Standing rules: `AGENT_PROMPT_REMOTE_EXPERIMENTS.md` and the shared-server
patterns of `ROUND9_OVERNIGHT_REMOTE_RUNBOOK.md` apply in full._

## Scope — absolute

Run **E10b** (`run_calibrated_estimated_covariance`) and **E11**
(`run_corrected_angles`). Nothing else. **E6–E9 and the legacy E10 are archived
evidence**: read-only inputs to the gates, never rerun or modified. No
manuscript, figure, table, bibliography, threshold, seed-set, covariance-setting
or hypothesis edit of any kind.

## The one structural rule: run OUTSIDE the worktree

The non-smoke runners require a **clean source tree**. Anything written into the
repository before they start — an audit file, a pytest log, a smoke directory —
would make them refuse to run. So **every byte this session produces goes to
`$RUNTIME` outside the Git worktree**, and results are copied into the
repository only after both scientific runs have finished.

```bash
export RUNTIME=${RUNTIME:-$HOME/round10_runtime_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RUNTIME"
export R10_RUNTIME_ROOT="$RUNTIME"      # also permits an out-of-tree smoke root
```

## Identifiers — supplied by the dispatch prompt

A commit cannot contain its own SHA, so this file hard-codes none. Export the
values from `ROUND10_REMOTE_RERUN_AGENT_PROMPT.md`:

```bash
export HANDOFF_BRANCH=codex/rps-round10-handoff-hardening
export CONTENT_COMMIT=<from dispatch prompt>
export FREEZE_COMMIT=<from dispatch prompt>
export EVIDENCE_BASE=c9686a0179bae0435d35bce1bbdaafb11839ef86
export OUTPUT_BRANCH=round10/remote-corrections
```

Use a strict shell throughout:

```bash
set -euo pipefail
```

## Step 1 — fetch and verify the exact handoff

```bash
cd ~/noise-weighted-subspace-reconstruction
git fetch --all --prune || echo "ROUND10_HANDOFF_FETCH_FAILED"   # record, never hide
git cat-file -e "$CONTENT_COMMIT^{commit}"
git cat-file -e "$FREEZE_COMMIT^{commit}"
```

## Step 2 — verify ancestry

```bash
git merge-base --is-ancestor "$EVIDENCE_BASE" "$CONTENT_COMMIT"
git merge-base --is-ancestor "$FREEZE_COMMIT" "$CONTENT_COMMIT"
```

Both must succeed; a failure is a global stop.

## Step 3 — verify a clean worktree

```bash
test -z "$(git status --porcelain)"   # dirty => STOP, report the paths, never stash
```

## Step 4 — create or safely resume the output branch

```bash
if git show-ref --verify --quiet "refs/heads/$OUTPUT_BRANCH"; then
  git switch "$OUTPUT_BRANCH"
  git merge-base --is-ancestor "$CONTENT_COMMIT" HEAD   # must descend from the pin
  git status --short                                     # record what already exists
else
  git switch --detach "$CONTENT_COMMIT"
  git switch -c "$OUTPUT_BRANCH"
fi
```

**Never** `git branch -f`, `reset --hard`, `checkout -B`, or delete this branch.
If it exists but does not descend from `$CONTENT_COMMIT`, stop and report.

## Step 5 — record all 15 archive digests (before), out of tree

```bash
export PYTHONPATH=src:experiments/synthetic
.venv/bin/python - > "$RUNTIME/archive_audit_before.json" <<'PY'
import json
from experiments.synthetic.nfpa_confirmatory import r10_validation as rv
audit = rv.audit_authoritative_archive()
print(json.dumps(audit, indent=2, sort_keys=True))
assert audit["passed"], "ARCHIVE CHANGED BEFORE EXECUTION — GLOBAL STOP"
PY
```

## Step 6 — the complete repository test suite (exit code propagated)

```bash
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
set +e
nice -n 10 timeout 90m .venv/bin/python -m pytest -q \
  > "$RUNTIME/pytest_full.log" 2>&1
PYTEST_STATUS=$?
set -e
tail -5 "$RUNTIME/pytest_full.log"
grep -E "tests? collected|passed|failed" "$RUNTIME/pytest_full.log" | tail -3
test "$PYTEST_STATUS" -eq 0 || { echo "COMPLETE SUITE FAILED (exit $PYTEST_STATUS) — GLOBAL STOP"; exit 1; }
```

**Do not pipe pytest into `tee`.** A pipeline reports the exit status of the
last command, so `pytest ... | tee log` returns `tee`'s success and a failed or
timed-out suite passes silently. Redirect to a file and test the captured status
explicitly, as above. Record the collected count and the final line.

The log lives in `$RUNTIME`, not in the repository, so the suite does not dirty
the tree. (The suite no longer rewrites `src/NFPA/nfpa_demo_results.png`: the
demo honours `NFPA_DEMO_OUT_DIR` and the test redirects it.)

## Step 7 — E10b, then E11, sequentially, out of tree, with the freeze SHA

```bash
nice -n 10 timeout 3h .venv/bin/python -m nfpa_confirmatory.run_calibrated_estimated_covariance \
  --implementation-freeze-commit "$FREEZE_COMMIT" \
  --out "$RUNTIME/E10b_calibrated_estimated_covariance"

nice -n 10 timeout 4h .venv/bin/python -m nfpa_confirmatory.run_corrected_angles \
  --implementation-freeze-commit "$FREEZE_COMMIT" \
  --out "$RUNTIME/E11_corrected_angles"
```

Sequential; never parallelize seeds. Caps may be **raised only before launch**,
justified from measured smoke or archived per-seed timing, and never reduced
after an outcome.

Each runner: refuses to start without the freeze SHA; refuses a dirty source
tree; records `implementation_freeze_commit`, `execution_content_commit` and
`source_tree_dirty_at_start` separately; registers every shard with its SHA-256;
re-derives every gate from the final rows with the reproduction errors
**recomputed from the authoritative E1/E9 tables**; runs the deterministic
summary-regeneration audit; and writes the completed manifest last.

## Step 8 — resume only through the strengthened validation

If a cap or interruption stops a run, resume with the same command plus
`--resume`. Resume rejects an unregistered shard, a digest mismatch, or shards
from a different freeze/execution identity. An interrupted run leaves
`status = in_progress`; **never cite an `in_progress` directory as a result.**

**Continuation rule:** an experiment-specific *scientific* outcome, however
unfavorable, is never a reason to stop, rerun, or adjust anything. A shared
configuration, test, provenance, or archive-integrity failure stops everything.

## Step 9 — server smoke (after the scientific runs)

```bash
nice -n 10 timeout 30m .venv/bin/python -m nfpa_confirmatory.run_corrected_angles \
  --smoke --out "$RUNTIME/Round10_smoke/E11_server"
nice -n 10 timeout 30m .venv/bin/python -m nfpa_confirmatory.run_calibrated_estimated_covariance \
  --smoke --out "$RUNTIME/Round10_smoke/E10b_server"
```

Smoke runs last precisely because it is an implementation check, not evidence;
its manifests carry `evidence_class = implementation_check_not_scientific_evidence`.

## Step 10 — verify the completed manifests

```bash
.venv/bin/python - <<PY
import json, pathlib
for name in ("E10b_calibrated_estimated_covariance", "E11_corrected_angles"):
    m = json.loads((pathlib.Path("$RUNTIME")/name/"manifest.json").read_text())
    assert m["status"] == "complete", (name, m["status"])
    assert m["summary_regeneration_audit"]["passed"] is True
    assert m["provenance"]["implementation_freeze_commit"] == "$FREEZE_COMMIT"
    assert m["provenance"]["source_tree_dirty_at_start"] is False
    print(name, "OK", m["summary_digest"])
PY
```

## Step 11 — recheck all 15 archive digests (after)

Repeat Step 5 into `$RUNTIME/archive_audit_after.json` and diff against the
before file. Any difference is a global stop, reported loudly.

## Step 12 — copy results into the repository

Only now does anything enter the worktree:

```bash
cp -a "$RUNTIME/E10b_calibrated_estimated_covariance" results/rps2026/
cp -a "$RUNTIME/E11_corrected_angles"                 results/rps2026/
mkdir -p results/rps2026/Round10_remote
cp -a "$RUNTIME/archive_audit_before.json" "$RUNTIME/archive_audit_after.json" \
      "$RUNTIME/pytest_full.log" results/rps2026/Round10_remote/
cp -a "$RUNTIME/Round10_smoke" results/rps2026/ 2>/dev/null || true
git status --short
git diff --check
```

## Step 13 — write the report BEFORE the evidence commit

Write `docs/plans/rps2026/ROUND10_REMOTE_REPORT.md` now: identifiers actually
used; machine state; the complete-suite collected count, exit status and result;
per-experiment wall times; every gate passed or failed; digests before/after;
the exact commands; and

```text
push_status: pending
```

The report is part of the evidence, not a recollection of it.

## Step 14 — commit the evidence and report together

```bash
git add -A && git commit -m "round10 remote: E10b and E11 corrected results with report"
EVIDENCE_COMMIT=$(git rev-parse HEAD)
```

## Step 15 — push and verify

```bash
git push -u origin "$OUTPUT_BRANCH"
git fetch origin "$OUTPUT_BRANCH"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/$OUTPUT_BRANCH)"
```

## Step 16 — synchronization addendum, then push it

Because the push status could not honestly be known when the report was
written, record it in an addendum rather than amending the evidence commit:

```bash
cat >> docs/plans/rps2026/ROUND10_REMOTE_REPORT.md <<EOF

## Synchronization addendum

evidence_commit: $EVIDENCE_COMMIT
push_status: verified
origin_tip_after_push: $(git rev-parse origin/$OUTPUT_BRANCH)
EOF
git add -A && git commit -m "round10 remote: synchronization addendum"
git push origin "$OUTPUT_BRANCH"
```

If the push failed, write `push_status: FAILED`, the exact error, and the local
`$EVIDENCE_COMMIT`; commit the addendum locally; never claim synchronization.

## Step 17 — verify the final remote tip

```bash
git fetch origin "$OUTPUT_BRANCH"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/$OUTPUT_BRANCH)" \
  && echo "REMOTE TIP VERIFIED"
```

## Required literal terminal fields

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

Report each field's true value. If any is false, explain it and end with
`ROUND10_REMOTE_CORRECTIONS_PARTIAL_WITH_RECORDED_FAILURES`. Print
`ROUND10_REMOTE_CORRECTIONS_READY` only when every field is true **and** the
remote tip is verified.
