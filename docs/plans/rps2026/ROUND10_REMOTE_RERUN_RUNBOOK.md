# Round 10 remote rerun runbook — E10b and E11 only (hardened)

_Server execution of the two frozen corrected runners._
_Standing rules: `AGENT_PROMPT_REMOTE_EXPERIMENTS.md` and the shared-server
patterns of `ROUND9_OVERNIGHT_REMOTE_RUNBOOK.md` apply in full._

## Scope — absolute

Run **E10b** (`run_calibrated_estimated_covariance`) and **E11**
(`run_corrected_angles`). Nothing else. **E6–E9 and the legacy E10 are archived
evidence**: read-only inputs to the gates, never rerun or modified. No
manuscript, figure, table, bibliography, threshold, seed-set, covariance-setting
or hypothesis edit of any kind.

## Identifiers — supplied by the dispatch prompt, not by this file

A commit cannot contain its own SHA, so this runbook does not hard-code one.
Take these from the dispatch prompt (`ROUND10_REMOTE_RERUN_AGENT_PROMPT.md`,
whose metadata-only pin commit carries the values) or from the environment:

| Variable | Meaning |
|---|---|
| `HANDOFF_BRANCH` | `codex/rps-round10-handoff-hardening` |
| `CONTENT_COMMIT` | exact commit to execute (`NEW_ROUND10_CONTENT_COMMIT`) |
| `FREEZE_COMMIT` | `NEW_ROUND10_IMPLEMENTATION_FREEZE`, passed to both runners |
| `EVIDENCE_BASE` | `c9686a0179bae0435d35bce1bbdaafb11839ef86` |
| `OUTPUT_BRANCH` | `round10/remote-corrections` |

```bash
export HANDOFF_BRANCH=codex/rps-round10-handoff-hardening
export CONTENT_COMMIT=<from dispatch prompt>
export FREEZE_COMMIT=<from dispatch prompt>
export EVIDENCE_BASE=c9686a0179bae0435d35bce1bbdaafb11839ef86
export OUTPUT_BRANCH=round10/remote-corrections
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
git merge-base --is-ancestor "$EVIDENCE_BASE"  "$CONTENT_COMMIT"
git merge-base --is-ancestor "$FREEZE_COMMIT"  "$CONTENT_COMMIT"
```

Both must succeed. A failure is a global stop.

## Step 3 — verify a clean worktree

```bash
test -z "$(git status --porcelain)"   # dirty => STOP, report the paths, do not stash
```

The runners enforce this again themselves for non-smoke runs.

## Step 4 — create or safely resume the output branch

```bash
if git show-ref --verify --quiet "refs/heads/$OUTPUT_BRANCH"; then
  # Interrupted earlier run: verify, never overwrite.
  git switch "$OUTPUT_BRANCH"
  git merge-base --is-ancestor "$CONTENT_COMMIT" HEAD   # must descend from the pin
  git status --short                                     # record what already exists
  # Record which experiment directories exist and their manifest "status" fields.
else
  git switch --detach "$CONTENT_COMMIT"
  git switch -c "$OUTPUT_BRANCH"
fi
```

**Never** `git branch -f`, `reset --hard`, `checkout -B`, or delete this branch.
If it exists but does not descend from `$CONTENT_COMMIT`, stop and report.

## Step 5 — record all 15 archive digests (before)

```bash
export PYTHONPATH=src:experiments/synthetic
.venv/bin/python - <<'PY'
import json
from experiments.synthetic.nfpa_confirmatory import r10_validation as rv
audit = rv.audit_authoritative_archive()
print(json.dumps(audit, indent=2, sort_keys=True))
assert audit["passed"], "ARCHIVE CHANGED BEFORE EXECUTION — GLOBAL STOP"
PY
```

Save the output as `results/rps2026/Round10_remote/archive_audit_before.json`.

## Step 6 — the complete repository test suite

This is the real gate. Run the **whole suite**, from the repository root,
without `NFPA_CONFIRMATORY_FAST`, so the full-config Round 10 control and the
TIDMAD tests actually execute:

```bash
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nice -n 10 timeout 90m .venv/bin/python -m pytest -q 2>&1 | tee results/rps2026/Round10_remote/pytest_full.log
```

Record the **collected test count** and the final line. Any failure is a global
stop. (A short focused run beforehand is fine, but it is not the suite and must
not be described as one.)

## Step 7 — server smoke on reserved seeds

```bash
nice -n 10 timeout 30m .venv/bin/python -m nfpa_confirmatory.run_corrected_angles \
  --smoke --out results/rps2026/Round10_smoke/E11_server
nice -n 10 timeout 30m .venv/bin/python -m nfpa_confirmatory.run_calibrated_estimated_covariance \
  --smoke --out results/rps2026/Round10_smoke/E10b_server
```

Smoke manifests carry `evidence_class = implementation_check_not_scientific_evidence`.
They are never cited as evidence.

## Step 8 — E10b, then E11, sequentially, with the explicit freeze SHA

```bash
nice -n 10 timeout 3h .venv/bin/python -m nfpa_confirmatory.run_calibrated_estimated_covariance \
  --implementation-freeze-commit "$FREEZE_COMMIT" \
  --out results/rps2026/E10b_calibrated_estimated_covariance

nice -n 10 timeout 4h .venv/bin/python -m nfpa_confirmatory.run_corrected_angles \
  --implementation-freeze-commit "$FREEZE_COMMIT" \
  --out results/rps2026/E11_corrected_angles
```

Do not parallelize seeds. Caps may be **raised only before launch**, justified
from measured smoke or archived per-seed timing, and never reduced after an
outcome. The runners refuse to start without the freeze SHA, refuse a dirty
tree, and record `implementation_freeze_commit` and `execution_content_commit`
separately.

## Step 9 — resume only through the strengthened validation

If a cap or interruption stops a run, resume with the same command plus
`--resume`. Every gate is re-derived from the **final** rows — fresh and reused
alike — so a stale or altered shard cannot inherit a passing verdict. An
interrupted run leaves `status = in_progress`; only a fully validated run
writes `status = complete`. **Never cite an `in_progress` directory as a
result.**

**Continuation rule:** an experiment-specific *scientific* outcome, however
unfavorable, is never a reason to stop, rerun, or adjust anything. A shared
configuration, test, provenance, or archive-integrity failure stops everything.

## Step 10 — regenerate summaries and audits from the raw archived outputs

Both runners already perform the deterministic summary-regeneration audit and
write `summary_digest`. Confirm in each manifest:

```bash
.venv/bin/python - <<'PY'
import json, pathlib
for name in ("E10b_calibrated_estimated_covariance", "E11_corrected_angles"):
    m = json.loads(pathlib.Path("results/rps2026", name, "manifest.json").read_text())
    print(name, m["status"], m["summary_regeneration_audit"], m.get("summary_digest"))
    assert m["status"] == "complete"
    assert m["summary_regeneration_audit"]["passed"] is True
PY
```

## Step 11 — recheck all 15 archive digests (after)

Repeat Step 5, saving `archive_audit_after.json`. Any difference from
`archive_audit_before.json` is a global stop, reported loudly.

## Step 12 — write the report BEFORE committing

Write `docs/plans/rps2026/ROUND10_REMOTE_REPORT.md` now, while the evidence is
in front of you: identifiers actually used; machine state before/after; the
complete-suite collected count and result; smoke paths; per-experiment wall
times; every gate passed or failed; digests before/after; the exact commands
run; and the push outcome (filled in after Step 15, or marked as failed).

The previous revision of this runbook committed before writing the report,
which invited a report reconstructed from memory. The report is part of the
evidence, not a summary of it.

## Steps 13–14 — check, then commit results and report together

```bash
git status --short          # only new files under results/rps2026/{E10b_*,E11_*,Round10_smoke,Round10_remote}
git diff --check
git add -A && git commit -m "round10 remote: E10b and E11 corrected results with report"
```

## Steps 15–17 — push and verify

```bash
git push -u origin "$OUTPUT_BRANCH"
git fetch origin "$OUTPUT_BRANCH"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/$OUTPUT_BRANCH)"
```

If credentials prevent the push: commit locally, report the exact local SHA and
the exact failure text, and mark synchronization as NOT achieved. Never claim a
push that did not happen.

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

Report the true value of every field. If any is false, explain it and end with
`ROUND10_REMOTE_CORRECTIONS_PARTIAL_WITH_RECORDED_FAILURES`. Print
`ROUND10_REMOTE_CORRECTIONS_READY` only when every field above is true.
