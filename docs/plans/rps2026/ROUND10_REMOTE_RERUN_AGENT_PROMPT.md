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
CONTENT_COMMIT   = f9c844d967b4d268d1c5beaf5c52f59edacbb199
FREEZE_COMMIT    = 34c47ae71a79b9569457dc54915e36ba5791ebb9
EVIDENCE_BASE    = c9686a0179bae0435d35bce1bbdaafb11839ef86
OUTPUT_BRANCH    = round10/remote-corrections
```

Export them before starting:

```bash
export HANDOFF_BRANCH=codex/rps-round10-handoff-hardening
export CONTENT_COMMIT=f9c844d967b4d268d1c5beaf5c52f59edacbb199
export FREEZE_COMMIT=34c47ae71a79b9569457dc54915e36ba5791ebb9
export EVIDENCE_BASE=c9686a0179bae0435d35bce1bbdaafb11839ef86
export OUTPUT_BRANCH=round10/remote-corrections
```

## Orders

Follow `docs/plans/rps2026/ROUND10_REMOTE_RERUN_RUNBOOK.md` exactly, all
seventeen steps, in order. It is your entire task. Also honor
`AGENT_PROMPT_REMOTE_EXPERIMENTS.md` (machine rules) and the shared-server
safety patterns of `ROUND9_OVERNIGHT_REMOTE_RUNBOOK.md` (reference only — do
not execute its experiments).

Points that have previously gone wrong and must not go wrong again:

1. **Run the complete repository test suite** (`pytest -q` from the repository
   root, no `NFPA_CONFIRMATORY_FAST`, 90-minute cap). Record the collected
   count. A focused suite is not the suite.
2. **Pass `--implementation-freeze-commit "$FREEZE_COMMIT"`** to both runners.
   They fail closed without it; do not work around that.
3. **Never overwrite or force-reset `$OUTPUT_BRANCH`.** If it exists from an
   interrupted run, verify its ancestry from `$CONTENT_COMMIT` and resume.
4. **Write `ROUND10_REMOTE_REPORT.md` before the final commit**, not after.
5. **Record all 15 archive digests before and after**, using
   `r10_validation.audit_authoritative_archive()`.
6. **Never cite an `in_progress` manifest as a result.**
7. **Never claim a push that did not occur.**

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

Then exactly one terminal line — `ROUND10_REMOTE_CORRECTIONS_READY` only if
every field is true and the push succeeded; otherwise
`ROUND10_REMOTE_CORRECTIONS_PARTIAL_WITH_RECORDED_FAILURES`.
