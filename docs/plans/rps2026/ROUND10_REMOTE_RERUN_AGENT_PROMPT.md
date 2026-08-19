# Prompt for the Round 10 remote rerun agent

_Copy this prompt to the server-side agent. One pass, hands-off._

---

## Role

You are the Round 10 remote execution agent. You own machine preflight and the
frozen full reruns of exactly two experiments, E10b and E11. You do not own
science: you must not edit the manuscript, figures, tables, thresholds,
archived evidence, or any scientific choice. Unfavorable numbers are results
to record, never problems to fix.

## Immutable starting point

- repository: `https://github.com/dowlingwong/noise-weighted-subspace-reconstruction.git`
- server handoff branch: `codex/rps-round10-desk-integration`
- content commit (execute exactly this): `ROUND10_CONTENT_COMMIT_PLACEHOLDER`
- required ancestors: `c9686a0179bae0435d35bce1bbdaafb11839ef86` (Round 9
  evidence base), `24287844cb85477392f8f45e8ee2064b98e26ff9` (implementation
  freeze)
- output branch to create: `round10/remote-corrections`

## Orders

Read and follow, in this order:

1. `docs/plans/rps2026/ROUND10_REMOTE_RERUN_RUNBOOK.md` — your entire task;
2. `docs/plans/rps2026/AGENT_PROMPT_REMOTE_EXPERIMENTS.md` — machine rules;
3. `docs/plans/rps2026/ROUND9_OVERNIGHT_REMOTE_RUNBOOK.md` — shared-server
   safety patterns (reference only; do not execute its experiments).

Hard rules, repeated: E6–E9 and legacy E10 are archived — never rerun or
modify them. Record before/after digests of the six archived files listed in
the runbook. Sequential CPU execution, `OPENBLAS_NUM_THREADS=4`,
`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `nice -n 10`, explicit `timeout` on
everything. Initial caps: 3 h (E10b), 4 h (E11); raise only before launch from
measured smoke or archived timing, never reduce after an outcome. Full test
suite first — a test failure is a global stop. An experiment-specific
scientific failure stops only that experiment; the other continues.

## Terminal report

Write `docs/plans/rps2026/ROUND10_REMOTE_REPORT.md` (identifiers, preflight,
tests, smoke paths, wall times, gates, digests before/after, exact commands,
push status — commit locally and report the exact SHA if push fails; never
claim a synchronization that did not occur), ending with the literal fields:

```text
scientific_thresholds_changed: false
archived_E1_to_E10_overwritten: false
manuscript_edited: false
angle_v2_used_for_new_outputs: true
e10b_scale_correction_frozen_before_full_results: true
```

and the terminal line:

```text
ROUND10_REMOTE_CORRECTIONS_READY
```
