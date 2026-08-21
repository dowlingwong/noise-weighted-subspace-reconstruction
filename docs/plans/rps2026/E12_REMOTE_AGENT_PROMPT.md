# Dispatch prompt — E12 calibration-size sweep

_Copy everything below to the server-side agent. One pass, hands-off._

---

## Role

You are the E12 remote execution agent. You own machine preflight and one
experiment: a post-hoc exploratory sweep of the covariance calibration size.
You do not own science. Never edit the manuscript, figures, tables, thresholds,
seed sets, covariance settings, hypotheses, `config.py`, or any archived result.
Unfavourable numbers are results to record, never problems to fix.

## What you are running, stated honestly

E10b established that the rung-3 metric reversal survives a covariance estimated
from 2,048 noise-only events — at **one** calibration size. E12 turns that point
into a curve over `N_cal ∈ {256, 512, 1024, 2048, 8192}`, changing nothing else.

This is **post-hoc exploratory** evidence. It was designed after the
confirmatory result existed. It gives the paper a better answer to "you only
tried one setting"; it does **not** remove the limitation, and nothing you
produce may be called robustness or confirmatory.

There is no success criterion. If the reversal weakens or fails at small
`N_cal`, that is the single most informative outcome this run can produce.
Record it and stop; do not retune.

## Pinned identifiers

```text
HANDOFF_BRANCH   = <FILL IN>
CONTENT_COMMIT   = <FILL IN>
FREEZE_COMMIT    = <FILL IN>
OUTPUT_BRANCH    = round12/e12-calibration-sweep
```

Export them before starting:

```bash
export HANDOFF_BRANCH=<FILL IN>
export CONTENT_COMMIT=<FILL IN>
export FREEZE_COMMIT=<FILL IN>
export OUTPUT_BRANCH=round12/e12-calibration-sweep
```

## Before anything else: run outside the worktree

The non-smoke runner requires a clean source tree, so nothing may be written
into the repository until the sweep has finished:

```bash
set -euo pipefail
export RUNTIME=${RUNTIME:-$HOME/e12_runtime_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RUNTIME"
export R10_RUNTIME_ROOT="$RUNTIME"
export PYTHONPATH=src:experiments/synthetic
```

## Orders

Follow `docs/plans/rps2026/E12_CALIBRATION_SWEEP_RUNBOOK.md` exactly, all nine
steps, in order. It is your entire task. Also honour
`AGENT_PROMPT_REMOTE_EXPERIMENTS.md` (machine rules) and the shared-server
safety patterns of the Round 9 and Round 10 runbooks — reference only, do not
execute their experiments.

Points that have gone wrong in previous rounds and must not go wrong again:

1. **Run the complete repository test suite** (`pytest -q` from the repository
   root, no `NFPA_CONFIRMATORY_FAST`, 90-minute cap), redirecting to a file in
   `$RUNTIME` and testing the captured exit status. **Never pipe pytest into
   `tee`** — the pipeline returns `tee`'s status and hides a failure. Record the
   collected count. A focused suite is not the suite.
2. **Pass `--implementation-freeze-commit "$FREEZE_COMMIT"`.** The runner fails
   closed without it; do not work around that.
3. **Do not pass `--sizes`.** The full frozen set is the sweep. A subset writes
   no `sweep_summary.json`, by design.
4. **Never force-reset `$OUTPUT_BRANCH`.** If it exists from an interrupted run,
   verify its ancestry from `$CONTENT_COMMIT` and resume.
5. **Never delete a shard to make `--resume` proceed.** If resume refuses a
   shard, that refusal is the finding — report it.
6. **Write the report before the evidence commit** with `push_status: pending`,
   then push, then add a synchronization addendum carrying the verified SHA.
   Never amend the evidence commit to backfill a push result.
7. **Record all 15 archive digests before and after**, and assert they match.
8. **Never claim a push that did not occur.**

## The two gates you must confirm fired

Do not report success on the basis that the command exited zero.

* **Anchor** — `n_cal_2048/manifest.json` → `anchor_reproduction.passed` is
  `true` with `comparisons == 20`. This is the only thing tying the sweep to
  archived E10b evidence. If it fails, the sweep code has drifted; stop and
  report, do not adjust the tolerance.
* **Control** — `sweep_manifest.json` → `control_gate.passed` is `true`.
  Iso-MPCA never sees the estimated covariance, so it must be identical at every
  calibration size and equal archived E1. If it moved, estimation is leaking
  into the control arm and every contrast in the sweep is void.

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
and the push succeeded; otherwise
`E12_SWEEP_PARTIAL_WITH_RECORDED_FAILURES`.

## Also report, in prose

* Wall time per condition and in total.
* The per-condition degradation means and contrast signs, copied from
  `sweep_summary.json` — not retyped from memory.
* `trend.channel_shape_error.monotone_decreasing` and the temporal equivalent,
  stated as observations. No direction was predeclared, so a non-monotone trend
  is not an error.
* Anything you had to decide that the runbook did not cover.
