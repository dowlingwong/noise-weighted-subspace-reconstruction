# E12 run 1 — failed at the first condition gate, after 6 h 42 m

**Date:** 2026-08-24 · **Machine:** `bms2` · **Pinned tree:** `75cbc403…`
**Outcome:** `E12_SWEEP_PARTIAL_WITH_RECORDED_FAILURES`. No scientific evidence
was produced, nothing was committed, and no branch was created.

## What happened

`n_cal_256` completed all ten seeds at 03:27:13, and the condition gate then
rejected the runner's own output:

```
IntegrityError: E12_calibration_size_sweep[n_cal_256]: seed 0 analysis_label mismatch
```

Deterministic: a resume reproduced it in 35 s after reusing all ten validated
shards. It would have hit every condition, so no calibration size could ever
have validated.

## Root cause — mine, not the machine's

`run_calibration_sweep.py` imports `run_seed` from the E10b runner. That
function stamped E10b's module constant into every row:

* rows carried `analysis_label = single_condition_covariance_estimation_sensitivity`
* the gate required `post_hoc_exploratory_calibration_size_sweep`

The sweep defined the correct label and used it in the manifest, but never
passed it to the row builder. The manifest said one thing and the rows said
another; the gate believed the rows.

Reproducible in three seconds at smoke dimensions:

```python
rows, _ = run_seed(910000, replace(smoke_config(), N_calibration=64), None)
rows[0]["analysis_label"] == sw.ANALYSIS_LABEL   # False
```

That diagnostic cost nothing and was never run before dispatch.

## Why 26 tests and a full suite passed anyway

Two independent blind spots, and they lined up.

1. **The gate is behind `if not smoke:`.** Both smoke runs skipped condition
   validation entirely. The dispatch prompt warned that smoke bypasses the
   *clean-tree* gate; the same guard also skips the *condition* gate, which
   nobody had noticed.
2. **Every test performed the substitution the runner omitted.** The fixtures
   built rows by hand and set `analysis_label = sw.ANALYSIS_LABEL` before
   validating. One test even asserted that E10b-labelled rows *must* raise this
   exact error. The suite encoded the correct contract and the pipeline violated
   it, but nothing ran the two together — so the tests proved the gate worked
   while never asking whether the runner satisfied it.

A suite can be adversarial about its subject and still never connect the
producer to the consumer. That is the lesson worth keeping.

## Fix

* `run_seed` takes `analysis_label` as a keyword argument, defaulting to E10b's
  constant, so E10b behaviour is byte-identical and omitting the argument stays
  a visible mistake rather than a silent one.
* `run_condition` passes the sweep's label.
* Two new tests, both mutation-checked — reverting the one-line fix makes both
  fail, restoring it makes both pass:
  * `test_the_runner_itself_writes_rows_the_gate_accepts` invokes the runner as
    the runbook does, then validates the rows it wrote. It deliberately does
    **not** pass the label itself; that was the flaw in the original suite.
  * `test_the_sweep_runner_passes_its_own_label` guards the call site directly.

## Cost, and what it changes

One condition took 6 h 42 m rather than the estimated ~1 h — CPU
oversubscription, 40 BLAS threads on 20 physical cores against a competing
workload. Five conditions is therefore ~30 wall-hours here, not ~5.

At that price, four days from the submission deadline, for post-hoc exploratory
evidence that lives in the appendix and explicitly cannot remove the §5
limitation, E12 is **not worth re-running before submission**. The fix is kept
so the sweep is available for the full-length paper.

## What was not damaged

15/15 archive digests identical before and after; the three pinned digests
byte-identical; worktree clean; `HEAD` still `75cbc403…`, tree still
`a4e44220…`. No shard deleted, no tolerance touched, no `--sizes` passed, the
freeze commit always supplied. The remote agent stopped and reported rather than
patching a pinned tree, which is exactly right.

One partial observation, stated carefully: all ten shards were written, and a
shard is only written after that seed's isotropic control reproduces archived
E1. So per-seed control reproduction passed at `n_cal = 256`. The cross-condition
control gate never ran, so covariance-independence is **unestablished, not
disproven**.
