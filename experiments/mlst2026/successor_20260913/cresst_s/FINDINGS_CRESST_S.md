# FINDINGS — CRESST-S (L5)

**Status: `configured` → `locally tested`.  Not `ready to transfer` until the local
acceptance report says so, and certainly not `executed`, `results validated` or
`claim supported`.**

**This file contains no outcomes.**  It records what the package will and will not be
able to establish.  Every number the package produces locally is either a reproduction
of an already-archived number or a development-fixture value that is never citable.

Design inputs: MS `plan/MLST_COMPLETENESS_AND_LOCAL_BUILD_2026-09-13.md` §L5 and
MS `reviews/2026-09-12-mlst-paper-review.md` finding S5.

---

## The thing that governs everything else

**There are seven run/channel clusters and they come from two runs.**  Run 33 contributes
two channels; run 35 contributes five.  Channels within a run share cryostat, readout and
live-time conditions.  Seven correlated clusters are not seven independent detectors, and
two runs are not a sample of runs.

Three consequences, each of which the package enforces mechanically rather than only
stating:

1. **No new-run or new-detector coverage can be established here.**  The archive's
   hierarchical construction resamples the seven channel occurrences as if they were
   exchangeable draws from a population of channels.  That assumption is not testable
   from two runs.  Part D measures what it costs when it is false — as a property of the
   procedure, on simulated data — and that is the strongest statement available without
   new runs.
2. **A between-run variance component is not identifiable.**  With `G = 2` the between-run
   mean square has one degree of freedom; the moment estimator is a difference of two mean
   squares and can be negative; its exact chi-square 95 % interval spans a factor of about
   `5.1e3`; and any channel-level heterogeneity aligned with the run split is completely
   confounded with it.  The package reports the arithmetic, marks it `identified: false`,
   and its reducer **refuses** (exit 4) any output that claims otherwise from fewer than
   four runs.
3. **Every cluster-aware number is a descriptive sensitivity of the archive.**  The
   run-stratified, leave-one-run-out and leave-one-channel-out reports carry that label in
   the payload itself, and a test asserts the label is present.  None of it is population
   inference and none of it may be re-described as such downstream.

---

## What the package does

### A. Recomputation from the archived residuals (control / algebraic cells, 0 fits)

Recomputes, from `event_residuals_lossless.npz`, the per-channel fractional reduction
`1 - R_EMPCA/R_PCA`, the equal-channel mean, the ratio-of-means estimand, and each of the
ten interval constructions the archive reports (channel-fixed event bootstrap;
hierarchical channel-then-event; block lengths 16/64/256; both ratio-of-means variants),
and compares each against the archived value.

* Point estimates are compared against **two independent archived sources** — the
  10 September 2026 re-run `t13a_results.json` *and* the sensitivity archive's own
  `channel_summary.csv` — at `1e-12` absolute.
* Interval endpoints are compared at a declared Monte-Carlo tolerance derived from the
  replicate count (`4` MC standard errors, doubled for tail shape; `0.302 * sd` when both
  runs use the archived 10 000 replicates).  The reduced output also reports how many
  constructions came back **bit-identical**, which they are expected to be when the
  replicate count, seed and stream order all match the archive.
* **A deviation beyond tolerance is a FAILURE of this package, not a new result.**  The
  protocol says so, the reducer exits 4, and no recomputed number may be quoted until it
  is resolved.

The package fits nothing here: the residuals were already computed by the frozen CRESST
validation, so this is arithmetic and resampling on archived numbers.  It reports 0 fits
and 0 solves and says why rather than inflating either count.

### B. Cluster-aware descriptive reporting (the L5 deliverable)

Run-stratified estimates (run 33 alone, two channels; run 35 alone, five),
leave-one-run-out, leave-one-channel-out, per-channel effects, and a variance
decomposition reporting the **within-channel** (finite-event) and **between-channel**
components with an explicit statement that a **between-run** component is not identifiable
from two runs.

Two structural facts are checked rather than asserted:

* with two runs, leave-one-run-out **is** the run-stratified estimate of the retained run
  (checked to `1e-12`; a property of having two runs, not evidence about either);
* a stratum of two channels supports one cluster-level degree of freedom, so its cluster
  interval is flagged `cluster_interval_is_coarse`.

### C. Prepared, unexecuted robustness design

A declared primary setting (**rank 6, stationary PSD covariance, shrinkage 0.1, full time
cut**) and the sensitivity grid that *would* be run: ranks 2/4/6/8/12 in the pilot arm;
ranks 4/6/8 x shrinkages 0.0/0.05/0.1/0.2/0.4 x time cuts full/first-half/second-half in
the confirmation arm, over all seven channels.  350 cells, 1050 estimator fits, enumerated
with their seeds, their paired contrasts and their dependence structure:

* the contrast group is `(channel, time cut)`; all settings in a group share one seed and
  therefore one event partition, so a within-group difference against the group's
  reference setting is a paired contrast;
* time-cut contrasts are formed at the reference setting against the `full` cut of the
  same channel;
* within-run dependence is recorded on every row (run 33 supplies 2 of 7 clusters, run 35
  supplies 5) and any aggregate across channels inherits it.

**This grid has not been executed, here or anywhere.**  It needs the raw CRESST pulse
traces, which are **not present in this container**.  No new detector data exists here and
none was sought.  Every production robustness cell is marked `NOT EXECUTABLE IN THIS
ENVIRONMENT` in the enumerated matrix and raises
`input not available: raw CRESST pulse data root (<CRESST_DATA_ROOT>) ...`, which is
charged in the ledger as a failed attempt and exits non-zero.  `validate` and `enumerate`
still succeed, because they compute nothing and open no data.

Executing the grid would still not produce a new-run or new-detector result.  It would
produce paired robustness contrasts on the same seven clusters from the same two runs.

### D. Calibration of any proposed inference

Before a hierarchical interval could be used for generalization, its coverage has to be
calibrated under a stated dependence model.  The package simulates channel/run structured
data with a **declared** between-run variance and measures the coverage of the archive's
own hierarchical construction as a function of the number of runs (2, 4, 8) and channels
per run (2, 5), at `sd_run = 0` and `sd_run = 0.3`, reporting every rate with a Wilson
interval and beside an analytic diagnostic (`true variance / bootstrap-implied variance`).

**This is calibration of a procedure, not evidence about CRESST.**  Both the payload label
and the reduced statement say so.  A generator parameter chosen near the archived scale
does not make the result a CRESST measurement.

---

## What this package cannot establish, stated once more, plainly

* **Not a replication.**  Seven channels from two runs are seven correlated clusters.
* **Not coverage for a new run or a new detector.**  Nothing here demonstrates that any
  interval would cover an effect measured on a run or detector not in the archive.
* **Not an identified between-run variance.**  Two runs give one degree of freedom.
* **Not executed robustness.**  The rank/shrinkage/time-cut grid is enumerated, seeded,
  paired and costed only.
* **Not CRESST evidence from the calibration.**  It is a property of the estimator applied
  to a declared generator.
* **Not new data.**  No CRESST raw data is present in this environment.
* **Not a correction to the archive.**  Where this package disagrees with the archived
  numbers beyond tolerance, the package is wrong until shown otherwise.

## What would be needed to say more

* **Independent runs.**  A new-run interval needs runs, not channels.  The calibration in
  part D exists precisely to quantify how many, under a stated dependence model, and to
  make the answer inspectable before anyone claims otherwise.
* **A `<CRESST_DATA_ROOT>` adapter.**  The robustness grid needs raw pulses in the
  `cresst-s-pulse-root/1` interface of `SCHEMA.md`.  The adapter is deliberately not
  written here: there is no raw data in this container to test it against, and an untested
  adapter would be a silent failure mode.
* **A declared dependence model for the real detector.**  Part D calibrates against a
  generator the protocol declares.  Transferring its conclusion to CRESST would need an
  argument that the real between-run structure resembles the declared one — which is a
  modelling claim about the instrument, not a statistical one about the archive.

## Local acceptance evidence (what has been run here)

`validate`, `enumerate`, a 29-cell development fixture run, its reduction, its resource
estimate, and `tests/test_successor_20260913_cresst_s.py`.  Fixture outputs are marked
`development_only` throughout and are **not** results.  The production matrix has been
enumerated and costed; it has **not** been run.
