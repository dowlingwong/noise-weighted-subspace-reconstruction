# Conditional experiments — disposition

The amendment makes several experiments *conditional* on which claim the paper retains:
"If retained scope calls for robustness, prepare rank/shrinkage/time-cut sensitivities…",
"No fresh GWOSC campaign is required to deliver C07's current claim." A conditional
experiment must not be silently omitted: the disposition has to be recorded, with the
claim it is conditional on and what actually happened to that claim.

This file is that record. One row per conditional experiment. `Built` means a runnable
package exists and passed local acceptance; it never means executed.

Status vocabulary: `configured` → `locally tested` → `ready to transfer` → `executed` →
`results validated` → `claim supported`. Nothing here is past `ready to transfer`.

---

## CE1 — CRESST rank / shrinkage / time-cut robustness

**Conditional on:** retaining a CRESST claim that generalises beyond the single archived
operating point (one rank, one shrinkage, one time cut).

**What happened to the claim.** It was *narrowed*, not retained. The manuscript now states
an **archive-specific same-rank residual effect**, and Limitations says in terms that
"Rank/shrinkage robustness and time-cut sensitivity are *deferred*, not established."
Register entries C06.02 (`delivered-narrowed`) and C06.04 (`deferred-remote`).

**Disposition: BUILT ANYWAY, and not claimed.** The narrowing removes the *obligation* to
run it, not the value of having it. `cresst_s/robustness_design.py` carries the design
with declared primary settings, paired contrasts and explicit within-run dependence, as
the amendment asks. 395 cells / 1,056 fits enumerated; 350 of those cells are marked
`NOT EXECUTABLE` in the run matrix itself, because the raw CRESST traces are not in this
repository and the package refuses to invent them.

**Why it was built rather than dropped.** Deleting it would have reduced the completeness
backlog by removing an obligation rather than discharging it — the thing the amendment
prohibits. Recording it as deferred keeps the original question visible.

**What it would and would not settle.** It would establish whether the residual effect
survives a change of rank, shrinkage or time cut *within this archive*. It would **not**
give coverage for a new run or a new detector: with two runs no between-run component is
identified, and `cluster_report.py` has a gate that refuses to report one rather than
printing a number two clusters cannot support. Independent detector or run replication is
a data need and is recorded separately (see CE5).

---

## CE2 — A fresh GWOSC campaign

**Conditional on:** strengthening the GWOSC claim beyond the calibration boundary the
archive already shows.

**What happened to the claim.** Retained unchanged and unstrengthened: one 256-second
interval, one event region, two PSD models, reported as a boundary result. The amendment
says in terms that no fresh campaign is required for C07's current claim.

**Disposition: NOT BUILT, deliberately.** No GWOSC successor package exists and none is
needed. C07.01 and C07.02 are `delivered-negative` and `delivered-verified` from archived
evidence.

**What would change this.** Any attempt to claim that the PSD-domain GLS statistic and a
finite FIR one are equivalent, or that a Gaussian PSD model is adequate for nonstationary
real noise, would need new work. The manuscript claims neither.

---

## CE3 — T22 inference calibration

**Conditional on:** using the two-stage bootstrap for generalization rather than as a
conditional diagnostic.

**What happened to the claim.** The verdicts are **withheld** — the frozen dependence
assessment did not stabilise (V(32)/V(16) = 1.745, nine complete blocks) and the rule
declared before development withholds every benefit verdict in that case. The intervals
and p-values are labelled conditional diagnostics.

**Disposition: BUILT.** `t22s` carries the inference-calibration simulator (727 cells,
0 fits — the cells are simulation replicates, which is why the fit count is zero and not
an omission). The amendment requires calibrating a proposed inference under stated
assumptions *before* using it for generalization, which is precisely the order here: the
manuscript does not use it for generalization, and the calibration is unrun.

**Boundary that must not be crossed.** A passing calibration would establish calibration
**under the simulator's assumptions**. It would not establish adequacy on instrument data
that has not been examined. The register says this at C09.05 and the package's
`FINDINGS_T22S.md` repeats it.

---

## CE4 — T22 matched whole-window / narrowband comparison

**Conditional on:** making any cross-stage comparison, which the manuscript currently
refuses.

**What happened to the claim.** No cross-stage difference is quoted anywhere. The single
whole-window reference in the T22 section is labelled "for reference only" and unmatched,
and the figure deliberately draws no whole-window value (register C08.04).

**Disposition: INTERFACE BUILT, NOT RUNNABLE ON REAL DATA HERE.**
`t22s/matched_stage.py` implements the matched design and is exercised against fixtures.
It also produces the **exposure inventory** — the record of which instrument regions the
earlier stages already examined — which is why the archived run trees travel with the
handoff.

**Prerequisite that is not satisfied.** A matched run on real data needs TIDMAD `.h5`
files that this repository does not carry, and needs hold-out regions the exposure
inventory shows are unspent. Running it on already-examined regions would not be
independent evidence, and `t22s/REMOTE_COMMANDS.md` says so and forbids it under this
protocol.

---

## CE5 — Independent CRESST detector / run replication

**Conditional on:** any CRESST claim about a new run or a new detector.

**What happened to the claim.** No such claim is made.

**Disposition: NOT BUILDABLE. This is a data need, not a compute need.** No reanalysis of
the two archived runs can identify a between-run component, and no amount of resampling
manufactures independent units. Recorded in the remaining-dependencies list rather than as
a package, because building a runner for data that does not exist would misrepresent the
obligation as addressable by execution.

---

## CE6 — An in-domain test of the covariance diagnostic with an independently estimated Σ̂

**Conditional on:** presenting the first-order displacement result as a diagnostic a
practitioner can evaluate on their own data.

**What happened to the claim.** Narrowed (register C01.06, C03.02). The manuscript labels
it explanatory under a known or reference covariance and says in the Discussion that it is
"a modelling statement, not yet a diagnostic a practitioner can evaluate on their own
data".

**Disposition: PARTIALLY BUILT.** `t19s` tests the observable-by-observable residual-order
reading, which is the part that can be tested with engineered perturbations. It does
**not** test an independently estimated Σ̂: the perturbations are engineered in the sample
eigenbasis at a fixed training draw, by design, so the covariance term is isolated. The
in-domain test with an estimated Σ̂, and a finite-perturbation theorem informative at large
λ₁/δ, are separate missing results and are listed in the manuscript's "What would settle
these" and in the remaining-dependencies list.

---

## Summary

| id | conditional on | claim outcome | package built | executed |
|---|---|---|---|---|
| CE1 | CRESST robustness beyond one operating point | narrowed; deferral stated | yes (`cresst_s`, 350 cells not executable here) | no |
| CE2 | strengthening GWOSC beyond the boundary | retained unchanged | no, and none needed | — |
| CE3 | using the T22 bootstrap for generalization | verdicts withheld | yes (`t22s`, calibration) | no |
| CE4 | any cross-stage T22 comparison | refused; none quoted | interface only | no |
| CE5 | CRESST new-run / new-detector coverage | not claimed | not buildable (data need) | — |
| CE6 | covariance result as a deployment diagnostic | narrowed to explanatory | partially (`t19s`) | no |

No conditional experiment was dropped to reduce the backlog. Two (CE2, CE5) are not built
for stated reasons; four exist as packages or interfaces and none has been executed.
