# FINDINGS — T22-S (L4)

**Status: configured → locally tested (fixtures only).  This file is the
package's findings placeholder required by the round's README.  No outcome is
reported here and none may be quoted from a fixture run.**

## What the package will be able to establish, once executed remotely and validated

1. For the archived T22 inference procedure — run through the archived code
   path, not a re-implementation — on synthetic files with declared within-file
   dependence (AR(1) φ ∈ {0, 0.3, 0.6, 0.85}, optional random-walk drift),
   cross-unit heterogeneity (τ ∈ {0, 0.01, 0.03}, Gaussian or one dominant
   file), heavy tails (t₃) and transfer-calibration uncertainty from 64
   calibration windows:
   * the coverage of the 95 % two-stage equal-file interval for the declared
     estimand μ, with Monte-Carlo (Wilson) uncertainty, unconditionally and
     given not withheld, against the fixed band [0.925, 0.975];
   * the type-I error of each null claim and of "any null claim supported",
     against the fixed bound 0.07;
   * the power at three effect sizes and 4, 8, 16 files, unconditionally and
     given not withheld;
   * **the withholding rate, counted separately** — a procedure that mostly
     withholds has no demonstrated useful power even when its type-I error is
     controlled, and the feasibility rule uses that rate;
   * the coverage and length of the secondary equal-file t-interval and the
     random-effects interval beside the primary;
   * the sign of the bias of the equal-file point estimate with its MC error.
2. A per-file-count feasibility table under rules fixed in the protocol
   (minimum 4 independent files, withholding ≤ 0.5 on non-drift scenarios,
   power ≥ 0.5 at the medium effect, tolerances met), consumed by
   `matched_stage.feasibility`.
3. The pilot's frozen bootstrap replicate setting (`archived` = 2000/10000/400
   or `reduced` = 1000/4000/200), chosen by a rule fixed before the
   confirmation scenarios; the block grids are the archived ones, unchanged.

## What it will not be able to establish

* **Adequacy on unexamined instrument data.**  The synthetic dependence model
  is declared, not measured; the instrument's dependence structure at the
  injection bin is unknown (the archived aggregate curve was still rising at
  b = 32, V(32)/V(16) = 1.745).  A nominal coverage on the synthetic grid is a
  statement about the procedure under those models, not about the data.
* Any change to the archived T22 verdicts: all three remain withheld because
  the frozen dependence assessment did not stabilise; the archived intervals
  and p-values remain conditional diagnostics; the whole-window versus
  narrowband comparison remains unmatched; 416 executions / 240 distinct fits
  against a cap of 250 stand as disclosed.
* Any instrument result from part B.  The matched interface (identical
  windows, labels, allocation and scorer for whole-window and band estimators;
  band width as an input selected on development data only) is exercised on a
  synthetic HDF5 fixture with the TIDMAD layout, read through the archived
  opener under an exposure allowance.  No hold-out data exists in this
  environment; the release file list is not in the archive and is an unresolved
  input; a region never read in an already examined file is not an independent
  unit; a matched rerun on already exposed data must be labelled accordingly.

## Known design limitations, recorded now

* The dependence gate in the simulation aggregates 4 development files × 4
  contrasts = 16 curves; the archive aggregated 220 (4 stretches × 5 bands × 11
  contrasts).  Because the declared aggregation takes the maximum, more curves
  make the gate stricter; the synthetic withholding rate is therefore a lower
  bound for the archived configuration at the same dependence.
* Only the injection-bin slice is simulated; the learned estimates are
  generated as correlated residuals (κ = 0.99), not fitted.  This is what
  makes the estimand exact and the fits zero; it does not test the
  band-selection or admissibility stages.
* The complete null puts the readout line SNR at 1 (the only way every
  contrast is null at once); the partial null (recovery real, denoising and
  metric null) is the realistic null and is in the catalogue at every file
  count.

## Not established, registered direction not met, verdict withheld

These are different facts and the reducer's vocabulary keeps them apart: a
claim can be `not-supported` (point estimate against EMPCA), `ordering-only`
(favourable point estimate, rule not met), `supported`, or `inconclusive`
(withheld by the dependence gate whatever the estimate).  The calibration
reports all four rates; only the last is "withholding".
