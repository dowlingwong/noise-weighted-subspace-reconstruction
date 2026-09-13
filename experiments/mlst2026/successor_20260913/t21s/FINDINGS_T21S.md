# FINDINGS — T21-S (L3): independent geometry and excitation

**Status: configured / locally tested.  No production cell has been drawn.  This
file contains no outcome.**  Fixture (development-namespace) numbers exist only in
local test directories and are not citable.

## What stands regardless of this package

* **All five registered T21 predictions remain NOT met** (archive,
  `experiments/paper1/t21_excitation.py`, results of 11 September 2026).  T21-S
  cannot repair, soften or re-score those verdicts: it runs a different geometry,
  a different learner input and a different grid.  Whatever T21-S returns, the
  manuscript sentence about T21 stays "5/5 registered predictions not met".
* The spike/loadings explanation of that outcome is **post hoc** until this
  package (or an equivalent independent test) has executed, been validated and
  been integrated.  Until then the manuscript must state it as post hoc.
* **The geometry confound**: the recovery of a direction u_j by a rank-k projector
  can reflect overlap with *other* excited directions.  In the archived family the
  Jacobian columns had Gram entries as large as -0.652, so an "unexcited" latent
  was not a direction absent from the signal span.  Unique-component recovery
  (u_j projected off the other planted directions) is exactly the observable that
  separates the two; the archive did not measure it.  An unexcited latent and a
  direction absent from the signal span are not interchangeable.
* Archive artefacts are preserved as artefacts, not turned into claims: the
  weak-grid artefact (s in {0.03, 0.1} were near-replicas of s = 0 because induced
  variance is proportional to s^2), the constant-target probe artefact at s = 0,
  and the fact that theta/sqrt(gamma) alone does not fix c^2 (no universal
  detection edge, no abstention threshold).

## What this package can establish (once executed and validated)

* Whether the frozen predictor `sum_i loading^2(j,i) * c^2(theta_i, gamma)`,
  `c^2 = (1 - gamma/theta^2)/(1 + gamma/theta)` for theta > sqrt(gamma) else 0,
  reproduces the seed-mean rank-4 PCA recovery of each planted direction within
  0.05 (absolute) on every primary (configuration, direction) pair, in an
  orthogonal and in a correlated Gram arm, at gamma = 2 and gamma = 0.25, on
  Gaussian spiked data with identity noise.
* Whether, in the correlated arm, an unexcited direction (theta_j = 0) shows
  ordinary recovery well above chance while its unique component sits at chance:
  the overlap confound observed directly, with predicted values recorded before
  any confirmation draw.
* A pure-noise (all-zero) control at chance k/d, an all-strong reference, and the
  identity unique == ordinary in the orthogonal arm, as pipeline checks.

Interpretation is fixed in `PROTOCOL_T21S_v2.json -> failure_interpretation`:

| outcome | reading |
|---|---|
| H1 fails in the correlated arm | the post-hoc spike/loadings explanation is **not supported** |
| H1 fails in the orthogonal arm | the isolated-spike overlap itself does not describe this learner at these sizes; the loadings step is untested |
| H1 passes in both arms | the explanation is **supported on synthetic Gaussian spiked data only** |
| disagreement confined to near-edge pairs | the expected finite-d limitation of an asymptotic formula; neither support nor refutation |
| all-zero control off chance | pipeline error; the reduce is refused; nothing is concluded |
| pilot regimes not populated | design error; confirmation refused; new protocol version |

## What this package cannot establish

* Anything about the archived pulse family, its Jacobian directions, its Gram or
  its spectrum (the geometry here is deliberately different).
* Recoverability of physical latents in any real detector, or any universal
  detection edge / abstention threshold.
* Validity of the predictor outside Gaussian spiked data, identity noise, rank-k
  PCA with k = J, and the two aspect ratios run.
* Any repair of the five unmet registered predictions.

## Known limitations of the design (declared before execution)

* The predictor ignores the chance floor k/d = 0.0156 at d = 256; that is inside
  the primary tolerance and no chance-corrected variant is scored.
* The isolated-spike formula is asymptotic; near the BBP edge finite-d
  corrections are expected, hence the data-free `near_edge` class with tolerance
  0.10, reported separately.
* Configurations share seeds (paired draws); uncertainty is across seeds within a
  configuration only.
* With 20 confirmation seeds a 0.05 tolerance on the seed mean is resolvable but
  not generous; a failure by a small margin is a failure under the rule, not a
  rounding matter.

## Outcome

_(empty: to be written only from a validated production `REDUCED.json`, quoting
its `protocol_sha256`, `matrix_sha256` and the `FROZEN_PREDICTOR.json` sha256)_
