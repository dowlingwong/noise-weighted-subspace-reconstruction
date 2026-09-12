# T19-S v2 findings — placeholder (no outcome)

Status: **locally tested** (fixture and acceptance tests pass on the development
namespace).  Not `ready to transfer` until the local acceptance report signs
off; not `executed`; nothing below is a result.  The fixture outputs
(`d = 32`, three development seeds) are development-only and are not quoted.

Design inputs: `../../successor_20260912/t19s/` (protocol, amendment, findings,
`t19s_observables.py`), the archived `experiments/paper1/t19_coupling.py` and
`g1.py`, MS plan `MLST_COMPLETENESS_AND_LOCAL_BUILD_2026-09-13.md` (L1) and
review findings S1, S4.  `PROTOCOL_T19S_v2.json` lists what it corrects.

## What the package will be able to establish

1. **The observable/order table on G1, for the paper's model.**  Along fixed
   perturbation directions of the G1 population operator (d = 256, rank 2),
   whether the graph, projector-matrix (operator and Frobenius),
   projector-Frobenius-norm and principal-angle residuals of the top-2 invariant
   subspace of `A(I+εE)⁻¹` have the predeclared orders — generic (2, 2, 2, 2),
   pure coupling (3, 2, 3, 3) — with the block-diagonal and zero controls exact,
   the 2-D example of review finding S1 reproduced (graph
   `t = −2ε/(1+√(1+8ε²))`, `‖P̂−P−δP‖_F/ε² → √2`), the exact identity
   `‖P(X)−P(0)‖_F² = 2 tr[S(I+S)⁻¹]` holding at every point, and the additive
   model `A+εG` reported alongside as a labelled control.  These are
   deterministic checks: a failure is a defect of the package (reduce exits 4),
   not a finding.

2. **A theorem-domain consistency check of Prop 3.4 on paired synthetic fits.**
   For every confirmation fit whose realised `η = ‖E‖_op λ̂₁/δ̂` (sample
   spectrum of the operator the fit diagonalises) is ≤ 0.1: whether
   `‖P̂ − P − δP‖_op ≤ 44 η²`, per error source — engineered C/F in the sample
   eigenbasis, and independently estimated covariances (full sample covariance,
   Toeplitz) from m noise-only draws with a population-basis error — with the
   count of selected-arm fits that fell outside the domain and bootstrap CIs
   across seeds.  Passing is a *necessary* consistency check on one generator;
   the bound is loose (44η² = 0.44 at η = 0.1, against remainders that the
   development fixture puts near 10⁻³ η²), so passing carries little information
   and failing would carry a lot.

3. **Which estimator reaches the theorem domain at which m.**  The realised
   `η` of the full sample covariance and of the Toeplitz estimate is reported
   for every m of the predeclared grid (2d … 2048d); the pilot rule freezes which
   m are scored.  If the unstructured estimator never enters η ≤ 0.1 at the
   allocated m while the structured (Toeplitz) estimate does, that is reported
   as a finding about the estimators' error rates on this generator, not
   repaired by enlarging m afterwards.

4. **Descriptive first-order agreement and residual orders on fits.**  The
   factor-of-two band on the projector Frobenius norm, by source and by class
   (in-domain, intermediate, stress, beyond) with the stress range
   `η ∈ [0.2, 1.3]` labelled outside the theorem; and per-seed log-log slopes
   of families C and F along fixed sample-basis directions, compared with the
   predeclared orders within 0.3 as an *outcome*.

5. **The training-sample arm** (E = 0, N ∈ {N/4, N, 4N, 16N}): the additive
   perturbation `D = Â_N − A_pop`, its `η_D`, and the four observables against
   the additive first-order prediction — descriptive; the 44η² bound is not
   asserted for it.

## What the package cannot establish

* A finite-perturbation theorem, or any extension of `44η²` beyond `η ≤ 0.1`.
  Stress-range agreement, however good, does not validate the proposition.
* That the archived registered T19 slope prediction is met.  It remains
  **not met**; this package explains why one order for every observable was a
  category error (the archived residual is the *graph* residual, the theorem's
  is the projector-*matrix* remainder) and does not re-score the registration.
* A deployment surrogate for `E`.  Every arm needs the true covariance to form
  `E = Σ^{-1/2}(Σ̂−Σ)Σ^{-1/2}`; the engineered arms demonstrate a mechanism
  under a known reference covariance (review finding S4), and the estimated
  arms measure how a specific estimator's error maps to `η` on this generator.
* Anything about a real instrument or a non-Gaussian, non-stationary noise
  model; G1 is circulant Gaussian noise with a rank-2 generic signal basis.
* Anything from the fixture: `g1_like_d32` is not G1, and development seeds
  are never citable.

## Interpretation rules fixed before execution

* Sample versus population coordinates are recorded for every η, gap and basis;
  the classification uses the realised sample-spectrum η for engineered and
  estimated arms and `η_D` (population) for the D arm.
* Ratios are undefined (`null`) when the first-order prediction is zero; the
  absolute residual is then compared with the bound directly.
* Confirmation is a fixed allocation (100 seeds, every arm) scored once against
  the selection frozen by the pilot reducer; no arm, seed or threshold is
  changed after outcomes are seen.  Failure regions are reported by
  (source, family, ε or m, η, λ_min, remainder/η²).
