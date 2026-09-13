# T20-S v2 — findings placeholder

**Status: configured → locally tested.** No pilot or confirmation seed has been drawn.
This file contains no outcome. It states what the package will and will not be able to
establish once a remote production run returns, is validated and is integrated.

## What the package measures

On a **new analytic Gaussian family** — z = s + n with s ~ N(0, K(θ)), n ~ N(0, I) in
whitened coordinates, K(θ) = V(θ) diag(μ) V(θ)ᵀ with a fixed spectrum μ and a geometry
knob θ that rotates the top-k eigenspace off the product class (C = 8, T = 32, k = 2 × 3):

1. **Invariants before fitting** (algebraic cells, solves only): eigenvalues of K(θ) equal
   μ to 1e-10 at every θ, trace/rank/SNR/Δ_k fixed, while k − cap(θ) moves.
2. **A valid interval for the approximation cost** a(θ) = R_K(P*_F) − R_K(P*_U):
   the partial-trace lower bound Δ_k (k − cap) and the best-found upper bound from
   (R_pop + 1) population-ALS restarts. θ = 0 is the exact a = 0 control.
3. **Paired learner contrasts** d(N) = R_K(P̂_F) − R_K(P̂_U) on identical datasets, with
   x_U ≥ 0 exact and x_F^ref reference-relative, over the fixed grid
   N ∈ {400, 800, 1000, 1250, 1600, 3200} at θ ∈ {0, 0.12, 0.25, 0.40}, with a
   per-θ simultaneous band (max-|t| bootstrap over seeds; Bonferroni-t fallback) and
   the resulting **sign profile** and adjacent sign brackets.
4. **A pilot-frozen scaling test**: at most one of {none, c/N, c/N^α, c/N + d/N²} per
   target (x_U; d − a_ref) is selected on pilot seeds by a two-fold out-of-sample rule,
   frozen to `FROZEN_SELECTION.json`, and scored on fresh confirmation seeds at every
   grid N.

## What it will be able to establish (if the run returns and validates)

* whether, for this family and these learners, the sign of d(N) over the grid behaves
  as the population interval for a and first-order theory predict (bracket above the
  core grid at θ = 0.12, inside at 0.25, below at 0.40), reported as the observed
  profile — possibly nonmonotone — with simultaneous uncertainty;
* which adjacent grid pairs bracket a sign change per θ (never an interpolated N*);
* whether a frozen scaling model, if any, predicts the fresh-seed means inside the
  simultaneous band at every grid N; a frozen choice of `none` means only the
  bracket/profile is reported, which is a complete and reportable outcome;
* how wide the optimality-gap interval [a_lower, a_best] is at each θ, how the
  reference-relative x_F^ref and the restart diagnostics behave, and how often the
  training-loss-selected restart is also the population-best one.

## What it will not be able to establish

* **Anything about the archived Monte-Carlo pulse-family proxy.** The historical
  verdicts stand: the registered c/N law was refuted on that proxy (T20 stage B);
  N* ≈ 960 is a log-linear interpolation of stage-B means; the sign reversal between
  N = 800 and N = 1600 on fresh seeds (stage C) applies to the archived proxy and the
  implemented learners. This analytic generator does not certify that proxy and cannot
  revise those verdicts.
* **A certified restricted population optimum.** Best-found values are upper bounds;
  restart agreement is not a certificate; x_F^ref is not asserted non-negative.
* **A universal scaling law.** The Gaussian family with a fixed spectrum is exactly the
  regime in which first-order PCA theory gives x_U ≈ c_U / N; a selected c/N here says
  nothing about other families, and a refuted one says nothing beyond this family.
* **Separation of sampling from optimization error where restart spread is of the
  order of the contrast.** Such cells are flagged (`optimization_unresolved`) and any
  (θ, N) with more than half of its seeds flagged is reported as unresolved. If restart
  spread turns out negligible for R = 4 on this family, the honest statement is that
  optimization error was not resolvable at this restart budget, not that it is absent
  in general.
* **An exact crossover location.** Only adjacent-cell brackets and the profile.

## Design decisions recorded for the audit

* θ values were chosen from population quantities (the a interval per θ and the
  first-order c_U = 3010) before any pilot or confirmation seed was drawn; development
  probes (never citable) were used for timing and to confirm c_U.
* For d(N) the scaling candidates are fitted to d − a_best(θ): the asymptote is a
  population constant, never a fitted intercept.
* The pilot reduce uses `--allow-partial` only because confirmation cells do not yet
  exist; the reducer itself refuses to freeze on an incomplete pilot.
* Fixture (development namespace) outcomes exist only to test the machinery.
