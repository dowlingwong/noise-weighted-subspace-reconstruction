# t19s — T19-S v2: covariance-error observables, theorem-domain check and residual orders on G1

protocol sha256 recorded in reductions; file `t19s/PROTOCOL_T19S_v2.json`

## Question

On the frozen generator G1 (d = 256, rank 2, N = 512): (1) do the four displacement observables of the covariance-error model A(I+E)^{-1} have the predeclared leading residual orders along fixed perturbation directions, for pure-coupling and generic errors, and do the block-diagonal and zero controls leave the subspace exactly unchanged; (2) on paired synthetic fits with engineered errors (sample eigenbasis), independently estimated covariances (full sample covariance and Toeplitz, population-basis error) and training-sample error, does every fit realised inside the theorem domain (eta <= 0.1) satisfy the absolute remainder bound 44 eta^2, how well does the first-order projector norm agree descriptively, and what residual orders are measured along fixed directions?

## Hypotheses

- H1 (deterministic, covariance model): along a fixed direction the residual orders are generic = (graph 2, projector_matrix 2, projector_norm 2, angle 2) and pure_coupling = (3, 2, 3, 3); block-diagonal and zero E leave the subspace exactly unchanged; the 2-D example A = diag(2,1), E = eps[[0,1],[1,0]], k = 1 has graph coordinate t = -2 eps/(1 + sqrt(1 + 8 eps^2)) and ||P_hat - P - delta P||_F / eps^2 -> sqrt(2)
- H2 (theorem-domain check): every confirmation fit realised at eta <= 0.1 with lambda_min(I+E) > 0 satisfies ||P_hat - P - delta P||_op <= 44 eta^2 (a necessary consistency check of Prop 3.4 on G1; passing does not prove the proposition)
- H3 (descriptive): the projector Frobenius-norm displacement agrees with its first-order value sqrt(2)||X1||_F within a factor of two on in-domain fits; the stress range eta in [0.2, 1.3] is reported as outside the theorem
- H4 (stress test): the residual orders of families C (pure coupling) and F (generic) measured on paired fits along fixed sample-basis directions agree with H1 within 0.3; the archived registered T19 slope prediction (one order for every observable) remains NOT MET and is not re-scored here

## Failure interpretation

- **H1_fails**: a deterministic check failure is a code or derivation defect: reduce exits 4; nothing is reported as an outcome until it is fixed in a new protocol version
- **H2_fails**: report the violating fits by (source, family, eps or m, eta, lambda_min, remainder/eta^2); do not adjust eps, m or the classification to recover; a violation inside eta <= 0.1 is evidence against the stated constant on G1 and is reported as such
- **H3_fails**: descriptive; report the ratio distribution by class
- **H4_fails**: report the measured slopes with CIs; the registered T19 prediction stays NOT MET regardless
- **estimated_arm_never_in_domain**: if no m reaches eta <= 0.1 for an estimator, report the realised eta by m and state that the theorem domain is not reachable at the allocated m for that estimator; this is a finding, not a reason to enlarge m after the fact

## Explicitly not claimed

- a finite-perturbation theorem or any extension of the 44 eta^2 bound beyond eta <= 0.1
- that the stress range eta in [0.2, 1.3] validates or extends the proposition
- that passing the theorem-domain check proves Prop 3.4 (it is a necessary consistency check on one generator)
- that a deployment surrogate for E (which needs the true covariance) has been validated: the engineered arms demonstrate a mechanism under a known reference covariance (review finding S4)
- that the archived registered T19 slope prediction is met: it remains NOT MET; this package explains why one order per observable was a category error and does not re-score it
- anything about a real instrument

---

# t20s — T20-S v2: geometry knob with a fixed spectrum, valid optimality-gap intervals, paired learner contrasts on a fixed N grid with simultaneous inference, and pilot-frozen scaling-model selection

protocol sha256 recorded in reductions; file `t20s/PROTOCOL_T20S_v2.json`

## Question

On an analytic Gaussian family whose whitened clean-signal second moment K(theta) has a FIXED spectrum and a geometry knob theta that moves the top-k eigenspace off the product class: how does the paired class contrast d(N) = R_K(P_F_hat) - R_K(P_U_hat) behave over a fixed N grid, how large is the valid optimality-gap interval for the approximation cost a, and does any declared scaling model, selected on pilot seeds only, predict fresh-seed confirmation cells?

## Hypotheses

- H1 (control, exact): at theta = 0 the top-k eigenspace is a product subspace by construction, so the partial-trace lower bound is 0 and the best-found population a is 0 to 1e-8.
- H2 (invariants, exact): for every theta in the grid the eigenvalues of K(theta) equal mu to 1e-10, and trace, rank, SNR and Delta_k are unchanged, while k - cap(theta) and the interval [Delta_k (k - cap), a_best] change with theta.
- H3 (descriptive, scored by simultaneous CI on confirmation seeds): the sign profile of d(N) over the grid within each theta > 0; the design predicts, from population quantities and first-order theory only, a profile with a bracket above the core grid for theta_1, inside the core grid for theta_2 and below it for theta_3. The profile is reported as observed, nonmonotone if so; no crossing is interpolated.
- H4 (scaling, pilot-frozen): at most one of {none, c/N, c/N^alpha, c/N + d/N^2} is selected for x_U(N) and at most one for d(N) - a_ref(theta) on pilot seeds; its frozen per-theta predictions are scored against the confirmation simultaneous band at every grid N. If the frozen choice is 'none', confirmation reports only the bracket/profile.

## Failure interpretation

- **theta_invariant_failure**: the theta is recorded as excluded in the algebraic cell and dropped from selection, scoring and profiles; the remaining thetas stand; the generator is not re-tuned
- **frozen_none**: no scaling model is tested; confirmation reports brackets and profiles only; this is a reportable outcome
- **prediction_outside_band**: the frozen model is not supported at the grid for that theta; a partial pattern (inside at some N) is reported as such, not as partial support
- **unresolved_cells**: where restart spread is of the order of the contrast the package cannot separate sampling from optimization error; such cells are flagged and excluded from any mechanism statement
- **no_decided_signs**: if the simultaneous band contains 0 at every N for a theta, the profile is undetermined and no bracket is claimed
- **failed_or_retried_cells**: charged in the ledger and never refunded; a cell missing at reduce refuses the reduce (exit 3) unless --allow-partial, which labels the output partial

## Explicitly not claimed

- a certified restricted population optimum P*_F (best-found values are upper bounds only)
- an exact crossover location N*
- universality of any scaling law beyond this analytic family and these learners
- anything about the archived Monte-Carlo pulse-family proxy: the historical c/N refutation, the 960 interpolation and the 800/1600 sign reversal stand and are neither confirmed nor revised here
- separation of sampling from optimization error in cells where restart spread is of the order of the contrast
- any scientific result from fixture (development namespace) runs

---

# t21s — T21-S: independent geometry and excitation -- does the post-hoc spike/loadings explanation predict per-direction recovery?

protocol sha256 recorded in reductions; file `t21s/PROTOCOL_T21S_v2.json`

## Question

On an independent synthetic geometry (planted unit directions in whitened R^d with an orthogonal and a correlated Gram, Gaussian spiked signal covariance, identity noise, two aspect ratios gamma = d/N), does the frozen isolated-spike-times-loadings predictor reproduce the rank-k PCA recovery of each planted direction, and does unique-component recovery separate a direction's own excitation from its overlap with other excited directions?

## Hypotheses

- H1 (predictor, primary): for every confirmation configuration and direction classified 'primary' (no population spike carrying >= 0.05 of the direction lies in the BBP band 0.7 <= theta/sqrt(gamma) <= 1.8), the seed-mean ordinary recovery ||P u_j||^2/||u_j||^2 lies within 0.05 (absolute) of the frozen prediction sum_i loading^2(j,i) c^2(theta_i, gamma), in BOTH geometry arms and BOTH aspect ratios.
- H1' (near-edge, secondary): configurations/directions classified 'near_edge' agree within 0.10 (finite-d corrections of the asymptotic overlap formula are largest at the transition; reported, not decisive).
- H2 (overlap confound made explicit): in the correlated arm with the varied direction at theta_j = 0, ordinary recovery of u_j is predicted and observed well above chance while unique-component recovery of u_j^perp is at chance; in the orthogonal arm the two recoveries coincide identically.
- H0 (all-zero control): with K = 0 the mean recovery equals the chance level k/d within z = 5 standard errors of the exact Beta(k/2,(d-k)/2) law (a pipeline check, not a scientific hypothesis).

## Failure interpretation

- **predictor_fails_on_correlated**: if H1 fails in the correlated arm the post-hoc spike/loadings explanation of the archived T21 outcome is NOT supported; the archived verdicts (5/5 predictions not met) are unchanged either way
- **predictor_fails_on_orthogonal_only**: if H1 fails in the orthogonal arm the isolated-spike overlap itself does not describe this learner at these sizes; the loadings step is then untested
- **predictor_passes**: if H1 passes in both arms the explanation is supported on synthetic Gaussian spiked data with identity noise only; nothing follows for the pulse family, for real detectors, or for any universal detection edge or abstention threshold
- **near_edge_disagreement**: disagreement confined to near-edge pairs is reported as the expected finite-d limitation of an asymptotic formula, not as support and not as refutation
- **all_zero_control_fails**: pipeline error: the run is refused at reduce (exit 4); nothing scientific is concluded
- **regimes_not_populated**: design error found at pilot: confirmation is refused; the grid is revised in a new protocol version

## Explicitly not claimed

- any repair of the five unmet registered T21 predictions (they remain NOT met; see FINDINGS_T21S.md)
- any statement about the archived pulse family, its Jacobian directions or its Gram/spectrum
- a universal detection edge in theta/sqrt(gamma) (c^2 depends on theta and gamma separately) or any abstention threshold
- recoverability or non-recoverability of physical latents in any real detector
- validity of the predictor beyond Gaussian spiked data with identity noise, rank-k PCA and k = J
- any conclusion from fixture (development-only) outputs

---

# t22s — T22-S (L4): synthetic calibration of the archived T22 inference pipeline and a matched whole-window/narrowband real-data interface

protocol sha256 recorded in reductions; file `t22s/PROTOCOL_T22S_v2.json`

## Question

Does the archived T22 inference procedure (joint calibration/evaluation circular-block bootstrap per file, maximum-aggregated block-variance dependence gate, two-stage hierarchical file bootstrap of the equal-file mean, Holm across three claims) have nominal coverage, controlled type-I error and demonstrated useful power -- with its withholding rate counted separately -- on synthetic files with declared dependence, cross-unit heterogeneity and transfer-calibration uncertainty; and what does a matched real-data comparison need before it can be attempted?

## Hypotheses

- H1 (coverage): the two-stage 95% interval for the equal-file mean covers the declared estimand at a rate inside the tolerance band on every validation scenario without drift.
- H2 (type-I): P(any null claim 'supported') is at or below 0.07 on every null scenario, and per null claim.
- H3 (withholding): the withholding rate P(dependence inconclusive) is reported on every scenario; it is expected to rise with phi and to approach one under drift.  H3 is descriptive: no tolerance is set on it, but feasibility (below) uses it.
- H4 (power): P('supported' | alternative) is reported for the small, medium and large effects at 4, 8 and 16 files, unconditionally and given not withheld.  No minimum is hypothesised; the feasibility rule uses the medium effect.
- H5 (secondary): the equal-file Student-t interval and the random-effects interval are reported beside the primary for comparison of coverage and length.
- H6 (bias): the sign of the bias of the equal-file point estimate is reported with its Monte-Carlo standard error.

## Failure interpretation

- **coverage_outside_tolerance**: the archived interval is not calibrated for that dependence/heterogeneity/file-count regime; reported as such, with the t-interval beside it; no re-tuning on confirmation seeds
- **type1_above_tolerance**: the declared decision rule over-claims in that regime; the archived verdicts (already withheld) are unaffected, and any future instrument run at that regime is infeasible
- **high_withholding**: the procedure has no demonstrated useful power in that regime even if type-I is controlled; counted separately and fed into the feasibility rule
- **control_failure**: the run is not reducible (exit 4); the simulator or the wrapper is wrong, not the archived pipeline
- **feasibility_failure**: the correct output of part B; it is reported as such
- **adequacy_on_instrument_data**: NOT established by any outcome here: the dependence structure of unexamined instrument data is unknown

## Explicitly not claimed

- any change to the archived T22 verdicts (all withheld), intervals or p-values
- adequacy of the archived inference on unexamined instrument data
- that the synthetic dependence model (AR(1) plus optional random-walk drift) is the instrument's dependence structure
- any instrument result from part B: the matched interface is exercised on synthetic HDF5 fixtures only, no hold-out exists in this container, and the release file list is an unresolved input
- that a matched rerun on already exposed data would be independent evidence

---

# cresst_s — CRESST-S (L5): cluster-aware reporting of the archived CRESST channel analysis, with a prepared robustness design and a calibration of the archive's hierarchical construction

protocol sha256 recorded in reductions; file `cresst_s/PROTOCOL_CRESST_S_v1.json`

## Question

What can seven run/channel clusters from two CRESST runs support, and what would have to be executed and calibrated before any of it could be read as generalization to a new run or a new detector?

## Hypotheses

- H1 (reproduction). Every point estimate and every interval endpoint of the archived channel-robustness analysis can be recomputed from the archived event residuals. Point estimates must agree exactly (1e-12); interval endpoints must agree within the declared Monte-Carlo tolerance. A deviation beyond tolerance is a FAILURE of this package, not a new result.
- H2 (cluster structure). The seven run/channel clusters are unbalanced across two runs (2 and 5 channels). Run-stratified, leave-one-run-out and leave-one-channel-out reporting will show how much of the archived equal-channel mean is carried by which clusters. This is descriptive sensitivity of the archive.
- H3 (non-identifiability). A between-run variance component is not identifiable from two runs. The package must report the within-channel and between-channel components and must refuse to report a between-run component as identified.
- H4 (procedure calibration). The coverage of the archive's own hierarchical construction for a population mean depends on the number of runs. Under a declared between-run variance the construction is expected to undercover, and coverage is expected to rise with the number of runs. This is a property of the procedure, not evidence about CRESST.
- H5 (robustness feasibility). The rank / shrinkage / time-cut robustness grid can be enumerated, seeded, paired and costed now, but cannot be executed without raw CRESST pulse traces, which are absent from this environment.

## Failure interpretation

- **reproduction_deviation**: A deviation beyond tolerance is a failure of THIS PACKAGE (a wiring, environment or version fault), not a new scientific result, and no recomputed number may be reported until it is resolved.
- **control_failure**: A failed control invalidates every cell of the same run root; the reducer refuses.
- **robustness_cells**: Failing with 'input not available' is the CORRECT and expected behaviour in this environment. It is recorded as a charged failed attempt and is not evidence of anything scientific.
- **calibration_undercoverage**: Undercoverage is a reportable property of the procedure. It is not a defect of CRESST and it is not evidence that the archived effects are absent.
- **null_outcome**: If the calibration shows the construction cannot be repaired at two runs, the correct output is that a new-run claim needs new runs, not a re-labelled archive interval.

## Explicitly not claimed

- NOT a replication. Seven channels from two runs are seven correlated clusters, not seven independent detectors.
- NOT coverage for a new run. Nothing in this package establishes that any interval would cover a new-run or new-detector effect; the calibration measures the opposite for the archive's own design point.
- NOT an identified between-run variance. Two runs give one between-run degree of freedom, and the package refuses to report a component from it.
- NOT executed robustness. The rank / shrinkage / time-cut grid is enumerated and costed only; no robustness cell has been run, here or elsewhere.
- NOT evidence about CRESST from the calibration. The calibration simulates a declared generator; it is a property of the procedure.
- NOT new data. No CRESST raw data is present in this environment, no new detector data exists here, and none was sought.
- NOT a change to the archived numbers. Where this package disagrees with the archive beyond tolerance, the package is wrong until shown otherwise.

---

