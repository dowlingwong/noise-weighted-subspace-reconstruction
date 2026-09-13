# T19-S v2 — result schema

Status: `configured` → `locally tested`.  Nothing in a fixture root is a result.

## Run root

| file | writer | content |
|---|---|---|
| `run_matrix.jsonl`, `run_matrix_summary.json` | `enumerate` / first `run` | one row per cell: `cell_id`, `role`, `seed`, `family`, `params`, `n_fits`, `n_solves`; counts by role and family |
| `protocol_snapshot.json` | same | the protocol as loaded (with `_path`, `_sha256`) |
| `EXECUTION_LEDGER.jsonl` | `common.ledger` | every attempt, charged at start; `running/completed/failed/interrupted` |
| `cells/<cell_id>.json` | `run` | one per completed cell (schema below) |
| `PILOT_SELECTION.json` | pilot-stage `reduce` (or the fixture reduce, flagged `development_only`) | the frozen scoring selection; never overwritten |
| `REDUCED.json` | `reduce` | reduction summary (below) |
| `RESOURCE_ESTIMATE.json` | `estimate` | common estimate for the matrix of the run root it is pointed at |
| `RESOURCE_ESTIMATE_PRODUCTION.json` | `estimate-production` | fixture timings × declared scale × the **production** matrix |

## Accounting

* **fit** = one call of `synthetic.fit_subspace` on drawn data (rank-2 CW-PCA; one
  symmetric inverse square root of `Sigma_hat` plus one thin SVD of the whitened
  centred training block).  Charged against `budget.estimator_fits_cap`.
* **solve** = one perturbed-subspace computation on engineered or analytic matrices
  (`observables.covariance_model_subspace`: two symmetric eigendecompositions, or
  `additive_model_subspace`: one), or the one eigendecomposition of the sample
  operator per paired cell.  Counted and costed, not capped.  Diagnostic algebra
  (norms, first-order formulas, `E` of an estimated covariance, `D` of a
  training block) is not a solve.
* per paired cell: `fits = 1 + 2·n_eps + 2·n_m + (n_N − 1)`,
  `solves = 1 + 2·n_eps + 2·n_m` (production: 32 / 29; fixture: 15 / 13).
* per deterministic cell: `fits = 0`, `solves = n_eps` (8; the 2-D example 5).
* bootstrap replicates (2000 per CI) are reported as `replicates`, never as fits.

## Cell file (`cells/<cell_id>.json`)

Common envelope (from `common.results.CELL_REQUIRED`): `cell_id`, `attempt_uuid`,
`role`, `seed`, `family`, `params`, `fits_used`, `solves_used`, `development_only`,
`wall_s`, `provenance` (code commit, source hashes incl. `t19_coupling.py` and
`g1.py`, numpy version, thread limits, protocol and matrix hashes, fixture
overrides), and `result`.

`result` always has (`runner.RESULT_REQUIRED`): `kind` (`deterministic` |
`paired`), `d`, `k`, `model` (`covariance` = the paper's model; `additive` =
control), `records` (list), `summary` (dict), `checks` (dict of booleans; a
`false` anywhere makes `reduce` exit 4).

### Deterministic cell (`kind = deterministic`)

Fixed operator `A` (`operator`: `g1_d256` = G1 population whitened operator,
`analytic_d32` = fixture operator λ = (20, 6, 1×30) in a random orthogonal basis,
`two_d_example` = diag(2,1)), one unit-operator-norm direction `E0`
(`direction_seed`, `E0_blocks` = operator norms of the 11/21/22 blocks in the
eigenbasis), one ε path (`params.eps_grid`).  `spectrum`: λ₁, λ_k, λ_{k+1},
δ = λ_k − λ_{k+1}, λ₁/δ, retained internal gaps, min/max cross gap (population
coordinates of the fixed operator).

Each entry of `records` (one per ε):

| field | units / formula |
|---|---|
| `eps`, `eps_op` | nominal ε; ‖εE0‖_op (equal, E0 has unit operator norm) |
| `eta`, `eta_basis` | η = ε λ₁/δ of the fixed operator (`population (fixed operator)`) |
| `lambda_min_IplusE` | λ_min(I + εE0) (covariance model; `null` for the additive control) |
| `valid` | I + εE0 positive definite |
| `first_order_identifiable` | no retained/discarded eigenvalue tie (min |λ_i − λ_j| > 1e-12); when false X1 is NaN and is never divided through |
| `eigensolver_residual` | max_i ‖M v_i − μ_i v_i‖₂, M = A(I+εE0)⁻¹ (or A + εG), on the un-orthonormalised eigenvectors |
| `perturbed_separation` | μ_k − μ_{k+1} of the perturbed operator |
| `in_domain` | η ≤ 0.1 |
| `observables.graph` | `measured_fro` = ‖X‖_F, `predicted_fro` = ‖X1‖_F, `residual_fro` = ‖X − X1‖_F, `ratio` = ‖X‖_F/‖X1‖_F or `null` (absolute branch) |
| `observables.projector_matrix` | `residual_op` = ‖dP − δP‖_op (the theorem's quantity), `residual_fro`, measured/predicted operator and Frobenius norms, `ratio_op` |
| `observables.projector_norm` | `measured` = ‖dP‖_F, `predicted` = √2‖X1‖_F, `residual` = |measured − predicted|, `ratio` |
| `observables.angle` | `measured_sin` = singular values of Q_discᵀV (sin of principal angles, descending), `predicted_sin` = singular values of X1, `residual` = ‖measured − predicted‖₂, `ratio` = max sin ratio |
| `observables.identity_abs_error` | |‖dP‖_F² − 2 tr[S(I+S)⁻¹]|, S = XᵀX (exact identity) |
| `observables.absolute_branch` | ‖X1‖_F ≤ 1e-12: ratios are `null` |
| `bound` | `applies` (covariance model only), `constant` 44, `eta`, `value` = 44η², `residual_op`, `violation` = max(0, residual_op − 44η²), `satisfied` = violation ≤ 1e-12 (numerical floor; η = 0 gives bound 0), `remainder_over_eta2`, `in_domain` |
| `two_d` (2-D example only) | `t_measured` = X[0,0], `t_closed_form` = −2ε/(1+√(1+8ε²)), `abs_error_t`, `residual_over_eps2_measured` = ‖P̂ − P − δP‖_F/ε², `residual_over_eps2_closed_form`, `reference` (1.414143 / 1.414196 / 1.414209 at ε = 0.01 / 0.005 / 0.0025) |

`slopes[<observable>]`: paired log-log OLS along the fixed path over the in-domain
points (`window = "eta <= 0.1"`, `eps_used`, `n_points`, `n_excluded_floor`
for residuals ≤ 1e-13; `slope` is `null` below 3 points).  `predeclared_orders`
(null for the exact families), `slope_tolerance` 0.25.

`checks` (all must be true): `all_points_valid`, `projector_identity_lt_1e-10`;
order families: `order_graph`, `order_projector_matrix`, `order_projector_norm`,
`order_angle`, `orders_within_tolerance`, `bound_satisfied_in_domain`
(covariance); 2-D: `two_d_graph_closed_form_lt_1e-12`,
`two_d_projector_reference_lt_1e-5`, `two_d_projector_over_eps2_closed_form_lt_1e-9`;
exact families: `absolute_branch_all_points`, `ratio_recorded_null`,
`max_displacement_lt_1e-12` (zero) / `max_displacement_lt_1e-09` (block-diagonal),
`bound_satisfied_all_points`, plus the numeric maxima.

### Paired cell (`kind = paired`)

One training draw (`seed`; `rng_offsets.train + seed`), one calibration stream
(`rng_offsets.calibration + seed`, streamed in 8192-row chunks, nested prefixes
for every m), one direction stream (`rng_offsets.direction + seed`).
`generator` (`g1_d256` production; `g1_like_d32` fixture) with `generator_check`
(G1 recorded cond / λ₁ / Δλ, `reference_applies` only at d = 256).
`spectrum_sample` (of the sample operator Â = Σ^{-1/2} R̂ Σ^{-1/2}, centred) and
`spectrum_population` (of A_pop), both with δ and λ₁/δ.  `basis_notes` states
the coordinate system of every arm.  `oracle`: graph of the oracle fit in the
sample basis (must be < 1e-8) and in the population basis, its additive
first-order prediction from D, sin θ to the planted subspace, `eta_D_population`.
`selection_sha256`: digest of `PILOT_SELECTION.json` seen by a confirmation cell
(`null` otherwise).

Each entry of `records` (one per charged non-oracle fit):

| field | meaning |
|---|---|
| `arm`, `source`, `family`, `basis`, `fit_index` | `engineered` (C pure coupling, F generic; `basis = sample`), `estimated` (`S_full`, `S_toeplitz`; E defined with the population Σ, analysed in the sample basis), `training_sample` (D; `basis = population`) |
| `eps_op`, `eps_fro` | ‖E‖_op, ‖E‖_F of the covariance error actually used (`E_used`; after a floor the effective E) |
| `eps_nominal`, `eps_realized_minus_nominal`, `direction_shared_along_path` | engineered arms |
| `m`, `m_over_d`, `eps_op_raw`, `E_defined_with` | estimated arms |
| `N`, `eta_D`, `perturbation_model` | D arm (additive A_pop + D) |
| `lambda_min_IplusE` | from the model-identity solve (`null` for D) |
| `eta_sample`, `eta_population`, `eta_used_for_classification`, `eta_used_is` | ε λ̂₁/δ̂ (sample) and ε λ₁/δ (population); which one classifies |
| `domain_class`, `in_domain` | `in_domain` (η ≤ 0.1, λ_min > 0) / `intermediate` / `stress` ([0.2, 1.3]) / `beyond` / `indefinite` |
| `first_order_identifiable` | as above |
| `observables` | as in the deterministic cell, with Q = Q̂ (sample basis) for engineered/estimated arms and Q = Q_pop for D; X1 from the sample spectrum and Ê = Q̂ᵀEQ̂ (covariance formula) or from D_q (additive formula) |
| `bound` | as above; `applies = false` for D (the bound is stated for the covariance model) |
| `eigensolver_residual`, `model_identity_sin_theta`, `perturbed_separation` | from the deterministic solve of Â(I+E)⁻¹; `model_identity_sin_theta` = sin θ_max between the fitted subspace and the solved invariant subspace (resolution floor of `sin_theta_max` ≈ 1.5e-8; the check is < 1e-6; `null` after a floor) |
| `sin_theta_to_oracle`, `sin_theta_to_planted` | paired displacement and recovery |
| `floor_applied`, `floor_value` | eigenvalue floor of the estimated covariance (archive policy, 1e-6 · tr/d) |
| `E_blocks_sample`, `E_blocks_population`, `D_blocks_population` | operator norms of the 11/21/22 blocks |

`summary`: `n_records`, `n_in_domain`, `domain_counts`.  `checks`:
`oracle_graph_in_sample_basis_lt_1e-8`, `engineered_eps_realized_equals_nominal`,
`model_identity_sin_theta_lt_1e-6`, `projector_identity_lt_1e-10`.

## `PILOT_SELECTION.json` (schema `t19s-pilot-selection/1`)

`development_only`, `rule` (copied from the protocol), `source_role`,
`source_cells_sha256` (digest of the (cell_id, attempt_uuid) pairs it was
computed from), `engineered.eps_in_domain` (≤ 3 largest ε with median realised
η_sample ≤ 0.1), `engineered.eps_stress` (smallest and largest ε with median
η_sample in [0.2, 1.3]), `engineered.median_eta_sample_by_eps`,
`estimated.<S_full|S_toeplitz>.{m_in_domain, m_stress, median_eta_sample_by_m,
theorem_domain_reachable}`.  Written once; a re-reduce that implies a different
selection is refused (exit 3).

## `REDUCED.json`

Common envelope (`package`, hashes, `reconciliation`, `ledger`, `partial`,
`development_only`) plus:

* `scientific_checks.passed` (always true when the file exists; failures exit 4);
* `deterministic.order_table.{covariance, additive_control}.<family>` with
  `predeclared`, `operator`, `directions[]` (slopes per observable, points,
  `within_tolerance`, max identity error, max eigensolver residual, the 2-D rows);
  `deterministic.controls.{zero, block_diagonal}` maxima;
* `stage` (`development` | `pilot` | `confirmation` | `deterministic_only`),
  `selection` (file, sha256, `written_now`);
* `pilot`: the selection tables;
* `confirmation` (also for the development stage, flagged):
  * `a_theorem_domain_check.by_source.<C|F|S_full|S_toeplitz>`: selected-arm fits,
    scored in-domain fits, selected-but-out-of-domain counts by class,
    `fraction_bound_satisfied`, `n_violations`, `violations[]` (seed, arm, η,
    λ_min, residual, violation), `max_remainder_over_eta2`, `per_seed_fraction_ci`
    (percentile bootstrap over seeds, 2000 replicates, seed 20260913+1),
    `undefined_reason`; `pooled` incl. `all_in_domain_fits_any_arm`;
  * `b_first_order_agreement_norm.by_family_and_class.<family>/<class>`: n,
    absolute-branch count, `fraction_in_band` for ratio ∈ [0.5, 2], ratio median /
    min / max, graph and angle ratio medians, per-seed CI, `label`
    (`theorem domain` / `outside the theorem`);
  * `c_residual_orders.by_family.<C|F>`: `window`, `eps_used`, per observable
    `predeclared_order`, `mean_slope_ci`, `slope_median`, `order_consistent`
    (|mean − order| ≤ 0.3; an outcome, `is_outcome_not_assertion`);
  * `d_training_sample_arm.by_N`: η_D median, class counts, ratio medians,
    remainder/η_D² median, `bound_asserted = false`;
  * `census_by_arm`: every attempt by arm with class counts, η median and basis,
    floors, `selected_for_scoring`;
  * `stress_label`.
* `status_vocabulary`.

## Exit codes

0 ok · 1 cell failure (charged; also a confirmation cell started before the
selection was frozen) · 2 budget refusal · 3 reconciliation (missing / extra /
duplicate / unattributed / tampered cells, frozen selection missing, differing or
tampered) · 4 schema **or a scientific check failed** · 5 duplicate completion ·
6 protocol / usage error.

## Fit accounting (`fits_used`, `fits_observed`)

`fits_used` is the number of estimator invocations on drawn data the cell actually
performed.  It is **observed**, not restated: `synthetic.fit_subspace` — the package's only estimator invocation on drawn data — is instrumented, so a paired cell's charge is the number of CW-PCA fits it ran.  `common.cli` runs every cell
inside a `common.fitcount` counting region and refuses the cell (exit 4, charged) unless
the count declared in the run matrix (`n_fits`), the count the cell reports and the count
the instrumentation saw all agree.  Each cell file therefore also carries

| key | type | meaning |
|---|---|---|
| `fits_observed.total` | int | estimator invocations instrumented while this cell ran; equal to `fits_used` and to the matrix's `n_fits` |
| `fits_observed.by_label` | dict | the same total split by call site (`cw_pca`) |

`solves_used` (algebraic/analytic solves that fit no drawn data) is counted by the
package itself where it performs them and is not part of the `fitcount` instrumentation.
