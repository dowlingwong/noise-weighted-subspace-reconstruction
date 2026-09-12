# T20-S v2 — result schema

Status vocabulary: this package is **configured → locally tested**. Nothing in a run
root produced with `--fixture` is citable (`development_only: true`).

Every cell file `cells/<cell_id>.json` carries the common envelope
(`cell_id, attempt_uuid, role, seed, family, params, fits_used, solves_used, result,
development_only, wall_s, provenance`; see `common/results.py`). The keys required
of `result` for **every** cell (checked by the common schema check, `RESULT_REQUIRED`
in `runner.py`) are

| key | type | meaning |
|---|---|---|
| `kind` | str | `algebraic` or `fit` |
| `theta`, `theta_index` | float, int | geometry knob and its index in the fixed grid |
| `a_lower_bound` | float | Δ_k (k − cap(θ)), the partial-trace lower bound on a (Prop. 3.6) |
| `a_best_found` | float | R_K(P_F^best) − R_K(P*_U): best of (R_pop + 1) population-ALS restarts; an **upper** bound on a |
| `a_interval` | list | `[a_lower_bound, a_best_found]` — a is reported as this interval; never certified |
| `Delta_k` | float | μ_k − μ_{k+1} |
| `eigenvalues_max_abs_dev` | float | max |eig K(θ) − μ|; must be ≤ 1e-10 |
| `R_U_star`, `R_F_best_found` | float | R_K(P*_U), R_K(P_F^best) |
| `checks_passed` | bool | the package's scientific self-checks ran at completion |

All risks are in units of whitened variance (noise covariance is I in whitened
coordinates). `R_K(P) = tr K − tr(P K)` is evaluated exactly from the analytic K(θ),
never from held-out Monte Carlo.

## Algebraic cell (`kind = "algebraic"`, one per θ, seed `null`, family `algebraic`)

Solves only (`n_fits = 0`, `n_solves = 1 + (R_pop + 1) + 1`: one eigendecomposition,
R_pop random + 1 HOSVD population-ALS solves via the archived `als_population`, one bound).
Additional keys:

| key | formula / meaning |
|---|---|
| `trace_K`, `trace_mu`, `numerical_rank`, `d`, `k`, `snr_per_coordinate` | tr K(θ); Σμ; #eig > 1e-12·μ_1; CT; k_c k_t; tr K / d |
| `partial_trace` | `{k, sum_top_kc_M_c, sum_top_kt_M_t, cap, k_minus_cap}` with M_c = tr_t P*_U, M_t = tr_c P*_U (Fortran vec c + C t) |
| `k_minus_cap` | the nonseparability measure k − cap(θ) |
| `pop_als` | `{n_inits, best_init, restart_spread_captured, hosvd_captured, best_captured}` of the population ALS (captured = tr(P_F K)) |
| `is_theta0_control`, `theta0_control_passed` | θ = 0: requires `a_lower_bound == 0` and `a_best_found ≤ 1e-8` |
| `spectrum_fixed` | `eigenvalues_max_abs_dev ≤ 1e-10` |
| `nonseparability_moves_across_grid` | k − cap pairwise distinct (1e-6) over the whole θ grid |
| `k_minus_cap_by_theta`, `a_interval_by_theta` | the same invariants for every θ of the grid (recorded in every algebraic cell) |
| `invariant_problems`, `excluded`, `excluded_thetas_across_grid` | invariant failures; an excluded θ is dropped from inference by the reducer, never re-tuned |

## Fit cell (`kind = "fit"`, family `pilot` / `confirmation`; role = family in production, `development` in a fixture)

`n_fits = 1 + R` (one closed-form top-k PCA of the centred sample second moment,
R ALS restarts via the archived `als_fit`), `n_solves = 0`. Both learners see the
same z_i (paired). Data: z_i = s_i + n_i, s ~ N(0, K(θ)), n ~ N(0, I), seed stream
`[seed, theta_index, N, phase_code, geometry_seed]`.

| key | formula / meaning |
|---|---|
| `phase`, `N`, `N_index`, `band` | pilot/confirmation; sample size; index in the fixed grid; `core` or `anchor` |
| `R`, `C`, `T`, `k_c`, `k_t`, `k` | restart budget and dimensions |
| `R_U_hat` | R_K(P̂_U), P̂_U = top-k eigenprojector of the centred sample second moment |
| `R_F_hat` | R_K(P̂_F), P̂_F = product projector of the training-loss-selected ALS restart |
| `d` | R_F_hat − R_U_hat (paired class contrast; sign profile over N is the reported object) |
| `x_U` | R_U_hat − R_U_star (excess risk of PCA; exact; ≥ 0 because P*_U is the global minimiser) |
| `x_F_ref` | R_F_hat − R_F_best_found (**reference-relative** excess risk of the ALS fit; contains estimation *and* optimization error; may be negative; not asserted non-negative) |
| `a_ref` | = `a_best_found` (the reference used in the identity) |
| `identity_residual` | d − (a_ref + x_F_ref − x_U); must be ≤ 1e-9 in magnitude (exact identity with these definitions) |
| `restart_train_losses` | training MSE per restart (the selection objective), length R |
| `restart_pop_risks` | R_K of each restart's product projector, length R (recorded, never used for selection) |
| `restart_converged` | per-restart convergence flag from `als_fit` |
| `selected_restart`, `als_selected_restart`, `selected_by` | argmin of the training losses; `als_fit`'s own choice; must be `"training_loss"` and equal |
| `population_best_restart`, `selection_matches_population_best` | argmin of `restart_pop_risks`; whether selection coincided with it (diagnostic) |
| `restart_spread_loss`, `restart_spread_risk` | max − min over restarts of the training loss / of R_K |
| `optimization_unresolved` | `restart_spread_risk ≥ |d|`: the cell cannot separate sampling from optimization error |
| `pca_train_residual`, `als_train_loss` | training objectives of the two learners |
| `fits` | `{pca: 1, als_restarts: R}` — the accounting the ledger checks against `n_fits` |

## Self-checks (`ScienceError`, exit 4, at completion and again at reduce)

Algebraic: spectrum fixed; 0 ≤ `a_lower_bound` ≤ `a_best_found` + 1e-9; `a_lower_bound`
= Δ_k (k − cap); θ = 0 control; interval consistency.
Fit: selection by training loss and equal to argmin; R restart records; identity residual;
d, x_U, x_F_ref, a_best consistent with the recorded risks; x_U ≥ −1e-9; valid a interval;
θ = 0 control; selected restart's recorded population risk consistent with R_F_hat;
restart spread and the unresolved flag consistent; spectrum fixed.
Cross-cell (reduce): every fit cell's reference quantities equal the algebraic cell's for
the same θ to 1e-9.

## Reduced output `REDUCED.json`

Common envelope (`reconciliation, ledger, partial, development_only`) plus

* `phase` (`pilot` or `confirmation`), `pilot_complete`, `confirmation_complete`,
  `design`, `grid`, `invariants` (per θ), `nonseparability_moves`, `excluded_thetas`,
  `controls`, `vocabulary`, `historical_note`, `frozen_choice`;
* pilot phase: `pilot.per_theta` (below), `pilot.selection` (per target: selected model,
  two-fold CV errors of all candidates, halves, frozen per-θ parameters and grid
  predictions), `frozen_selection_file`, `frozen_utc`, `partial_note`;
* confirmation phase: `confirmation.per_theta`, `confirmation.frozen_selection`
  (what was frozen), `confirmation.scoring` (per target and θ: predictions, inside/outside
  the simultaneous band per N, studentized deviations, `supported_at_grid`), `partial`.

`per_theta[θ]`: `a_interval`, `k_minus_cap`, `Delta_k`, `excluded`, `n_seeds`,
`d_simultaneous` and `x_U_simultaneous` (`mean, se, method, q_max_t, half_width,
lower, upper, bonferroni{t_crit, half_width, lower, upper}, replicates, valid_replicates,
fallback_reason`), `sign_profile` (`signs` per N in `{+, −, 0}`, `profile`,
`adjacent_brackets` indices, `n_decided`, `monotone_sign_change`),
`adjacent_sign_brackets_N`, `by_N[N]` (`d_mean, d_sim_ci, d_sign, x_U_mean, x_F_ref_mean,
R_U_hat_mean, R_F_hat_mean, restart_spread_risk_median, restart_spread_loss_median,
selection_matches_population_best_frac, optimization_unresolved_frac,
sampling_vs_optimization_unresolved, als_converged_frac`), `max_identity_residual`,
`min_x_U`, `control_theta0`.

## `FROZEN_SELECTION.json` (written once by the pilot reducer; never overwritten)

`schema t20s-frozen-selection/1`, `protocol_sha256`, `matrix_sha256`, `development_only`,
`pilot_cells`, `pilot_seeds`, `pilot_digest` (sha256 over the sorted (cell_id, result
digest) pairs of the pilot cells), `confirmation_cells_present_at_freeze` (must be 0),
`grid`, `candidates`, `algorithm`, `selection{x_U, d_minus_a_ref}` each with `selected`,
`cv_error`, `reason`, `fit_half`, `holdout_half`, `frozen_per_theta{θ: params,
prediction_at_grid, offset_a_ref}`, `usable_theta_indices`; `excluded_thetas`; `frozen_utc`.
For `d_minus_a_ref` the candidates are fitted to d − a_best(θ) and the prediction adds
a_best(θ) back: the intercept is a population constant, never a fitted parameter.

## Scaling candidates and selection

`none` (per-N mean of the fitting seeds; the non-parametric bracket), `c_over_N`,
`c_over_N_alpha` (α ∈ [0, 4], bounded scalar minimisation, c closed form),
`c_over_N_d_over_N2` (linear least squares). Two-fold: pilot seeds sorted, first half
fits (per-N means), second half scores (per-seed MSE), and vice versa; errors summed over
usable θ; the best parametric candidate is selected only if strictly better than `none`.

## Simultaneous band

Per θ, rows = seeds, columns = the N grid: max-|t| bootstrap (2000 replicates, seed
20260913 + θ index; + 1000 for confirmation) of the studentized deviations of the column
means; band m_N ± q · SE_N. Used only with ≥ 8 seeds and ≥ 1000 valid replicates;
otherwise the Bonferroni-t band (α / (2 n_N), n − 1 df) is reported and `method`
says `bonferroni_t_fallback` with a `fallback_reason`. Signs are decided by the band;
an interpolated crossing is never computed.
