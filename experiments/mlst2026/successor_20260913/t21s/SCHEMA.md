# T21-S result schema

Status of this package: **configured** → **locally tested** (fixture only).  Nothing
below is executed evidence.

Per-cell files live at `<RUN_ROOT>/cells/<cell_id>.json` with the common envelope
(`cell_id`, `attempt_uuid`, `role`, `seed`, `family`, `params`, `fits_used`,
`solves_used`, `result`, `development_only`, `wall_s`, `provenance`).  Every cell
result carries `kind` (str) and `scientific_checks_ok` (bool) — `RESULT_REQUIRED` in
`runner.py`, enforced by `common.cli`.

## Cell parameters (`params`)

| key | values | meaning |
|---|---|---|
| `phase` | `pilot`, `confirmation` | design phase; equals the role in production, is carried in params for fixture (development) cells |
| `geometry` | `orthogonal`, `correlated` | Gram arm; `design.geometry.gram[<arm>]` in the protocol, realised by `geometry.directions` |
| `gamma_arm` | `hi` (γ = 2), `lo` (γ = 0.25) | aspect ratio arm; γ = d/N |
| `d`, `n_train`, `k` | ints | ambient dimension, training draws, fitted rank (k = J = 4) |
| `kind` | `grid`, `all_zero`, `all_strong` | grid cell (one direction varied), pure-noise control, all-strong control |
| `varied_direction` | 0..3 or null | index j of the varied direction (grid only) |
| `ratio` | 0, 0.5, 1, 1.5, 2, 4 or null | θ_j / √γ of the varied direction (grid only); the other three sit at `strong_ratio` = 3 |

Configuration key (used for grouping across seeds and in `FROZEN_PREDICTOR.json`):
`<geometry>|<gamma_arm>|N<n_train>|<kind>|j<varied>|r<ratio>` (`-` where not applicable).

## Data cell `result` (`DATA_RESULT_REQUIRED`)

All quantities are dimensionless.  "Exact K" means `K = U diag(θ) Uᵀ` built from the
frozen geometry, never from data.  `P` is the d × k orthonormal PCA basis.

| field | type | formula / definition |
|---|---|---|
| `kind` | str | copy of `params.kind` |
| `d`, `n_train`, `k` | int | as in params |
| `gamma`, `sqrt_gamma` | float | γ = d / n_train and √γ (the BBP scale) |
| `chance_level` | float | k / d = E‖P u‖² for a Haar-random rank-k projector |
| `theta` | list[4] float | excess population eigenvalue assigned to each planted direction: `geometry.theta_vector` |
| `theta_over_sqrt_gamma` | list[4] float | θ_j / √γ |
| `population_excess_eigenvalues` | list[4] float | top-4 eigenvalues λ_i of the exact K, descending (values < 1e-12 clipped to 0) |
| `c2_per_spike` | list[4] float | c²(λ_i, γ) = (1 − γ/λ_i²)/(1 + γ/λ_i) for λ_i > √γ else 0 (DGJ 2018 eq. 1.6) |
| `loadings_sq` | list[4][4] float | loading²(j, i) = (u_j · v_i)² for the population eigenvectors v_i of K |
| `unique_loadings_sq` | list[4] of list[4] or null | the same for the normalised unique component u_j^⊥ / ‖u_j^⊥‖; null when undefined |
| `predicted_recovery` | list[4] float | **frozen predictor** Σ_i loading²(j, i) c²(λ_i, γ) |
| `predicted_unique_recovery` | list[4] float or null | Σ_i (û_j^⊥ · v_i)² c²(λ_i, γ); null when undefined |
| `recovery` | list[4] float | **observed** ‖Pᵀ u_j‖² / ‖u_j‖² in [0, 1] |
| `unique_recovery` | list[4] float or null | ‖Pᵀ u_j^⊥‖² / ‖u_j^⊥‖², u_j^⊥ = u_j − proj_{span{u_i: i≠j}} u_j; null when undefined |
| `unique_defined` | list[4] bool | ‖u_j^⊥‖²/‖u_j‖² ≥ 1e-3 |
| `unique_norm_ratio` | list[4] float | ‖u_j^⊥‖²/‖u_j‖² = 1/(G⁻¹)_jj |
| `abs_error` | list[4] float | |recovery_j − predicted_recovery_j| (per cell; the scored statistic is on the seed mean) |
| `abs_error_unique` | list[4] float or null | |unique_recovery_j − predicted_unique_recovery_j| |
| `sum_recovery` | float | Σ_j recovery_j (≤ k in the orthogonal arm; a check) |
| `fitted_rank` | int | number of PCA components kept (= k) |
| `sample_top_eigenvalues` | list[k] float | top-k eigenvalues of Zcᵀ Zc / N of the centred training matrix (descriptive; noise floor 1) |
| `gram_realised_max_abs_error` | float | max |UᵀU − G| (must be ≤ 1e-10) |
| `edge_class` | list[4] str | `primary` or `near_edge` (data-free; see protocol `design.predictor.tolerance.classification`) |
| `scientific_checks_ok` | bool | the runner's own invariant check passed before the file was written |

Accounting per data cell: `fits_used = 1` (one SVD of drawn data), `solves_used = 1`
(one eigen-decomposition of the exact K).

## Algebraic cell `result` (`ALGEBRAIC_RESULT_REQUIRED`)

| field | type | meaning |
|---|---|---|
| `kind` | str | `algebraic` |
| `check` | str | `geometry_invariants` (per arm; `solves_used = 1`) or `predictor_sanity` (`solves_used = 0`) |
| `ok` | bool | every declared tolerance held |
| `details` | dict | `geometry.check_invariants(...)` or `predictor.sanity(...)` output per aspect ratio |

## `FROZEN_PREDICTOR.json` (written by the pilot reducer into the run root)

`schema` = `mlst-local-package-frozen-predictor/1`; `protocol_sha256`, `matrix_sha256`;
`predictor` {`formula_id`, `formula`, `coefficients_fitted_on_any_data: false`, `source`};
`tolerance` (copied from the protocol); `scoring_rule`; `pilot_regime_check`
(per geometry × γ arm: below / near / above statistics and `populated`);
`confirmation_permitted`; `configurations` {config key → `theta`, `predicted_recovery`,
`predicted_unique_recovery`, `unique_defined`, `edge_class`, `confirmation_cell_ids`,
`confirmation_seeds`}; `frozen_utc`; `development_only`.  It contains **no data
from any cell**: every prediction is a function of the design.

## `REDUCED.json` (package part; the common envelope adds reconciliation and ledger)

| key | content |
|---|---|
| `phase_completeness` | expected/observed/complete per phase (`pilot`, `confirmation`, `algebraic`) — the common `partial` flag does not know about phases; a pilot-only root is `partial: true` with `pilot.complete: true` |
| `controls.all_zero` | per geometry × γ: `n_seeds`, `mean_recovery` (per-seed direction mean), `chance_level`, `tolerance_abs` = z · sd_Beta / √n_seeds (z = 5), `pass` — a failure refuses the reduce (exit 4) |
| `controls.orthogonal_unique_equals_ordinary` | max |unique − ordinary| over orthogonal cells (must be ≤ 1e-9, checked per cell) |
| `controls.correlated_unique_differs` | the same in the correlated arm (descriptive) |
| `controls.all_strong` | seed-mean recovery per direction vs prediction for the all-strong reference cells |
| `algebraic` | check name → ok |
| `configurations` | per configuration: params, seeds, θ/√γ, population spikes /√γ, per direction {prediction, edge class, `recovery` and `unique_recovery` seed statistics: n, mean, sd, se, ci95 (fixed-seed percentile bootstrap, 1000 replicates)} |
| `pilot.regime_check` | the frozen regime rule evaluated on pilot cells; `populated` |
| `frozen_predictor` | path, sha256, `confirmation_permitted`, formula id |
| `confirmation.rows` | one row per (configuration, direction): `predicted`, `observed_mean`, `observed_se`, `observed_ci95`, `abs_error_of_mean`, `tolerance_abs`, `pass`, `ci95_covers_prediction`, and the `unique_*` counterparts |
| `confirmation.overall`, `per_geometry`, `per_arm` | counts of primary / near-edge / unique passes; `primary_all_pass` is H1 for that slice |
| `confirmation.verdict_by_geometry` | `supported on synthetic Gaussian spiked data only` / `not supported` (the protocol's `failure_interpretation` applies) |
| `confirmation.unexcited_direction_overlap_confound` | ratio-0 rows: ordinary vs unique recovery of the unexcited direction, observed and predicted |
| `historical_verdicts_preserved` | fixed sentence: all five registered T21 predictions remain NOT met |

## Exit codes (inherited from `common.cli`)

0 ok · 1 cell failure (charged) · 2 budget refusal · 3 reconciliation (missing / extra /
duplicate / unattributed / digest) · **4 schema or scientific check** (invariant broken,
prediction ≠ recomputation, all-zero control off chance, frozen file missing or
tampered) · 5 duplicate completion · 6 usage / protocol (including a confirmation run
without a permitting `FROZEN_PREDICTOR.json`, which charges nothing).
