# CRESST-S result schema (L5)

Every per-cell output is `cells/<cell_id>.json` under the run root, with the common
envelope (`cell_id`, `attempt_uuid`, `role`, `seed`, `family`, `params`, `fits_used`,
`solves_used`, `development_only`, `wall_s`, `provenance`) written by
`common.results`, plus the package-specific `result` block described here.

`result` always has exactly these four required keys:

| key | type | meaning |
|---|---|---|
| `kind` | str | one of `archive_point`, `archive_interval`, `cluster_report`, `science_control`, `calibration_control`, `calibration`, `robustness` |
| `passed` | bool | conjunction of `checks[*].passed`; a false value makes `reduce` refuse (exit 4) |
| `checks` | dict | named scientific checks, each `{passed: bool, ...}` |
| `summary` | dict | the payload described per family below |

## Units and conventions

* residuals are the archived **weighted** residuals of the CRESST validation, in the
  archive's own units; the package never rescales them;
* a **fraction** is dimensionless: `fraction_c = 1 - mean(R_EMPCA,c) / mean(R_PCA,c)`
  over the evaluation events of run/channel cluster `c`;
* `*_pct` fields are the same quantity times 100, kept because the archive reports
  percent in its tables;
* `theta = (1/C) sum_c fraction_c` (equal-channel mean, `C = 7`);
* `theta_ratio = 1 - mean_c mean(R_EMPCA,c) / mean_c mean(R_PCA,c)` (ratio of means);
* variance components are variances of a *fraction*, so their unit is (dimensionless)^2;
* `replicates` are bootstrap or Monte-Carlo draws.  **They are never fits.**
* times (`chronological_time_*`) are the archive's own chronological coordinate.

## Accounting

| quantity | rule |
|---|---|
| `n_fits` | one estimator invocation on drawn data.  `classic_pca` counts once; each EMPCA random restart counts once.  A robustness cell charges `1 + EMPCA_RESTARTS = 3`; the `empca_uniform_weight_limit` control charges `2 x 3 = 6` (a flat-PSD arm and a shaped-PSD arm).  Every other cell charges **0**. |
| `n_solves` | **0 for every cell.**  No cell performs an eigen or least-squares solve on engineered or analytic matrices.  The archive-recomputation cells reduce and resample residuals that the frozen CRESST validation already computed; that is arithmetic, not a solve, and the package says so rather than inflating the count. |
| `replicates` | `params.replicates` is the per-construction count the cell's code uses; `params.replicates_total` is what the cell actually draws (for example a `per_channel_effects` cell runs 7 channels x 2 block lengths, so `replicates_total = 14 * replicates`).  Totals per role appear in the reduced output's `replicates_by_role`. |

Production totals: **395 cells, 1056 fits, 0 solves**, cap 1400 (headroom 344).

## Roles

| role | used for | why |
|---|---|---|
| `algebraic` | `archive_point` | exact, deterministic arithmetic on the archived residuals; no randomness at all |
| `control` | `archive_interval`, `cluster_report`, `science_control`, `calib_control` | each of these cells is accepted against a *fixed prior value* — an archived endpoint, an exact identity, or a declared band — exactly as a control is.  None of them fits anything. |
| `pilot` | `robustness_pilot`, `calib_pilot` | reported, never scored; they carry no selection authority (the protocol's `inference.pilot_selection` is `NONE`) |
| `confirmation` | `robustness_confirm`, `calib_confirm` | scored against the acceptance rules frozen in the protocol |
| `development` | fixture cells in `robustness_pilot`, `robustness_confirm`, `calib_pilot`, `calib_confirm` | development-only; never citable |

Control and algebraic cells are **seedless** (`seed = null`) and draw from the archive's
own frozen `SeedSequence(20260713)` substreams or from a fixed internal stream declared
in the protocol.  This is required: the shared matrix checker refuses a control or
algebraic cell that uses a development seed.

## `kind = "archive_point"` (family `archive_point`, 1 cell)

```
summary.point_estimates.channels[]      channel_label, run, n_evaluation_events,
                                        chronological_time_first/last,
                                        mean_pca_weighted_residual   = mean over events of R_PCA
                                        mean_empca_weighted_residual = mean over events of R_EMPCA
                                        fraction = 1 - mean_empca/mean_pca ; fraction_pct
summary.point_estimates.theta_mean_of_channel_fractions   (1/C) sum_c fraction_c
summary.point_estimates.theta_ratio_of_channel_means      1 - mean_c(mean_empca)/mean_c(mean_pca)
summary.point_estimates.n_channels / n_runs / runs / n_evaluation_events
summary.comparison.per_channel[label]   abs deviations against t13a_results.json AND against
                                        channel_summary.csv (fraction_vs_channel_summary_csv)
summary.comparison.aggregate[estimand]  recomputed, archived, abs_deviation, within_tolerance
summary.comparison.tolerance_absolute   1e-12
summary.archived_inputs[role]           relative_path, sha256, declared_sha256, matches_protocol
checks: archived_inputs_match_protocol_sha256, point_estimates_reproduce_the_archive,
        seven_clusters_from_two_runs, event_count_matches_analysis_config
```

## `kind = "archive_interval"` (family `archive_interval`, 10 cells)

One cell per archived construction: `headline`, `channel_fixed`,
`ratio_of_means_{hierarchical,channel_fixed}`, `block{16,64,256}_{hierarchical,channel_fixed}`.

```
summary.recomputed          ci95_lo, ci95_hi, bootstrap_mean, bootstrap_sd, replicates,
                            construction, estimand, channel_resampling, declared_hierarchical,
                            event_block_length, rng_stream, rng_seed, point_estimate,
                            channel_layer_diagnostic{mean_distinct_channels_per_replicate,
                                                     expected_under_declared_rule,
                                                     channel_layer_resamples, n_clusters}
summary.archived            the corresponding block of t13a_results.json["intervals"]
summary.comparison.endpoints[key]  recomputed, archived, abs_deviation, tolerance, within_tolerance
summary.comparison.tolerance       Monte-Carlo tolerance (below)
summary.comparison.bit_identical_to_archive
checks: endpoints_reproduce_the_archive, channel_layer_matches_its_label,
        point_estimate_is_the_archived_one
```

**Monte-Carlo tolerance.**  A percentile endpoint from `R` bootstrap draws has
`SE = sqrt(p(1-p)/R) / f(q_p)`.  With the normal reference `f(q_p) = phi(z_p)/sd` this
is `2.672 * sd / sqrt(R)` at `p = 0.025`.  Two independent runs combine in quadrature;
the allowance is `4` such standard errors times a `2.0` shape inflation for the
bootstrap distribution being more skewed than the normal reference.  At `R = 10000` in
both runs the tolerance is `0.302 * sd`; at the fixture's `R = 1200` against the
archive's `10000` it is `0.653 * sd`.  Both constants are frozen in the protocol.

**Channel-layer fingerprint.**  `mean_distinct_channels_per_replicate` reproduces the
archive's documented channel-draw rule on a separate substream: exactly `C` for a
channels-fixed construction, `C(1-(1-1/C)^C) = 4.51` for a hierarchical one at `C = 7`.
A construction that resamples only events while calling itself hierarchical is visible
inside its own cell, independently of any digest.

## `kind = "cluster_report"` (family `cluster_report`, 5 cells)

`summary.report` names which one; `summary.payload` carries it; every payload carries
`label = "descriptive sensitivity of the archive - not population inference; seven
run/channel clusters from two runs cannot establish coverage for a new run or a new
detector"`.

| report | payload |
|---|---|
| `per_channel_effects` | `rows[]` with `fraction`, `event_iid`, `event_block64` (percentile summaries) — the archive's own per-channel paired event bootstrap |
| `leave_one_channel_out` | `rows[]` with `dropped_channel`, `theta`, `shift_from_headline_pct`, percent endpoints, plus `comparison` against the archive |
| `leave_one_run_out` | same with `dropped_run`, `retained_channels`, `retained_runs` |
| `run_stratified` | `strata[]` per run: `n_channels`, `theta`, `channel_fixed` and `hierarchical_within_run` percent endpoints, `cluster_degrees_of_freedom`, `cluster_interval_is_coarse`; plus `between_run_contrast` (a single 1-df difference) |
| `variance_decomposition` | see below |

### `variance_decomposition`

```
total_observed_dispersion_of_channel_fractions   Var_c(fraction_c), ddof = 1
within_channel.per_channel[]                     bootstrap_variance, bootstrap_sd per channel
                                                 (paired event bootstrap, channel held fixed)
within_channel.mean_bootstrap_variance           mean over channels
between_channel.moment_estimate                  max(0, total - mean within), df = C - 1 = 6
between_channel.share_of_total
between_run.degrees_of_freedom                   G - 1 = 1
between_run.mean_square_between_runs / _within_runs / c0
between_run.naive_moment_estimate                (MS_b - MS_w)/c0 -- arithmetic, may be negative
between_run.chi2_95_width_factor_at_this_df      ~5.1e3 at 1 df
between_run.identified                           MUST be false with fewer than 4 runs
between_run.identifiable_from_this_design        false
nested_attempt                                   which components separate and which do not
```

**A between-run variance component is not an estimand of this package.**  With `G = 2`
runs the between-run mean square carries one degree of freedom, the moment estimator is
a difference of two mean squares and can be negative, its exact chi-square 95 % interval
spans a factor of about `5.1e3`, and any channel-level heterogeneity aligned with the run
split is completely confounded with it.  The reducer refuses a decomposition that reports
it as identified from fewer than `min_runs_for_between_run_variance = 4` runs.

## `kind = "science_control"` (family `science_control`, 4 cells)

| control | identity checked |
|---|---|
| `block1_equals_iid` | `_block_indices(..., block=1)` is byte-identical to `_iid_indices(...)` on the same seed, and `block=64` differs |
| `zero_effect_identity` | with `R_EMPCA := R_PCA` every fraction is exactly 0 and the hierarchical interval is degenerate at 0 |
| `event_weight_constant` | the archived `event_weight` is constant (the estimand is a plain mean only then), and a perturbed copy is detected |
| `empca_uniform_weight_limit` | at shrinkage 1.0 the shrunk PSD is flat, weights are uniform, and weighted EMPCA reproduces classic PCA's held-out weighted residual within 1e-3 relative; the shrinkage-0.0 arm must not be degenerate.  **The only control that fits (6 fits).** |

## `kind = "calibration_control"` (family `calib_control`, 1 cell)

Two arms of the calibration generator at `sd_run = 0` and `sd_run = 0.3`.  The null arm's
coverage must lie in `[0.85, 0.99]`; the dependent arm must cover strictly less.

## `kind = "calibration"` (families `calib_pilot`, `calib_confirm`)

```
summary.coverage.design                       the full CalibrationDesign
summary.coverage.coverage_of_population_theta Wilson {rate, lo, hi, half_width, n, k}
summary.coverage.coverage_of_realised_channel_mean
                                              coverage of the realised equal-channel mean of the
                                              simulated true fractions -- separates "misses the
                                              population mean" from "interval too short"
summary.coverage.mean_interval_length
summary.coverage.analytic                     true_variance_of_equal_channel_mean,
                                              expected_bootstrap_implied_variance,
                                              true_over_implied,
                                              large_sample_coverage_normal_reference
summary.coverage.replicates                   monte_carlo, bootstrap_per_monte_carlo_replicate,
                                              bootstrap_total
```

## `kind = "robustness"` (families `robustness_pilot`, `robustness_confirm`)

**Not executable in this environment.**  A production cell raises
`input not available: raw CRESST pulse data root (<CRESST_DATA_ROOT>) ...`, which the CLI
charges as a failed ledger attempt and exits 1.  Development cells run the same code on
the synthetic pulse fixture.

```
summary.scored   rank, shrinkage, time_cut, n_train_events, n_eval_events,
                 n_components_in_representation, weighted_residual_pca,
                 weighted_residual_empca, fraction, empca_restarts, empca_iterations,
                 empca_best_train_chi2, n_fits, run, channel, channel_label,
                 contrast_group, is_primary_setting, synthetic_fixture_data
```

### `<CRESST_DATA_ROOT>` interface (`cresst-s-pulse-root/1`)

```
<root>/index.json
    {"format": "cresst-s-pulse-root/1", "synthetic": bool, "n_samples": int,
     "sample_rate_hz": float,
     "channels": [{"run": "33", "channel": "36.0", "file": "run33/ch36.0.npz",
                   "n_events": int, "n_noise": int}, ...]}
<root>/<file>   NPZ:  pulses (n_events, n_samples) float64, time ordered
                      times  (n_events,)          float64, ascending
                      noise  (n_noise, n_samples) float64, noise-only traces
```

The CRESST release does not ship in this layout.  An adapter is part of the remote work
and is deliberately **not** written here, because there is no raw data in this container
to test it against; `read_index` refuses any root whose `format` is not the declared one.

## Reduced output (`REDUCED.json`)

```
package, protocol_sha256, matrix_sha256, reconciliation, ledger, partial, development_only
families_present            {family: n cells}
replicates_by_role          {role: {cells, replicates, fits}}
scientific_checks
    cluster_layer_adds_variance                 per (estimand, block) pair: sd ratio >= 2.0
    leave_one_run_out_equals_run_stratified     exact identity, tolerance 1e-12
archive_recomputation       point_estimates, comparison, archived_inputs, verdict
archive_intervals           per construction: recomputed, archived, max_abs_deviation,
                            tolerance, bit_identical_to_archive; plus n_bit_identical
cluster_report              the five payloads
controls                    per control: passed + checks
calibration                 coverage_curves keyed "<family>|C<channels>|sd_run<sd>", each with
                            by_n_runs, coverage_increases_with_runs,
                            coverage_gain_fewest_to_most_runs; reference_arm; generator
robustness_design           per family: cells_enumerated, fits_planned, executed_here = 0,
                            contrast_plan (groups, n_paired_contrasts, n_time_cut_contrasts),
                            status = "not executed: requires <CRESST_DATA_ROOT> ..."
statement                   the scope statement (seven clusters, two runs, no new-run coverage)
```

## Exit codes

`0` ok, `1` cell failure (including `input not available`), `2` budget refusal,
`3` reconciliation (missing / extra / duplicate / unattributed / tampered),
`4` schema **or a failed scientific check**, `5` duplicate completion,
`6` protocol or usage error.

## Fit accounting (`fits_used`, `fits_observed`)

`fits_used` is the number of estimator invocations on drawn data the cell actually
performed.  It is **observed**, not restated: `robustness_design.fit_and_score` counts the `classic_pca` call and each EM restart where they happen, so a robustness cell and the estimator control charge what they ran.  Every other family fits nothing and charges 0; its Monte-Carlo draws are reported as `replicates`.  `common.cli` runs every cell
inside a `common.fitcount` counting region and refuses the cell (exit 4, charged) unless
the count declared in the run matrix (`n_fits`), the count the cell reports and the count
the instrumentation saw all agree.  Each cell file therefore also carries

| key | type | meaning |
|---|---|---|
| `fits_observed.total` | int | estimator invocations instrumented while this cell ran; equal to `fits_used` and to the matrix's `n_fits` |
| `fits_observed.by_label` | dict | the same total split by call site (`classic_pca`, `empca_restart`) |

`solves_used` (algebraic/analytic solves that fit no drawn data) is counted by the
package itself where it performs them and is not part of the `fitcount` instrumentation.
