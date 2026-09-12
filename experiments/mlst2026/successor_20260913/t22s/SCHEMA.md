# T22-S result schema

Every cell file `cells/<cell_id>.json` carries the common envelope
(`cell_id`, `attempt_uuid`, `role`, `seed`, `family`, `params`, `fits_used`,
`solves_used`, `result`, `development_only`, `wall_s`, `provenance`; see
`common/results.py`).  The package-level `result_required` is

| key | type | meaning |
|---|---|---|
| `result.kind` | str | `simulation`, `control` or `matched_fixture` |
| `result.passed` | bool | every scientific check of this cell passed (the reducer recomputes; a `false` is refused at reduce with exit 4) |
| `result.checks` | dict | the checks and their inputs (family-specific, below) |
| `result.summary` | dict | family-specific summary (below) |

Units: every contrast is a mean paired squared error difference in readout
physical-amplitude units, `Delta = |a_hat_A - a|^2 - |a_hat_B - a|^2` with
`a = 2 q / sqrt(w)` (`te.squared_errors`, `w = 1024` nominal in the simulator);
positive favours the second method (EMPCA) exactly as in the archive.

## `simulation` cells (families `sim_pilot`, `sim_confirm`)

`result.summary`:

| field | type | formula / meaning |
|---|---|---|
| `scenario_name` | str | catalogue name, e.g. `alt-medium-n8-phi0.6-tau0.01` |
| `n_mc` | int | Monte-Carlo replicates in this cell (`params.mc_replicates`) |
| `n_withheld` | int | replicates whose aggregate dependence verdict was `inconclusive` |
| `n_supported[claim]` | int | replicates whose verdict for the claim was `supported` |
| `n_covers[contrast]` | int | replicates whose two-stage interval covered `true_mu` |
| `true_mu[contrast]` | float | declared estimand: `lam_e 4/w`, `(lam_e - lam_p) 4/w`, `lam_z 4/w` |
| `effect_sizes` | dict | per-window sd of Delta over N (`sqrt(1 + (1-lam)^2 - 2 kappa^2 (1-lam))`), effect per window sd, effect per file z (`x sqrt(n_eval)`) |
| `replicates_used` | dict | `{file, hier, curve}` bootstrap replicate counts actually used |
| `setting` | str | `archived`, `reduced` (pilot) or the frozen label (confirmation) |
| `frozen_sha256` | str | digest of `FROZEN_CALIBRATION.json` used (`""` for pilot cells) |
| `replicates_total` | dict | bootstrap replicates charged: file `R_file x n_files x n_mc`, hier `R_hier x 4 x n_mc`, curve `R_curve x n_dev x 28 x n_mc` |
| `work_proxy_seconds` | float | the cost proxy of `runner.work_proxy_seconds` |
| `mutation` | str or null | TEST ONLY: the injected mutation label, if any |

`result.records[j]` -- one per Monte-Carlo replicate:

| field | meaning |
|---|---|
| `seed` | replicate seed `cell_seed * 10000 + j` |
| `dependence.verdict` | `te.select_block_length` on `te.aggregate_block_curves` (maximum rule) over the development files' 4 contrast curves |
| `dependence.block_length_evaluation/_calibration` | frozen block lengths passed to `te.joint_bootstrap` |
| `dependence.aggregate_curve[b]` | normalised aggregate `V(b)/V(1)` |
| `withheld` | `dependence.verdict == "inconclusive"` |
| `contrasts[key].point` | equal-file mean of the observed per-file `mean_delta_squared` (the estimand's estimator) |
| `contrasts[key].ci95_lo/hi, bootstrap_sd, p_two_sided, excludes_zero` | `t22.hierarchical_file_bootstrap` (two-stage percentile, `(r+1)/(R+1)` p) |
| `contrasts[key].t_lo/t_hi/t_mean` | `t22.t_interval` on the per-file points |
| `contrasts[key].re_estimate/re_lo/re_hi` | `t22.random_effects` (DerSimonian-Laird) |
| `contrasts[key].true_mu, covers, t_covers, bias, length, t_length` | coverage indicators against `true_mu`; `bias = point - true_mu` |
| `contrasts[key].per_file[name]` | per-file `mean`, joint-bootstrap `bootstrap_mean`, `bootstrap_sd` (`te.percentile_interval`) |
| `verdicts[claim]` | `verdict` (`supported`/`ordering-only`/`not-supported`/`inconclusive`), `p_raw`, `p_holm` from `t22.combine_stage` |
| `problems` | `pipeline.check_record` output at run time (must be empty) |

### Scientific invariants (`pipeline.check_record`; reduce exit 4 on violation)

1. dependence `inconclusive` ⇒ every claim `inconclusive` (the archived gate);
2. `supported` ⇒ point > 0, interval excludes zero, Holm p < 0.05; Holm p ≥ raw p;
3. coverage flags agree with the stored interval and the scenario's `true_mu`; `bias = point - true_mu`;
4. two-stage bootstrap variance equals `(1/n) mean_i[sigma_i^2 + (m_i - m_bar)^2]`
   (per-file bootstrap sd and mean) within `6 sqrt(2/R_hier) + 0.05` relative;
   the equal-file point equals the mean of the per-file points.

## `control` cells (family `control`, seedless)

`result.summary.control` names the control; `result.checks` holds its inputs
and `passed` flags:

| control | check | tolerance |
|---|---|---|
| `simulator_estimand` | realised mean of each exact-mu contrast with TRUE labels vs `mu x mean_i N_i` | 4 Monte-Carlo se |
| `ar1_autocorrelation` | lag-one autocorrelation of `complex_process` at phi = 0, 0.6 | 0.03 |
| `iid_bootstrap_se` | joint bootstrap sd at b = 1 vs `sd/sqrt(n)` | 25 % |
| `dependence_gate` | forced `inconclusive` ⇒ all claims withheld; forced `stabilised` on alt-large n = 4 ⇒ denoising `supported` | exact |
| `holm_arithmetic` | `t22.holm({a: .01, b: .04, c: .03}) == {a: .03, b: .06, c: .06}` | 1e-12 |

## `matched_fixture` cell (development only; charges 24 fits)

`result.summary.matched` is `matched_stage.matched_scores` output without the
window index lists; `result.checks`: identical evaluation windows, identical
label digest, the six non-learned estimators bit-identical across
representations, the B = 1 identity control (`tb.assert_single_bin_identity`,
1e-12), exposure guard engaged.  `exposure_status` is one of
`matched-rerun-on-already-exposed-data`, `unexposed-hold-out`,
`synthetic-fixture`; `label` is the sentence "a matched rerun on already
exposed data must be labelled accordingly".

## `REDUCED.json`

`pilot.scenarios["<scenario>|<setting>"]`, `confirmation.scenarios[<scenario>]`
-- each a `summarise_scenario` block:

| field | formula |
|---|---|
| `withholding` | Wilson block `{rate, lo, hi, half_width, n, k}` of `withheld` |
| `dependence_verdicts` | counts of `stabilised` / `stabilised-at-endpoint` / `inconclusive` |
| `contrasts[key].coverage_two_stage` | Wilson of `covers` (all replicates) |
| `contrasts[key].coverage_two_stage_given_not_withheld` | Wilson of `covers` over non-withheld replicates |
| `contrasts[key].coverage_t_interval` | Wilson of `t_covers` |
| `contrasts[key].mean_length_two_stage / mean_length_t_interval` | mean `length`, `t_length` |
| `contrasts[key].bias_mean, bias_mc_se, bias_sign, bias_sign_resolved, rmse_point` | mean bias, its MC se, sign, `|mean| > 2 se`, root mean square of bias |
| `contrasts[key].coverage_within_tolerance` | Wilson interval not entirely outside `[0.925, 0.975]` |
| `claims[c].is_null` | every contrast of the claim has `true_mu == 0` |
| `claims[c].supported / supported_given_not_withheld` | Wilson of `verdict == supported` |
| `claims[c].type1, type1_within_tolerance` | on null claims: `supported`; Wilson interval not entirely above 0.07 |
| `claims[c].power` | on non-null claims: `supported` |
| `any_supported`, `any_null_claim_supported` | Wilson of "at least one (null) claim supported" |
| `complete_null` | every claim null |

`frozen_calibration`: the content of `FROZEN_CALIBRATION.json` (schema
`mlst-t22s-frozen-calibration/1`: `setting`, `replicates`, block grids,
`tolerances`, `decision_rule`, `pilot_comparison`, `frozen_from_pilot_cells`,
`development_only`).  `confirmation.feasibility_table[n_files]`: per file
count, `type1_within_tolerance`, `coverage_within_tolerance`,
`withholding_rate_max` (non-drift scenarios), `power_medium_effect_min`,
`feasible_under_rules`.  `confirmation.tolerance_report`: the scenarios that
failed a tolerance.  `statement`: the scope statement (synthetic calibration
cannot establish adequacy on unexamined instrument data).

Exit codes: 0 ok; 1 cell failure (including a confirmation cell refusing to run
without the freeze); 2 budget; 3 reconciliation (also a frozen file that differs
from the deterministic pilot freeze, or confirmation cells scored under another
freeze); 4 schema or scientific-check failure; 6 usage/protocol (also the
command-line refusal to start confirmation-type cells without a freeze).
