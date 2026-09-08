# PREDECLARED — E2* (nonseparability alpha-sweep at the frozen protocol)

Frozen before the first evaluation run of this card.

## Predeclared block (verbatim from the task card)

> Predeclared. alpha in {0, 0.05, 0.1, 0.2, 0.4, 0.8}; seeds 0-9; methods NFPA,
> CW-PCA, Iso-MPCA, IsoPCA; rank 6 = (2,3); `PrimaryConfig` unchanged
> (`n_restarts = 10`); metric `test_normalized_chi2`; intervals: seed bootstrap.

> Pass. (a) At alpha = 0 the NFPA-CW-PCA held-out gap interval includes <= 0
> (restriction inactive when truth is separable). (b) The approximation term is
> non-decreasing in alpha with each successive interval overlapping or above the
> previous. (c) The metric term NFPA-Iso-MPCA is negative with interval
> excluding zero at every alpha. (d) Restart spread of chi2bar <= 10 % of the
> |metric term| at every alpha - if larger, the optimisation term is material and
> must be printed in section 6.4 as such (this is a reportable outcome, not a
> failure).

## Configuration SHA-256 (JSON of the full `PrimaryConfig`, sorted keys)

| alpha | sha256 | short config_hash |
|---|---|---|
| 0.0  | `09f7dbd811814c55636a1f36a6419a401d5ff57033ce13c36b97b61104fbe1fa` | `09f7dbd811814c55` |
| 0.05 | `a51dab48aff4eeb5429ec389a93abb188dd2e0be5b1ba3e29429a43732e6c51a` | `a51dab48aff4eeb5` |
| 0.1  | `98001a62855cb9a54903db9f9e2dcf3c9273ad957d2f7405b98255ccaacec4ef` | `98001a62855cb9a5` |
| 0.2  | `1b312f1a80bd99f0ce456ae4c50575b24481c8970d8bc0c4ec865acb90c5838f` | `1b312f1a80bd99f0` |
| 0.4  | `f9432adc045ef2ec0371afa9f14dd8a49d4530688ae6b5ecddfbbb5791bd42b3` | `f9432adc045ef2ec` |
| 0.8  | `336d55441a694fd4823986b74c355d93196480492f5543293d919606d1ee124e` | `336d55441a694fd4` |

Every other field of `PrimaryConfig` is unchanged: C=8, T=256, k_c=2, k_t=3
(rank 6), N_train=800, N_validation=400, N_test=2000, n_restarts=10,
n_als_iter=150, target_snr=10.0, channel_ar_rho=0.4, noise_kind=brownian,
als_convergence_tol=1e-10.

## Interval rule

Seed-level paired percentile bootstrap, `r9_common.bootstrap_paired`:
10 000 resamples, resampling unit = seed, bootstrap seed 20260808 (the frozen
`PrimaryConfig.bootstrap_seed`; the card's generic rule-3 seed 20260906 is not
used here because the interval routine of this harness is itself frozen).

## Restart-spread definition (the optimisation term)

For each (alpha, seed) the NFPA fit records, over its 10 ALS restarts:
`chi2_restart_min`, `chi2_restart_max` (held-out normalized chi2 of every
restart's factors), `projector_err_c_restart_sd`, `projector_err_t_restart_sd`
(population SD across restarts). The gate (d) quantity is
`mean over seeds of (chi2_restart_max - chi2_restart_min)` divided by
`|mean paired (NFPA - Iso-MPCA) chi2|`, per alpha.

## Outputs

`results/paper1/P1-E2/{alpha_sweep.csv, rows.jsonl, summary.json,
fig_alpha_cost.pdf, manifest.json}`.

Note (ground rule 4): the pre-existing column names of `alpha_sweep.csv` are
kept and the card's names are ADDED as extra columns
(`raw_mse`, `angle_to_cwpca_deg`, `projector_err_c`, `projector_err_t`), never
renamed.
