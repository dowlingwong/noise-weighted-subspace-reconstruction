# PREDECLARED — P1-E11 (excited support: does the data or the initialisation fix T_S^perp?)

Frozen before the first evaluation run of this card.

## Predeclared block (verbatim from the task card)

> Predeclared. Two training supports: S_full = {amplitude, rise_time,
> decay_time, secondary} and S_restricted = {amplitude, rise_time} (decay_time
> and secondary held at their means in the training draw - implement by
> overriding `sample_latents` output before `build_coefficients`). Rank k = 6
> for CW-PCA; (2,3) for NFPA. Seeds 0-9 (data); R = 8 initialisations per seed.
> Quantities per (support, seed, method): (i) `angle_to_TS_max`: largest
> principal angle between the learned whitened subspace and T_S (computed with
> `tangent_geometry` at 64 training latents); (ii) `complement_spread`: mean
> pairwise largest principal angle, across the R restarts, of the learned
> subspace's component orthogonal to T_S; (iii) `held_out_chi2` on a test set
> drawn from S_full (so the unexcited coordinates are present at test time).

> Predictions. P-a: `angle_to_TS_max` < 5 deg for both supports and both methods
> (the excited span is always recovered). P-b: under S_restricted,
> `complement_spread` > 30 deg (the complement is initialisation-determined);
> under S_full, `complement_spread` < 5 deg. P-c: under S_restricted the
> held-out chi2bar on S_full test data exceeds the S_full-trained value by an
> amount whose interval excludes zero, and is not reduced by more training data
> (rerun at 4x N_train: the gap must not shrink by more than 10 %).

## Configuration SHA-256 (JSON of the full `PrimaryConfig`, sorted keys)

| arm | N_train | sha256 | short config_hash |
|---|---|---|---|
| primary | 800  | `09f7dbd811814c55636a1f36a6419a401d5ff57033ce13c36b97b61104fbe1fa` | `09f7dbd811814c55` |
| P-c 4x  | 3200 | `f53d406418371ae0caf07e4d7b56a3152206bb474f6d0494b6c1229812d91720` | `f53d406418371ae0` |

`nonseparability_alpha = 0.0`; every other field of `PrimaryConfig` unchanged
(C=8, T=256, k_c=2, k_t=3, N_test=2000, n_als_iter=150,
als_convergence_tol=1e-10, target_snr=10.0, noise_kind=brownian).

## Operational definitions frozen here

- `S_restricted` sets `decay_time` and `secondary` to their distribution means
  (both exactly 0.0: `rng.normal(0.0, 0.15)` and `rng.normal(0.0, 0.5)`) in the
  *training* draw only. The original sampler is still called with the same `n`,
  so the RNG stream, the noise draws, the validation set and the test set are
  identical to the S_full run of the same seed.
- NFPA initialisations: `als_fit(..., n_restarts=1, seed=1000*seed + r)` for
  r = 0..R-1, exactly as the card prescribes.
- CW-PCA is a closed-form eigen-solve with no initialisation. It contributes one
  solution per seed, so its `complement_spread` is identically 0 by construction
  and is recorded, not interpreted; P-b is decided by NFPA (as the card says).
- `complement_spread`: for each initialisation, the learned whitened basis is
  projected onto T_S^perp and orthonormalised; the statistic is the mean over
  all restart pairs of the largest principal angle between those complements.
- P-c "the deployed model" is the initialisation with the lowest training loss.
- Interval: seed-level paired percentile bootstrap
  (`r9_common.bootstrap_paired`, 10 000 resamples, bootstrap seed 20260808).

## Declared deviation (compute budget), frozen before the run

R = 8 initialisations per seed in the primary (N_train = 800) arm, as the card
requires. In the P-c 4x arm (N_train = 3200, ~4x the cost per initialisation)
R = 4. P-c reads only the lowest-training-loss fit, so this affects the
selection depth, not the quantity being compared. Declared here, before any
4x-arm number was produced.

## Note on diagnostic columns

A smoke wiring run (`--smoke`, N_train=120, never citable) was executed before
this freeze to check the plumbing. Following it, purely diagnostic columns were
added: `angle_to_TS_min`, `angle_to_TS_median`, `angle_to_planted6_max`,
`ts_data_var_min`, `ts_data_var_max`, `noise_floor_var` (the Marchenko-Pastur
upper edge (1 + sqrt(d/N))^2 of the isotropic whitened noise spectrum). No
threshold, prediction, seed list, support or rank was changed.

## Outputs

`results/paper1/P1-E11/{excited_support.csv, complement_spread.csv, cells*.jsonl,
summary.json, fig_excited_support.pdf, manifest.json}`.
