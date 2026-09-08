# PREDECLARED — P1-E12 (tied versus untied linear autoencoders)

Frozen before the first evaluation run of this card.

## Predeclared block (verbatim from the task card)

> Predeclared. Data: S3 configuration (`configs/synthetic/s3_ae_bridge.yaml`,
> colored noise) and the planted generator at alpha = 0 (vectorised, rank 6).
> Arms: (a) tied + Sigma^-1 loss; (b) untied + Sigma^-1 loss; (c) tied + MSE;
> (d) untied + MSE. Seeds 0-9; optimiser L-BFGS to tolerance 1e-12
> (`converged` arm) and a second pass stopped at 200 iterations (`early` arm);
> l2 = 0. Metrics per arm: `angle_to_empca_max_deg`, `obliqueness`,
> `held_out_weighted_residual`, `held_out_raw_mse`, `iterations`.

> Predictions. (a) converged: angle < 0.01 deg, obliqueness < 1e-6.
> (b) converged: angle < 0.01 deg (same subspace at the optimum - the untied
> optimum of the weighted loss is still the Sigma-orthogonal projector) but
> early-stopped: obliqueness > 1e-2 and angle > 1 deg in >= 8/10 seeds.
> (c), (d) converged: angle to EMPCA > 30 deg under the S3 colored covariance
> (they find the PCA subspace); angle < 0.01 deg in the S4 white control (run
> the white control too, seeds 0-4).

## Frozen operational settings

- Arms exactly as above; `l2 = 0`.
- `converged`: L-BFGS-B with `gtol = 1e-12`, `ftol = 1e-16`, iteration cap
  20 000. `early`: the same optimiser, iteration cap 200.
  The cap is declared here because a single-seed pilot (archived under
  `attempts/pilot_maxiter5000/`, not citable) showed the untied weighted arm
  exhausting a 5000-iteration cap; the cap was raised so that "converged"
  means gradient-tolerance-limited wherever the optimiser can reach it. No
  prediction, threshold, seed list or arm was changed.
- Initialisation, all arms: a random Gaussian basis `D0`, `M`-orthonormalised
  under the arm's own training metric, with `E0 = D0^T M` — i.e. every arm
  starts at the *tied* configuration, so any tied/untied divergence is produced
  by the optimisation, not by a different starting point.
- Split: the first 70 % of each dataset's traces fit, the last 30 % held out.
- Scoring metric is always `Sigma^{-1}` (the S3 diagonal inverse-noise metric,
  or the separable Kronecker `Sigma^{-1}` for the planted arm), including for
  the MSE-trained arms; `angle_to_empca_max_deg` is the largest principal angle
  in that metric to the EMPCA (`Sigma^{-1}`-weighted PCA) subspace at matched
  rank.
- `obliqueness` = `||Pi - Pi*_Sigma||_F / ||Pi||_F` with `Pi = D E` and
  `Pi*_Sigma = D (D^T Sigma^{-1} D)^{-1} D^T Sigma^{-1}`.

## Datasets and seeds

| dataset | source | d | rank | noise | seeds |
|---|---|---|---|---|---|
| `S3`       | `configs/synthetic/s3_ae_bridge.yaml` via `make_s3_dataset` | 128 | 3 | diagonal, variance geomspace(0.02, 0.5) | 0-9 |
| `S4_white` | same signal generator, flat variance 0.1 (the control)      | 128 | 3 | white | 0-4 |
| `planted`  | `PrimaryConfig`, alpha = 0, vectorised (`config_hash 09f7dbd811814c55`, sha256 `09f7dbd811814c55636a1f36a6419a401d5ff57033ce13c36b97b61104fbe1fa`) | 2048 | 6 | separable colored | 0-2 |

Declared deviation (compute budget): the planted arm is run at seeds 0-2, not
0-9. It is a d = 2048 supporting arm; the card's primary arms (S3 and the S4
white control) carry the full predeclared seed sets. Declared before any
planted-arm number was produced.

## Required new test

`tests/test_untied_linear_ae.py::test_untied_mse_recovers_pca_subspace_under_identity_metric`
— at convergence with Sigma = I, the untied MSE autoencoder recovers the PCA
subspace to 1e-6 (the card's gate).

## Outputs

`results/paper1/P1-E12/{ae_ablation.csv, cells*.jsonl, summary.json,
fig_ae_ablation.pdf, manifest.json}`.

## Real-data arm

The optional CRESST arm of this card cannot be run: `PAPER1_DATA_ROOT`
(`/ceph/dwong/paper1_dataset`) is not reachable from this host and no CRESST
release is present locally. Reported as not run, with the reason.
