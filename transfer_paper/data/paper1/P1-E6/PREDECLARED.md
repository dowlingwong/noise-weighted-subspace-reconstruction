# PREDECLARED — E6* (rank-one corollary at C = 8, population regime)

Frozen before the first evaluation run of this card.

## Predeclared block (verbatim from the task card)

> Predeclared. n = 20 000 events, SNR 10, seeds 0-9, (k_c, k_t) = (1, 1),
> separable template.

> Steps. `uv run python experiments/paper1/rank_one_check.py --seeds 0 1 2 3 4
> 5 6 7 8 9 --n 20000`; if memory fails, chunk the second-moment accumulation
> (do not lower n). Additionally run n in {1000, 3000, 10000, 20000} at 3 seeds
> and record the temporal |cos theta| vs n so the finite-n approach to 1 is a
> curve.

> Pass. At n = 20 000: min over seeds of |cos theta_c| >= 0.999,
> |cos theta_t| >= 0.999, r(code, OF) >= 0.999. If |cos theta_t| plateaus below
> 0.999, report the plateau and the fitted 1 - |cos theta_t| ~ n^{-beta}
> exponent - that is the finite-sample statement Cor. 6.3 needs.

## Frozen settings

- C = 8, T = 256 (read from `PrimaryConfig`; config SHA-256
  `09f7dbd811814c55636a1f36a6419a401d5ff57033ce13c36b97b61104fbe1fa` for the
  alpha = 0 configuration those two fields come from — the rank-one check
  builds its own isotropic-whitened data and uses no other field).
- ALS: `als_fit(Z, 1, 1, n_iter=100, n_restarts=5, seed=seed)` (unchanged).
- Amplitudes `A ~ U(0.5, 1.5) * SNR`, SNR = 10; template
  `t propto exp(-i/0.1T)(1 - exp(-i/0.02T))`, normalised; `v` a normalised
  Gaussian channel vector drawn from the seed's own generator (unchanged).
- n-curve: n in {1000, 3000, 10000, 20000}, seeds 0, 1, 2.
- beta is fitted by least squares on `log(1 - |cos theta|) = a - beta log n`
  over the seed-mean cosine at each n of the curve.
- Chunking (the card's memory contingency) is applied to the code/OF
  projections only; n is never lowered and the ALS input is the full array.

## Outputs

`results/paper1/P1-E6/{rank_one_check.csv, points.jsonl, summary.json,
manifest.json}`. The card's column names
(`amp_corr_code_vs_of`, `amp_corr_code_vs_truth`) are added; the pre-existing
`amplitude_correlation_*` names are kept alongside them (ground rule 4).
