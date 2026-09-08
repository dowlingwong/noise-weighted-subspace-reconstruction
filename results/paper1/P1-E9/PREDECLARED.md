# PREDECLARED — E9* (physical-variable probes at 10 seeds)

Frozen before the first evaluation run of this card.

## Predeclared block (verbatim from the task card)

> Predeclared. seeds 0-9, alpha = 0, rank 6; methods NFPA, CW-PCA, Iso-MPCA,
> IsoPCA; ridge probes fit on train, scored on test; variables amplitude,
> rise_time, decay_time, secondary.

> Pass. For amplitude and rise_time, the across-seed interval of
> (NFPA - Iso-MPCA) and (CW-PCA - IsoPCA) probe R^2 excludes zero. Decay-time
> R^2 is reported whatever it is (the 4-seed run gave 0.15-0.28: a boundary,
> print it).

## Configuration

- `PrimaryConfig` with `nonseparability_alpha = 0.0`; config SHA-256
  `09f7dbd811814c55636a1f36a6419a401d5ff57033ce13c36b97b61104fbe1fa`
  (short `config_hash` `09f7dbd811814c55`); rank 6 = k_c*k_t = 2*3.
- Probe rule (unchanged from the archived 4-seed run): standardise each code
  column with training means and standard deviations, append an intercept,
  ridge 1e-3, solve with `numpy.linalg.solve`, score held-out R^2 on the test
  split.
- Interval: seed-level paired percentile bootstrap
  (`r9_common.paired_contrast` -> `bootstrap_paired`, 10 000 resamples,
  bootstrap seed 20260808). The gated contrasts are amplitude and rise_time;
  decay_time and secondary are reported, not gated.

## Outputs

`results/paper1/P1-E9/{probes.csv, corr_matrices.npz, summary.json,
manifest.json}`. Columns are the card's: `seed, method, variable, probe_r2,
max_abs_corr, participation` (plus `config_hash`, `k`, `runtime_seconds`).
The archived 4-seed run is preserved under
`results/paper1/P1-E9/attempts/pre_2026-09-06_HEAD_3026d25/`.
