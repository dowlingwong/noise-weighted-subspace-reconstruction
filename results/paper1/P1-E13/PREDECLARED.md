# PREDECLARED — P1-E13 (TIDMAD operating-point sweep, zero-predictor baseline, norm-matched control)

Frozen before the first evaluation run of this card.

## Predeclared block (verbatim from the task card)

> Predeclared. Files `abra_validation_0000.h5`, `0005`, `0010` (the three
> already used, no new selection); `window_samples` in {4096, 16384, 65536};
> `n_windows` adjusted so the same sample range is covered (512 -> 128 -> 32; if
> 32 windows give < 10 evaluation windows, report the row as underpowered rather
> than lengthening the range); rank 2; fit fraction 0.7; guard 4 windows;
> bootstrap 300 window-level and block-level (block 16 -> 4 -> 1); seed
> 20260727. Metric: paired absolute amplitude error PCA - EMPCA; plus amplitude
> RMSE of PCA, EMPCA, zero predictor.

> Predictions. T-a: at every (file, window) the three-way ordering raw-MSE ->
> PCA, weighted -> EMPCA, truth -> EMPCA holds (direction check). T-b: there
> exists a window length at which EMPCA amplitude RMSE < zero-predictor RMSE on
> file 0000 (the SNR floor is crossed) - if not, report the floor as "not
> reached at 65 536 samples". T-c: the norm-matched random basis degrades the
> primary metric with interval excluding zero on file 0000 at 4096 (the control
> is informative where the saturated one was not).

## Frozen settings (read from the config, not transcribed)

Everything other than `window_samples`, `n_windows` and `block_size` is taken
from `configs/tidmad/phase2_metric_reversal_real_noise.yaml`: rank 2,
`fit_fraction` 0.7, `guard_windows` 4, `bootstrap_reps` 300, seed 20260727,
`n_calibration_windows` 256, contract `docs/tidmad_data_contract.json`,
`science_glob` `tidmad/raw/abra_science_*.h5`.

| window_samples | n_windows | block_size |
|---|---|---|
| 4096  | 512 | 16 |
| 16384 | 128 | 4  |
| 65536 | 32  | 1  |

A row with fewer than 10 evaluation windows is flagged `underpowered = true`
and reported as such; the sample range is never lengthened to rescue it.

## Definitions frozen here

- **Zero predictor**: amplitude 0 for every window, so its per-window absolute
  error is `|truth|` and its RMSE is `sqrt(mean(truth^2))`. Reported next to
  PCA and EMPCA, with the same paired bootstrap
  (`zero_predictor_paired_difference_zero_minus_empca`).
- **`random_weighted_basis_norm_matched`**: a random `Sigma`-orthonormal basis
  of the same rank; the projected reconstruction is rescaled by
  `sqrt(E_fitted / E_random)`, where each `E` is the mean whitened energy of the
  corresponding *mean-centred* reconstruction on the **fit** windows. The
  existing `random_weighted_basis` control is left unchanged and reported
  alongside it.

## Status of this card on this host

**Not run.** The TIDMAD release is not reachable from the host this session
runs on: `PAPER1_DATA_ROOT` (default `/ceph/dwong/paper1_dataset`) does not
exist and no `tidmad/raw/*.h5` file is present locally. `summary.json` in this
directory carries the machine-readable `status: not_run` record with the paths
checked.

What was delivered instead, so the card is one cluster command away:

- `experiments/paper1/tidmad_window_sweep.py` — the sweep runner, which reads
  every non-swept parameter from the config above and writes
  `window_sweep.csv` with the card's columns;
- `src/noise_geometry/tidmad/analysis.py` — `run_metric_reversal` now returns a
  `zero_predictor` block (RMSEs of all three estimators plus the paired
  contrasts);
- `src/noise_geometry/tidmad/negative_controls.py` — the
  `random_weighted_basis_norm_matched` control, with its scale and the two
  energies recorded;
- `tests/test_tidmad_zero_predictor_and_norm_matched_control.py` — four tests,
  passing, that exercise both additions on synthetic pink-noise windows.

Both additions were smoke-run end to end in the stage's own synthetic mode
(`scripts/tidmad_phases.py --phase 2/6 --mode synthetic`), which is a wiring
check only and is not citable. One observation from that check is worth
carrying into the real run: in synthetic mode the norm-matched control still
saturates at the zero-recovery level, because a random *direction* recovers
nothing however its energy is scaled. T-c may therefore be uninformative for
the same reason the unmatched control was; that would itself be the answer to
report.
