# PREDECLARED — P1-E4b (hyperparameter table)

Frozen before the first run of this card.

## Task card (verbatim)

> Steps. Write `scripts/make_hyperparameter_table.py` that reads, and prints the
> source path for, every value: synthetic S1-S9 (`configs/synthetic/*.yaml`: d,
> N, SNR, noise kind, rank, seeds), planted generator
> (`nfpa_confirmatory/config.py`: C, d, (k_c,k_t), alpha, SNR,
> N_train/val/test, restarts, ALS tol, N_calibration), CRESST
> (`configs/cresst/cresst_validation.yaml`: rank, shrinkage, covariance model,
> split fraction, template-bank grid, baseline fraction), GWOSC
> (`configs/gwosc/*.yaml`: window, PSD estimator, FIR length, radii), TIDMAD
> (`configs/tidmad/phase*_real_noise.yaml`: window, rank, n_windows, split,
> guard, bootstrap, block). Emit `results/paper1/P1-E4b/hyperparameters.tex`
> (booktabs, one row per value, a "source" column) and `.csv`.

> Pass. Every row has a source path that exists; the author spot-checks five rows.

## Frozen rule

The table is generated from the files themselves — never transcribed. Every row
carries the repository-relative path it was read from; a row whose source file
is absent is emitted with `(MISSING)` appended rather than dropped, so a gap is
visible. Keys that are prose or output plumbing (`output`, `citation`,
`source_url`, `notes`, `contract`) are excluded; every other key of every
matched config is emitted, nested keys flattened with dots and homogeneous
lists kept as JSON.

The `PrimaryConfig` block is read from the dataclass, not from a YAML file, and
carries its own `config_hash` plus the derived `d = C*T` and `rank = k_c*k_t`.

## Outputs

`results/paper1/P1-E4b/{hyperparameters.tex, hyperparameters.csv, summary.json,
manifest.json}`. The `.tex` is a `longtable` (booktabs rules) — a `table`
environment would overflow a page at this row count.
