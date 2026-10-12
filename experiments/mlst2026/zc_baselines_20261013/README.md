# ZC kit: the baselines a practitioner would use (WS-C, review M5)

Design: MS `plan/onthefly/DESIGN_ZC_ZD_IMPLEMENTATION_2026-10-12.md` Part C (author defaults E1-E7 adopted;
registered readings R-C1 and R-C2). Machine-readable protocol: `protocol.json` (built only by
`design.build_protocol`). Kit identity: `KIT_SHA256.json` (SHA-256 of every file in this directory except itself).

Nothing in this kit was ever run on the confirmation namespace `zc-confirm-20261013`: the tests use the T25
FIXTURE geometry (C = 3, T = 4) in the namespace `zc-smoke-fixture`, and the timing benchmark uses the
development namespace `zc-dev-20261013` and records no risk.

## Study

T25 BASE generator (C = 8, T = 32, (k_c, k_t) = (2, 3), separable AR(1) noise), theta in {0, 0.12, 0.25, 0.40},
N in {400, 800, 1000, 1250, 1600, 3200}, 24 fresh seeds (SHA-256 of `zc-confirm-20261013/seed/<i>`).
Task = (seed, theta, N, metric arm); 2,880 tasks, one JSON record each.

| Metric arm | W | Methods |
|---|---|---|
| `oracle` | Sigma^{-1} | U, F, CV-select, N\*-rule, SHR |
| `kron256`, `kron1024` | inverse T25 Kronecker estimate from the first m calibration records | U, F, CV-select, N\*-rule, SHR |
| `full256`, `full1024` | inverse T25 regularized full estimate | U, F, CV-select |

Every method is scored by its exact clean risk in the true Sigma^{-1} metric (`t25.science.risks`).

* **U / F**: T25 recipe (`t25.science.unrestricted`; `t25.science.structured` with `starts`, 4 restarts,
  max_iter 6000, tolerance 1e-8), affine with the fitted mean.
* **CV-select**: 5-fold CV (fold assignment from `t25.rng(namespace, seed, theta, N, "cv-folds")`, the same
  folds for every arm and for SHR); choose U or F by the mean held-out observed weighted residual in the
  supplied metric; refit on all N.
* **N\*-rule**: product iff N < N\*_pred(theta) (theta = 0: product). N\*_pred from the T20-S rows of MS
  `evidence/data/y15_20261012/RESULT.json`, vendored as `pinned/y15_RESULT_20261012.json` (SHA-256
  `e33551b9…`, checked by the preflight); recomputation through the pinned constants path (Y19
  `predictions.constants` = TR09 `core.py`) is a kit test (agreement 1.4e-9 relative).
* **SHR**: top-k eigenspace of K_lambda = (1 - lambda) S_w + lambda NKP(S_w), S_w the whitened centred sample
  second moment in the supplied metric, NKP the Van Loan-Pitsianis nearest Kronecker product (`nkp.py`,
  both factors PSD-projected); lambda in {0, 0.1, ..., 1} by the same 5-fold CV; refit on all N. lambda = 0 is
  bitwise U. **lambda = 1 is not NFPA** (the top-k eigenvectors of A_t (x) A_c are products ranked by
  eigenvalue product and need not form a k_c x k_t grid): SHR is a third estimator, not an interpolation
  path between the hard classes.

Registered readings (approximate simultaneous Bonferroni Student-t, alpha = 0.05, df = 23):

* **R-C1** (family 4): oracle metric, cells (0.25, 400), (0.25, 800), (0.25, 1000), (0.40, 400); per-seed
  contrast clean(N\*-rule) - clean(CV-select). MET iff no cell is decided positive.
* **R-C2** (family 72): every (theta, N, metric) cell of the all-methods arms; per-seed contrast
  clean(SHR) - clean(better hard class), better class = smaller 24-seed mean clean risk in the cell.
  MET iff no cell is decided positive.

Reading notes (protocol `reading_notes`): an R-C1 cell where CV-select and the rule chose the same class in every
seed has contrast exactly 0 (zero SD -> undecided, i.e. "not larger"); report it as "CV and the rule chose the
same class in all seeds". The R-C2 reference class is chosen on the same seeds (biased low, conservative against
SHR); with 72 simultaneous cells a MET reading means "no cell showed SHR decidedly worse than the better hard class
(simultaneous Bonferroni-t over 72 cells)", not "SHR is safe". Unadjusted marginal 95 % intervals per cell and the
largest seed-mean contrast are reported descriptively in `summary.json`.

**Metric caveat (CV and SHR under an estimated metric).** For a fixed affine map (A, b) and an independent record
x = s + n, E‖x − A x − b‖²_W − E‖s − A x − b‖²_W = tr(WΣ) − 2 tr(WAΣ), which depends on the fitted map unless
W = Σ⁻¹ (Prop. 2.1 covers W = Σ⁻¹ only, where it equals d − 2k for every rank-k idempotent map). So only on the
oracle arm do held-out residual differences equal clean-risk differences in expectation; on the estimated arms CV
(and the SHR λ selection) ranks by the W-weighted observed (noisy-target) residual, which is neither the W-clean
nor the Σ⁻¹-clean risk, and the regret there combines the data-split cost with this metric mismatch. Suggested
manuscript sentence: "CV compares held-out residuals weighted by the supplied metric; by Proposition 2.1 these
differences equal clean-risk differences in expectation only for W = Σ⁻¹, so under an estimated metric CV ranks the
classes by the W-weighted observed residual, and its regret on those arms combines the data-split cost with this
metric mismatch."

Descriptive: regret (method - per-seed min(U, F), and method - per-cell better class), CV choice
frequencies, lambda_hat distribution, SHR clean risk along the lambda path, CV regret on the full arms.

## Files

| File | Role |
|---|---|
| `nkp.py` | rearrangement and PSD-projected nearest Kronecker product (T25 vec convention c + C t) |
| `methods.py` | `fit_U`, `fit_F`, `folds`, `cv_select`, `nstar_rule`, `fit_shr` (affine maps + diagnostics) |
| `nstar.py` | N\* table from the pinned Y15 RESULT.json; constants-path recomputation (tests) |
| `design.py` | seeds, grid, arms, methods, families, reading rules; `build_protocol`, `smoke_protocol` |
| `runner.py` | fail-closed preflight, resumable per-task JSON, reduction and readings, `results.h5` |
| `benchmark.py` | development timing (namespace `zc-dev-20261013`; no risk recorded) |
| `pinned/y15_RESULT_20261012.json` | verbatim copy of MS `evidence/data/y15_20261012/RESULT.json` |
| `tests/` | NKP, methods, runner smoke + resume + parallel, reading rules (fabricated), seeds, N\*, kit/preflight |

Reused code (imported unchanged; bytes pinned in `protocol.json` -> `dependencies`, T25-closure members
also checked against the accepted closure `01f2f82f…` by the X3 guard): `t25_production_20260930/science.py`
(`make_data`, `moments`, `estimate`, `starts`, `structured`, `unrestricted`, `map_for`, `risks`, `powers`,
`top`, `qr`, `rng`, `digest`), `t25_production_20260930/protocol.py` (`BASE`, `FIXTURE`),
`y19_20261012/runner.py` (`sha_file`, `sha_json`, `atomic_json`, `decide`, `environment`, `KitMismatch`),
`y19_20261012/generator.py` (`seed_int`).

## Environment

From the EXP root of a clean checkout at the frozen commit (no `.pyc` under `experiments/` or `src/`):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.
unset PYTHONPYCACHEPREFIX
```

Python >= 3.10, numpy, scipy, pytest; h5py optional (`results.npz` otherwise). Developed with Python 3.10.12,
numpy 2.2.6, scipy 1.15.3, h5py 3.16.0.

## Commands

```bash
# 1. kit identity and pinned dependencies
python -m experiments.mlst2026.zc_baselines_20261013.runner verify-kit
# 2. tests (about 1 minute; never touch the confirmation namespace)
python -m pytest experiments/mlst2026/zc_baselines_20261013/tests -q -p no:cacheprovider
# 3. the confirmation run (once; resumable with the same command if interrupted)
python -m experiments.mlst2026.zc_baselines_20261013.runner run --out experiments/mlst2026/zc_baselines_20261013_return --workers 16
# 4. (only if step 3 was interrupted after the last record) rebuild the summary
python -m experiments.mlst2026.zc_baselines_20261013.runner summarize --out experiments/mlst2026/zc_baselines_20261013_return
```

Before step 3 on the run host, calibrate the timing there (development namespace, output outside the kit):
`python -m experiments.mlst2026.zc_baselines_20261013.benchmark --out /tmp/zc_benchmark_host.json --N 400 3200`,
and prefer `--workers 32` so that even the all-restarts-at-the-iteration-cap worst case stays well below 24 h.

**Failure policy (fail-closed).** Any exception in a task stops the worker pool; records already written are kept
and repeating the same `run` command resumes. If the same task fails again (a deterministic exception, e.g. a rank
loss in `t25.qr` or a non-finite value refused by `atomic_json`), the run agent **stops and reports**: no code
change, no re-seeding, no dropped or replaced task, no partial summary presented as a reading (`exclusions: none`).
`summarize` without `--partial` re-verifies the kit and refuses unless `OUT/START.json` was written by this kit
identity and protocol; `--partial` output is diagnostic only.

`run` refuses to start unless the kit hashes, pinned dependency bytes, the accepted T25 closure, the thread
variables (= 1), `PYTHONDONTWRITEBYTECODE`, the protocol rebuild, the seed derivation, the N\* source SHA-256 and
the T25 BASE generator all agree. `OUT` must lie outside this directory. An existing record is never
recomputed or overwritten; `START.json` binds `OUT` to the kit identity and `ATTEMPTS.jsonl` logs every start.
Output: `records/theta<th>/N<N>/<metric>/seed<ii>.json`, `summary.json` (registered families, readings,
descriptive cells), `results.h5` (axes theta x N x metric x seed), `RUN_LOG.txt`. `--save-arrays` additionally
writes the fitted bases and mean per task (`arrays/…npz`, about 100 MB in total; off by default; every record
carries `training_sha`, `calibration_sha`, `metric_cov_sha` and `folds_sha` and is regenerable bit-for-bit).

Size and time (development benchmark, single thread, this VM): one task at N = 3200 takes 12.5-14.6 s on the
all-methods arms and 8.3-8.4 s on the full arms (all NFPA restarts stationary, about 300-520 iterations);
N = 400 is about 25-30 % cheaper. Total about 9 CPU-hours, i.e. about 35 min wall with 16 workers or
about 20 min with 32; peak RSS about 145 MB per worker. Worst case (every restart at the 6000-iteration
limit) about 15x, i.e. about 8.5 h with 16 workers.

Preparation-only commands (already run; rerunning changes the kit and requires a new freeze):
`python -m experiments.mlst2026.zc_baselines_20261013.runner write-protocol`,
`python -m experiments.mlst2026.zc_baselines_20261013.benchmark --out /tmp/zc_benchmark.json`,
`python -m experiments.mlst2026.zc_baselines_20261013.runner write-kit-hashes`.
