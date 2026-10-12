# Y19 kit: prospective first-order crossing test on a second generator (G2)

Registered protocol (human-readable): MS `plan/Y19_PROTOCOL_2026-10-12.md`.
Machine-readable protocol: `protocol.json`. Predictions (population quantities only): `predictions.json`.
Kit identity: `KIT_SHA256.json` (SHA-256 of every file in this directory except itself).

Nothing in this kit was ever run on sampled G2 data before the freeze: no G2 record was drawn,
no learner was fitted to G2, and no G2 risk contrast was computed. The smoke tests use the T25
fixture geometry; the timing benchmark uses a non-G2 generator of the same dimension.

## Files

| File | Role |
|---|---|
| `generator.py` | G2: C = 6, T = 64, (k_c, k_t) = (2, 3), separable AR(1) noise, 1/i spectrum, channel-dependent pulse-delay knob phi |
| `predictions.py` | population constants (c_U, c_F, a_best with 64 restarts, B_cert, N\*), phi-selection rule, `predictions.json` + `protocol.json` |
| `runner.py` | confirmation workflow (fail-closed preflight, resumable per-record JSON, NPZ arrays, `summary.json`, `results.h5`) |
| `validation_gate.py` | reproduces the saved T25 weight table through the same constants path |
| `benchmark.py` | timing on a non-G2 generator of the same d (no risk recorded) |
| `tests/` | validation gate, G2 separability/population checks, runner smoke (T25 fixture), seed namespace, kit identity, preflight |

Reused code (imported unchanged, bytes pinned in `protocol.json` → `dependencies`):
`t25_production_20260930/science.py` (`moments`, `starts`, `structured`, `unrestricted`, `map_for`,
`risks`, `representation`, `covariance`, `powers`, `rng`, `digest`; accepted T25 closure `01f2f82f…`,
checked by the X3 guard before import), `p1nl_dev_20261006/tr09_first_order/core.py` (first-order
constants), `successor_20260913/t20s/generator.py` (`partial_trace_cap`).

## Environment

From the EXP root of a clean checkout at the frozen commit (no `.pyc` under `experiments/` or `src/`):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
unset PYTHONPYCACHEPREFIX
```

Python ≥ 3.10, numpy, scipy (the T25 code uses `scipy.stats.t` and `scipy.linalg.expm`), pytest;
h5py optional (without it `results.npz` replaces `results.h5`).

## Commands

```bash
# 1. kit identity and pinned dependencies
python -m experiments.mlst2026.y19_20261012.runner verify-kit
# 2. tests (about 1 minute; never touch G2 records)
python -m pytest experiments/mlst2026/y19_20261012/tests -q -p no:cacheprovider
# 3. the confirmation run (once; resumable with the same command if interrupted)
python -m experiments.mlst2026.y19_20261012.runner run --out experiments/mlst2026/y19_20261012_return --workers 4
# 4. (only if step 3 was interrupted after the last record) rebuild the summary
python -m experiments.mlst2026.y19_20261012.runner summarize --out experiments/mlst2026/y19_20261012_return
```

`run` refuses to start unless the kit hashes, pinned dependency bytes, the accepted T25 closure, the
thread variables, the seed derivation and a population recomputation of `predictions.json`
(relative tolerance 1e-6) all agree. `OUT` must lie outside this directory. An existing record is
never recomputed or overwritten; `START.json` binds `OUT` to the kit identity and `ATTEMPTS.jsonl`
logs every start. Output: `records/<label>/seed<ii>_N<N>.json`, `arrays/…npz` (fitted bases,
restart factors, mean), `summary.json` (cells, simultaneous intervals, readings, secondary
descriptives), `results.h5`, `RUN_LOG.txt`.

Size: 480 records (20 cells × 24 seeds). Measured on a non-G2 d = 384 generator: about 3 s per
record single-threaded (4 NFPA restarts, ≈ 400–520 iterations each, all stationary), i.e. about
25 CPU-minutes, ≈ 7–10 min wall with 4 workers; worst case if every restart reached the 6000-
iteration limit, ≈ 5–6 CPU-hours.

Preparation-only commands (already run; rerunning `write` changes the kit and requires a new freeze):
`python -m experiments.mlst2026.y19_20261012.predictions write|check`,
`python -m experiments.mlst2026.y19_20261012.validation_gate`,
`python -m experiments.mlst2026.y19_20261012.benchmark --out /tmp/y19_benchmark.json`,
`python -m experiments.mlst2026.y19_20261012.runner write-kit-hashes`.
