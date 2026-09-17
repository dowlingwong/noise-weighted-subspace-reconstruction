# R2 — T20-S v3 theta=0.12 scaling and crossover

**Status: locally tested.** The fixture end-to-end (`run` + `reduce`) and the full
test file `tests/test_followup_20260917_r2.py` pass locally on the fixture
(development) protocol block. **No production cell has been run and there is no
scientific result here.** Fixture outputs are `development_only` and are never
citable.

New protocol `PROTOCOL_R2_T20_THETA012_v1.json` (`mlst-local-package/1`, package
`r2_t20_theta012`) superseding `successor_20260913/t20s/PROTOCOL_T20S_v2.json`.
It was designed after the September 2026 T20-S results were observed; the old
trees are immutable and are imported unchanged.

## What it tests

Two separate endpoints on the analytic fixed-spectrum geometry
`K(theta) = V_sep expm(theta A) diag(mu) ...` of the archived generator:

* **Endpoint (a)** — fresh-seed evaluation of the **OLD** frozen t20s prediction
  (`c/N + d/N^2` on `d - a_ref`) at the focal geometry `theta = 0.12` and the
  reference geometry `theta = 0.25`, over the old N grid
  `[400, 800, 1000, 1250, 1600, 3200]`. Predictions at `N = 6400, 12800` are
  reported and labelled `extrapolated_beyond_old_grid`.
* **Endpoint (b)** — a **NEW** model selected only on fresh pilot seeds and
  frozen (`FROZEN_MODEL.json`) before any independent confirmation cell, over the
  extended grid through `N = 12800`. `"none"` is a valid outcome.

`theta = 0.0` is the product control. The full four-geometry confirmation
(`theta = 0.40`) is a **declared, costed optional expansion** — not required.

### Offset convention (the September trap)

The t20s reducer stored `prediction_at_grid` **already including `a_ref`**
(`successor_20260913/t20s/runner.py:371`). R2 rebuilds every d-prediction from the
original `(c, d)` and the **current** algebraic `a_ref`, added exactly once. A
double application (`base + 2 a_ref`) is numerically distinct whenever `a_ref` is
nonzero and is **rejected at reduce with exit 4**. This is enforced by
`analysis.check_offset_applied_once`, exercised by the `offset_applied_once`
control cell, and covered by two regression tests (a direct unit test and a
reduce-level mutation test).

The two archived pilot selection margins are kept distinct: the reported 2.7%
margin was against `"none"` (CV `0.2975` vs `0.2895`); the margin against
`c/N^alpha` is the smaller `0.2907` vs `0.2895` (0.43% relative). A regression
test asserts they are different quantities.

## Matrix and budget (required production)

| role | cells | fits | solves |
|---|---:|---:|---:|
| algebraic (theta = 0.0, 0.12, 0.25) | 3 | 0 | 105 |
| pilot (16 seeds x 3 theta x 8 N) | 384 | 1920 | 0 |
| confirmation (24 seeds x 3 theta x 8 N) | 576 | 2880 | 0 |
| controls | 3 | 0 | 0 |
| **required total** | **966** | **4800** | **105** |

`estimator_fits_cap = 7000` = 4800 required + 1600 optional four-geometry
expansion + 600 retry headroom. The optional `theta = 0.40` expansion adds 321
cells, 1600 fits and 35 solves (1287 cells / 6400 fits / 140 solves total); it is
enabled only by `R2_T20_INCLUDE_OPTIONAL=1` (or
`runner._build_matrix(P, fixture, include_optional=True)`).

Fresh seed namespaces (never used before): development `[921000, 921999]`,
pilot `[23000, 23299]`, confirmation `[24000, 24499]`.

## Measured local per-cell timing

Single BLAS thread, implementation machine (7 GiB Mac). These are measurements
for the resource record, not production results:

* production fit cell (`d = 256`, `R = 4`, `als_n_iter = 150`):
  `N = 400`: **1.74 s**, `N = 800`: **3.46 s**, `N = 1600`: **6.73 s** (linear in N).
* production algebraic cell (cold cache, all required thetas): **1.07 s**.
* fixture fit cell (`d = 12`, `R = 2`) over the extended grid
  `N = 400 ... 12800`: 0.068 → 1.72 s, grid mean **0.490 s**.
* fixture algebraic cell: 0.0028 s.

Projected required production compute, from the measured linear trend
(~4.2e-3 s per N): about 116 s per (seed, theta) over the 8-N grid, times
120 (seed, theta) pairs ≈ **1.4e4 CPU-s ≈ 3.9 CPU-hours** single-threaded,
before the harness and reduction. This is a projection from measured points;
it is not a measured total.

## Resource estimate

No `BENCHMARK.json` is shipped with this package. `estimate` therefore reports
`measured_projection_available: false` and marks every family
`unmeasured`, and computes only the flat fixture-scale cross-check that
`common.cli` explicitly labels as **not the estimate**. This is deliberate: a
measured projection is not fabricated.

## Local verification

```
.venv/bin/pytest tests/test_followup_20260917_r2.py -q
.venv/bin/python -m experiments.mlst2026.followup_20260917.r2_t20_theta012 validate \
    --protocol experiments/mlst2026/followup_20260917/r2_t20_theta012/PROTOCOL_R2_T20_THETA012_v1.json
```

A fixture end-to-end run (development-only, never citable):

```
OUT=/tmp/r2_fixture
PKG=".venv/bin/python -m experiments.mlst2026.followup_20260917.r2_t20_theta012"
PROTO=experiments/mlst2026/followup_20260917/r2_t20_theta012/PROTOCOL_R2_T20_THETA012_v1.json
$PKG run    --protocol $PROTO --out $OUT --fixture --families algebraic pilot control
$PKG reduce --protocol $PROTO --out $OUT --allow-partial      # writes FROZEN_MODEL.json
$PKG run    --protocol $PROTO --out $OUT --fixture --families confirmation
$PKG reduce --protocol $PROTO --out $OUT
$PKG estimate --protocol $PROTO --out $OUT
```

## Production command

See `REMOTE_COMMANDS.md` for the exact remote command sequence. The production
run must execute the pilot phase and the pilot reduce (which writes
`FROZEN_MODEL.json`) **before** any confirmation cell is drawn; the runner
refuses a confirmation cell without the freeze (exit 6, nothing charged).

## Explicitly not claimed

* a certified restricted optimum `P*_F`, or an exact crossover `N*` (a crossing
  is reported only as an adjacent grid bracket, and only if adjacent decided
  opposite signs exist; otherwise the grid bound is reported and no crossing is
  claimed);
* any scientific result from fixture (development) runs;
* any transfer to the archived Monte-Carlo pulse-family proxy: the historical
  `c/N` refutation, the `N* ~ 960` interpolation and the 800/1600 sign reversal
  stand and are neither confirmed nor revised here;
* that the optional four-geometry expansion was executed;
* that the archived t20s verdicts are revised.

## A3 — simulator-truth audit

`TRUTH_AUDIT.json` (production design, all checks pass) is generated by:

```sh
python -m experiments.mlst2026.followup_20260917.r2_t20_theta012 truth-audit \
  --protocol PROTOCOL_R2_T20_THETA012_v1.json --out TRUTH_AUDIT.json --production
```

It audits the inherited analytic generator: fixed spectrum, finite values, no
clipping in the generator source, and the nonseparability invariant, and declares
the R1 truth-audit failure modes (heavy tails, heterogeneity-dependent generation,
dependence) absent by construction. A generator correction would be a new version
with disclosure.
