# R3: T21-S finite-size recovery before a loadings claim

**Status: locally tested (development fixture only); configured and ready to transfer. No production
pilot or confirmation cell has been run, locally or remotely. No scientific result exists yet.**

This is a new-protocol follow-up (`r3_t21_finite`, `PROTOCOL_R3_T21_FINITE_v1.json`) to the immutable
T21-S package under `experiments/mlst2026/successor_20260913/t21s/`. It was designed **after** the
September 2026 results were observed. It does not edit, re-run or re-score T21-S; the archived
geometry and predictor are imported unchanged.

## What it asks

The old T21-S primary failures both sat at `orthogonal|hi|N128` (γ = d/N = 2, N = 128, d = 256), the
place where finite-size smoothing of the asymptotic overlap is largest. R3 holds **γ, the population
spike strengths and the loading geometry fixed** while scaling d and N together over the ladder
`d0 = 256, 2·d0, 4·d0` and asks whether the archived asymptotic prediction approaches the observed
rank-k PCA recovery as the ambient dimension grows.

## Design

- **Ladder** (fixed γ): hi arm γ = 2 → N = 128, 256, 512; lo arm γ = 0.25 → N = 1024, 2048, 4096;
  d = 256, 512, 1024. The base rung's hi cell is exactly the old failure ambient size.
- **Targeted, not the full grid**: 6 grid configurations per geometry × γ arm × rung
  (below-edge / at-edge / above-edge, the two old failure types at ratio 0.5 and 2.0, and an upper
  transition at 1.5) plus `all_zero` and `all_strong` controls = 8 per cell, 96 configurations.
  Conclusions are targeted at these configurations and are **not** validation of the old
  104-configuration grid.
- **Baseline retained**: the archived `dgj2018-eq1.6` isolated-spike × loadings predictor, the 0.05
  primary / 0.10 near-edge tolerances and the `[0.7, 1.8]` band with `loading_min = 0.05`.
- **Finite-size endpoint (predeclared)**: for the two old failure configuration types, the seed-mean
  absolute error at `4·d0` must be ≤ 0.05 **and** strictly smaller than at `d0` (decline), and at
  `4·d0` every legacy-primary pair must pass 0.05 in the orthogonal arm (agreement = the prerequisite).
  Inference is across independent seeds only (sd, se, fixed-seed percentile CI95, 1000 replicates).
- **Alternative candidate (separate, labeled)**: a small predeclared family of wider near-edge bands,
  selected **only on fresh pilot seeds** under a predeclared rule (narrowest adequate band, or
  `none`), frozen in `FROZEN_EDGE.json` before any confirmation cell is drawn. Confirmation is
  refused without that file (exit 6, nothing charged). The candidate never replaces the legacy
  classification, which is always reported.
- **Gated interpretation**: unless the orthogonal isolated-spike prerequisite passes at the largest
  rung, a correlated-arm pass cannot support the added-loadings step and that step is marked
  **untested** (the archived `predictor_fails_on_orthogonal_only` rule carried forward).

## Seeds

Fresh namespaces, disjoint from all old and new packages:
development `[922000, 922999]`, pilot `[25000, 25299]`, confirmation `[26000, 26499]`.
Allocations: 12 pilot seeds and 24 confirmation seeds per configuration.

## Commands

Local development fixture (development seeds only, never citable):

```bash
PROTO=experiments/mlst2026/followup_20260917/r3_t21_finite/PROTOCOL_R3_T21_FINITE_v1.json
.venv/bin/pytest tests/test_followup_20260917_r3.py -q
.venv/bin/python -m experiments.mlst2026.followup_20260917.r3_t21_finite validate --protocol "$PROTO"
.venv/bin/python -m experiments.mlst2026.followup_20260917.r3_t21_finite enumerate --protocol "$PROTO" --out <RUN_ROOT>
```

Production execution is described exactly in [`REMOTE_COMMANDS.md`](REMOTE_COMMANDS.md). It is a
two-stage flow: run/reduce the pilot (which writes `FROZEN_EDGE.json`), then run/reduce confirmation.

## Measured resource estimate (this laptop)

`BENCHMARK.json` is a **measured** benchmark of the production-sized cell kernel (no production run,
no reserved seed; 28.7 s wall, numpy 2.4.6). Median per-cell seconds: d256 hi 0.012 / lo 0.031;
d512 hi 0.057 / lo 0.151; d1024 hi 0.33 / lo 0.83. The estimate projects ≈ 817 s of **cell compute**
for the 3456 production fits (all 25 families measured, `unmeasured_families: []`). This is the cell
kernel only; the estimate reports reduce and harness as unmeasured in this package, and the wall-time
rows are conditional on unknown server cores/speed/contention. The estimate's "benchmark Xeon"
wording is generic `common.cli` text; the actual measured platform is recorded in `BENCHMARK.json`.

## Status vocabulary

The only truthful status today is **configured / locally tested / ready to transfer**. It is *not*
"executed": no production cell exists, and no hypothesis is supported. The loadings step is
**untested** and remains untested until a fresh confirmation run passes the orthogonal prerequisite.

## Honesty notes and unresolved issues

- Fixture (`development_only`) output is never citable; the chance level at the fixture d exceeds the
  primary tolerance, so fixture agreement numbers are meaningless.
- R3 restates the old outcome verbatim: **296/298** primary pairs passed, the old verdict is **not
  supported overall**, and the **five registered T21 predictions remain NOT met**. Nothing here
  repairs them.
- The candidate is expected to be `none` at production if no wider band is adequate on pilot; that is
  a valid, reportable outcome, not a failure to be tuned away.
- If the ladder does not show the decline, the correct output is "endpoint not met", not a new grid
  chosen after seeing confirmation.
- Not yet done: a remote production run. No `REDUCED.json` with `development_only: false` exists.

## A3 — executable ladder-preservation receipt

`LADDER_PRESERVATION_RECEIPT.json` is an executable pass/fail receipt at every
rung of the production d-ladder (d = 256, 512, 1024), not protocol prose:

```sh
python -m experiments.mlst2026.followup_20260917.r3_t21_finite preservation-receipt \
  --protocol PROTOCOL_R3_T21_FINITE_v1.json --out LADDER_PRESERVATION_RECEIPT.json --production
```

All 32 configurations pass at every rung (Gram, population spectrum, Schur
unique-norm ratio, and c2 spike invariance). Exit 4 if any rung fails.
