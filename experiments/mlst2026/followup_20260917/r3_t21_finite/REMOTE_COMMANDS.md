# R3 remote execution commands

Package `r3_t21_finite` (T21-S finite-size recovery before a loadings claim).
Protocol `PROTOCOL_R3_T21_FINITE_v1.json`. Run from the repository root under the pinned
environment. Production execution is a **two-stage** flow: the pilot reduce writes the frozen
candidate that confirmation must have.

Set the paths:

```bash
export PROTOCOL=<PROTOCOL>          # e.g. experiments/mlst2026/followup_20260917/r3_t21_finite/PROTOCOL_R3_T21_FINITE_v1.json
export RUN_ROOT=<RUN_ROOT>          # e.g. /ceph/<user>/r3-t21-finite-20260917
```

## 0. Preflight (no production cells)

```bash
python -m experiments.mlst2026.followup_20260917.r3_t21_finite validate --protocol "$PROTOCOL"
python -m experiments.mlst2026.followup_20260917.r3_t21_finite enumerate --protocol "$PROTOCOL" --out "$RUN_ROOT"
```

`validate` must report `fits_within_cap: true` (3456 fits, cap 4600). `enumerate` writes
`run_matrix.jsonl` and `run_matrix_summary.json` and computes nothing.

## 1. Pilot stage (1152 fits)

```bash
python -m experiments.mlst2026.followup_20260917.r3_t21_finite run \
    --protocol "$PROTOCOL" --out "$RUN_ROOT" --roles pilot
python -m experiments.mlst2026.followup_20260917.r3_t21_finite reduce \
    --protocol "$PROTOCOL" --out "$RUN_ROOT" --allow-partial
```

The pilot reduce selects the alternative near-edge band on the pilot seeds only and writes
`$RUN_ROOT/FROZEN_EDGE.json`. Verify it exists before continuing. Running confirmation cells
before this file exists is refused (exit 6, nothing charged).

## 2. Confirmation stage (2304 fits)

```bash
python -m experiments.mlst2026.followup_20260917.r3_t21_finite run \
    --protocol "$PROTOCOL" --out "$RUN_ROOT" --roles confirmation
python -m experiments.mlst2026.followup_20260917.r3_t21_finite reduce \
    --protocol "$PROTOCOL" --out "$RUN_ROOT"
```

The final reduce writes `$RUN_ROOT/REDUCED.json` with the legacy scoring, the frozen candidate
scoring, the finite-size endpoint (`finite_size_endpoint`) and the gated interpretation
(`gated_interpretation`). The loadings step is only supported if the orthogonal prerequisite passes.

## 3. Resource estimate

```bash
python -m experiments.mlst2026.followup_20260917.r3_t21_finite estimate \
    --protocol "$PROTOCOL" --out "$RUN_ROOT"
```

## Parallelism

Cells are independent and may be split by explicit cell id across workers with disjoint ownership.
`<WORKERS>` is a scheduler choice; the estimate's wall-time rows are conditional on unknown server
cores, per-core speed, BLAS threading and filesystem flock. **Do not assume multi-host `flock` or a
shared-filesystem ledger is safe on the target without validating it**; the serial fallback is a
single `run` invocation with no `--roles` filter.

```bash
# serial fallback (safe anywhere, ~14 minutes of measured cell compute)
python -m experiments.mlst2026.followup_20260917.r3_t21_finite run \
    --protocol "$PROTOCOL" --out "$RUN_ROOT"
```

Use `<DATA_ROOT>` only for the archive-wide preflight if required by the host; R3 draws all its own
synthetic data and reads no external data root.

## Retries and refusals

- One retry per failed cell, charged; a second failure of the same cell stops the run for inspection.
- `reduce` without `--allow-partial` refuses an incomplete run (exit 3).
- A tampered cell (bytes differing from the ledger digest) is refused (exit 3); a tampered
  `FROZEN_EDGE.json` or a cell whose classification disagrees with the protocol band is refused
  (exit 4); an all-zero control off chance is refused (exit 4).
