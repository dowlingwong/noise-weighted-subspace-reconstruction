# T20-S v2 — remote commands

Placeholders `<DATA_ROOT>`, `<RUN_ROOT>`, `<WORKERS>` are **unresolved** on purpose:
no host, scheduler or hardware is assumed here. `<DATA_ROOT>` is the checkout of the
experiment repository; `<RUN_ROOT>` must be outside `results/` and `evidence/` (the
runner refuses those). This package draws no external data.

Every command below is run from `<DATA_ROOT>` with

```
export PYTHONPATH=<DATA_ROOT>
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # timings were measured single-threaded
PKG="python -m experiments.mlst2026.successor_20260913.t20s"
PROTO=experiments/mlst2026/successor_20260913/t20s/PROTOCOL_T20S_v2.json
```

## 0. Environment record and validation (computes nothing)

```
python -c "import numpy, scipy, sys; print(sys.version.split()[0], numpy.__version__, scipy.__version__)"   # compare with requirements.txt
$PKG validate  --protocol $PROTO
$PKG enumerate --protocol $PROTO --out <RUN_ROOT>
```

Expected from `enumerate`: 964 cells = 4 algebraic + 384 pilot + 576 confirmation;
4800 fits (5 per fit cell), 140 solves; `matrix_sha256` as printed by `validate` locally
(record both in the handoff). If the counts differ, stop: the protocol or code differs.

## 1. Pilot phase (fresh pilot seeds 3000–3015; algebraic cells first)

```
$PKG run --protocol $PROTO --out <RUN_ROOT> --families algebraic
$PKG run --protocol $PROTO --out <RUN_ROOT> --families pilot
```

Parallel execution: `run` is resumable and skips completed cells, so `<WORKERS>` may be
started on disjoint cell subsets with `--cells <id ...>` (ids from
`<RUN_ROOT>/run_matrix.jsonl`). All workers append to one `EXECUTION_LEDGER.jsonl`
with an advisory `flock`; this was verified on a local disk only. **Until the remote
filesystem's lock behaviour is checked, run one worker per ledger (serial fallback)**
or give each worker its own `--out` and reduce them separately (a separate root is a
separate run; do not merge cell files by hand).

## 2. Pilot reduce — freezes the selection (must precede any confirmation cell)

```
$PKG reduce --protocol $PROTO --out <RUN_ROOT> --allow-partial
```

`--allow-partial` is needed because the confirmation cells do not exist yet; the
package's own reducer still refuses (exit 3) unless **every** pilot and algebraic cell is
present, and it writes `<RUN_ROOT>/FROZEN_SELECTION.json` exactly once. `REDUCED.json`
carries `phase: pilot`, `pilot_complete: true` and a `partial_note`. Copy
`FROZEN_SELECTION.json` and `REDUCED.json` out now (they are the pilot record).

## 3. Confirmation phase (fresh seeds 4000–4023)

```
$PKG run --protocol $PROTO --out <RUN_ROOT> --families confirmation
```

Refused with exit 6 **before anything is charged** if `FROZEN_SELECTION.json` is missing
from `<RUN_ROOT>`, was frozen under a different protocol, or for a different matrix.

## 4. Confirmation reduce and resource record

```
$PKG reduce   --protocol $PROTO --out <RUN_ROOT>
$PKG estimate --protocol $PROTO --out <RUN_ROOT>
```

`reduce` refuses (exit 3) a missing/extra/duplicate/unattributed/tampered cell, a missing
`FROZEN_SELECTION.json`, or pilot cells whose digest no longer matches the frozen one;
exit 4 if any scientific self-check fails on a cell. `estimate` writes
`RESOURCE_ESTIMATE.json` from the measured benchmarks in `BENCHMARK.json` × the full
production matrix (every production `N` was timed at `d = 256`); its
`production_projection` block restates that total per family and adds the per-worker cold
population-reference cost the per-cell figures exclude. The protocol's flat
`fixture_to_production_scale` appears only as `cross_check_flat_fixture_scale` and is not
the estimate.

## Resume, failure and interruption behaviour

* every attempt is charged at start; a crash, `SIGKILL` or exception leaves a `running`
  → `interrupted`/`failed` record that is never refunded; the next `run` sweeps stale
  `running` records of dead local pids and re-attempts the cell (a new `attempt_uuid`);
* the cap is 5400 fits for 4800 planned: 120 whole-cell retries of headroom; when the cap
  would be exceeded the runner exits 2 and nothing more is started — report, do not raise
  the cap;
* `--allow-repeat` re-runs a completed cell (charged again, flagged in the ledger); never
  needed in the normal flow;
* a partial confirmation phase can be reduced with `--allow-partial` and is labelled
  `partial: true`; it is a partial result, not a confirmation.

## Return to local validation

Return the whole `<RUN_ROOT>` (cells/, `EXECUTION_LEDGER.jsonl`, `LEDGER_SUMMARY.json`,
`run_matrix*.json*`, `protocol_snapshot.json`, `FROZEN_SELECTION.json`, both
`REDUCED.json` snapshots, `RESOURCE_ESTIMATE.json`, stdout/stderr logs) plus the
environment record. Local validation re-runs `reduce` on the returned root and compares
`matrix_sha256`, `protocol_sha256` and the ledger totals before anything is integrated.
