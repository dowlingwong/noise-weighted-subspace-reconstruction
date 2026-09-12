# T19-S v2 — remote commands

Placeholders are **unresolved on purpose**: `<DATA_ROOT>` (checkout of EXP at the
transferred revision), `<RUN_ROOT>` (a writable directory outside `results/` and
`evidence/`), `<WORKERS>` (number of concurrent `run` processes, if any).  No
host, scheduler or hardware is assumed.  All commands run from `<DATA_ROOT>`
with the Python of `requirements.txt`.

```
export PYTHONPATH=<DATA_ROOT>
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # recorded in every cell's provenance
M="python -m experiments.mlst2026.successor_20260913.t19s"
PROTO=experiments/mlst2026/successor_20260913/t19s/PROTOCOL_T19S_v2.json
```

## 0. Check the transfer (computes nothing)

```
$M validate  --protocol $PROTO
$M enumerate --protocol $PROTO --out <RUN_ROOT>
sha256sum $PROTO                       # must equal protocol_sha256 in the local acceptance report
cat <RUN_ROOT>/run_matrix_summary.json # expected: 136 cells, 3840 fits, 3605 solves; pilot 20/640, confirmation 100/3200
```

## 1. Pilot (20 cells, 640 fits, seeds 1000–1019)

```
$M run --protocol $PROTO --out <RUN_ROOT> --roles pilot
```

Resume after a crash or interruption: rerun the same command.  Completed cells
are skipped; a `running` attempt whose process is gone is swept to
`interrupted` and stays charged; the retry is a new charged attempt.  A cell
that fails is retried at most twice (`budget.retry_policy`); the run stops at
the first failure (exit 1) so that the ledger is inspected before continuing.
With `<WORKERS>` > 1 start several identical `run` processes on the same
`<RUN_ROOT>` only if the filesystem honours advisory `flock` on
`EXECUTION_LEDGER.jsonl.lock`; otherwise run one worker (the serial fallback).

## 2. Freeze the pilot selection (partial reduce, by construction)

```
$M reduce --protocol $PROTO --out <RUN_ROOT> --allow-partial
cat <RUN_ROOT>/PILOT_SELECTION.json
```

`REDUCED.json` is labelled `partial: true` (confirmation cells absent) and
`stage: pilot`.  `PILOT_SELECTION.json` is written once and never overwritten;
its sha256 goes into the handoff log.  Do not edit it.

## 3. Confirmation (100 cells, 3200 fits, seeds 2000–2099) and the deterministic cells

```
$M run --protocol $PROTO --out <RUN_ROOT> --roles confirmation
$M run --protocol $PROTO --out <RUN_ROOT> --roles algebraic control
```

(or simply `$M run --protocol $PROTO --out <RUN_ROOT>` to run everything not yet
completed).  A confirmation cell started before step 2 refuses to run: the
attempt is charged, recorded `failed` in the ledger and the run exits 1.  Every
confirmation cell records the sha256 of the selection it saw.

## 4. Final reduce (complete)

```
$M reduce --protocol $PROTO --out <RUN_ROOT>
$M estimate --protocol $PROTO --out <RUN_ROOT>         # actual per-family wall times of this run
```

`reduce` refuses (exit 3) missing, extra, duplicated, unattributed or tampered
cells, a missing or tampered selection, or a selection that differs from the one
the pilot cells imply; it exits 4 if any deterministic or paired invariant
failed.  `--allow-partial` reduces what exists and labels the output partial.

## 5. Return

Return the whole `<RUN_ROOT>` (cells, ledger, `PILOT_SELECTION.json`,
`REDUCED.json`, `RESOURCE_ESTIMATE.json`, `run_matrix*`, `protocol_snapshot.json`)
plus stdout/stderr logs, `pip freeze`, `python -c "import numpy; numpy.show_config()"`,
and `uname -a`.  Do not post-process on the remote host.  The local round
validates the returned bytes against the ledger digests before anything is
integrated; the status then moves from `executed` to `results validated` only
after that check.

## Resource estimate (local, measured)

`RESOURCE_ESTIMATE_PRODUCTION.json` from the fixture root
(`$M estimate-production --protocol $PROTO --out <FIXTURE_ROOT>`): the local
fixture timings give 120 × 6.9 s (paired) + 16 × ≈ 1 s (deterministic) ≈ 840 CPU-s
≈ 14 CPU-minutes on one core at the local speed (direct measurement of
production-size cells: 6.5–7.8 s each), before retries; peak memory ≈ 0.3 GB per
worker; disk under 20 MB of JSON.  Retry headroom: 960 fits under the cap =
30 paired cells.
