# T21-S remote commands

Placeholders are deliberately unresolved: `<DATA_ROOT>` (repository checkout on the
remote host), `<RUN_ROOT>` (an explicit run root **outside** `results/` and
`evidence/`), `<WORKERS>` (number of concurrent worker processes).  No host,
scheduler or hardware is assumed.  T21-S reads no external data: `<DATA_ROOT>` is
only the code checkout.

```
cd <DATA_ROOT>
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PKG=experiments.mlst2026.successor_20260913.t21s
PROTO=experiments/mlst2026/successor_20260913/t21s/PROTOCOL_T21S_v2.json
```

## 0. Environment record and validation (computes nothing)

```
python -c "import sys, numpy; print(sys.version.split()[0], numpy.__version__)"   # compare with requirements.txt
python -m $PKG validate  --protocol $PROTO
python -m $PKG enumerate --protocol $PROTO --out <RUN_ROOT>
```

Expected counts (from the matrix, mechanically): 2499 cells; 2496 fits (416 pilot +
2080 confirmation); 2498 solves; 3 algebraic cells; cap 2750.  Record
`protocol_sha256` and `matrix_sha256` from the `validate` output in the return
manifest; a different `matrix_sha256` means a different design and must not be run.

## 1. Pilot phase (416 fits) and the algebraic checks

```
python -m $PKG run --protocol $PROTO --out <RUN_ROOT> --roles pilot algebraic
```

Parallel form: split the pilot by family across `<WORKERS>` processes, each with its
own `--families` argument (`pilot-orthogonal-hi`, `pilot-orthogonal-lo`,
`pilot-correlated-hi`, `pilot-correlated-lo`, `algebraic`); all workers share
`<RUN_ROOT>` and the single ledger (appends serialised by an advisory `flock` on a
sidecar lock file — verified on a local disk only; **on a network filesystem run one
worker per ledger unless the remote round has checked lock behaviour**).

## 2. Pilot reduce — writes `FROZEN_PREDICTOR.json`

```
python -m $PKG reduce --protocol $PROTO --out <RUN_ROOT> --allow-partial
```

`--allow-partial` is required here because the confirmation cells are not yet
present; the output is labelled `partial: true` with `phase_completeness.pilot.complete:
true`.  The reducer writes `<RUN_ROOT>/FROZEN_PREDICTOR.json` only when **every**
pilot cell is present.  Check `confirmation_permitted` in it:

* `true` — the grid populated the below / near / above regimes; continue;
* `false` — stop.  Confirmation is refused under this protocol version; report the
  `pilot_regime_check` block; a revised grid is a new protocol file (v3) with
  provenance to v2, never an edit of v2.

Copy `FROZEN_PREDICTOR.json` into the return manifest before step 3.

## 3. Confirmation phase (2080 fits)

```
python -m $PKG run --protocol $PROTO --out <RUN_ROOT> --roles confirmation
```

Refused with exit 6 and nothing charged if `FROZEN_PREDICTOR.json` is missing, was
frozen for another protocol/matrix, or does not permit confirmation.  Parallel form
as in step 1 with the `confirmation-*` families.

## 4. Final reduce (complete run required)

```
python -m $PKG reduce --protocol $PROTO --out <RUN_ROOT>
python -m $PKG estimate --protocol $PROTO --out <RUN_ROOT>     # actual timings, for the record
```

`REDUCED.json` then carries `confirmation.verdict_by_geometry`; the interpretation
is fixed in the protocol's `failure_interpretation`.

## Resume, failure and stop behaviour

* Every attempt is charged at start (`running` record) and never refunded; a crash
  leaves a `running` record that the next `run` sweeps to `interrupted` (charged).
* Re-running the same command resumes: completed cells are skipped, failed cells
  are retried (a new attempt, charged again).  Retry policy: one retry per failed
  cell; a second failure of the same cell stops the run for inspection.  Headroom
  under the cap: 254 fits.
* `run` exits 1 on the first cell failure (see the ledger `error` field), 2 when the
  next attempt would exceed the cap, 4 when a cell breaks a scientific invariant
  (charged), 5 on a duplicate completion without `--allow-repeat`.
* `reduce` exits 3 on missing / extra / duplicate / unattributed / tampered cells
  and 4 on a failed scientific check (all-zero control off chance, stored prediction
  ≠ recomputation, frozen file missing or tampered).  Never "fix" a cell file; rerun
  the cell with `--allow-repeat` (flagged and charged).
* Nothing in `<RUN_ROOT>` is evidence until it has been returned, validated locally
  and integrated (`executed` → `results validated` → `claim supported`).

## Return

`<RUN_ROOT>/` in full: `run_matrix.jsonl`, `run_matrix_summary.json`,
`protocol_snapshot.json`, `EXECUTION_LEDGER.jsonl`, `LEDGER_SUMMARY.json`, `cells/`,
`FROZEN_PREDICTOR.json`, `REDUCED.json`, `RESOURCE_ESTIMATE.json`, plus stdout/stderr
logs of every command and the environment record from step 0.
