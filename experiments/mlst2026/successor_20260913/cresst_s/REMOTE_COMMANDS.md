# CRESST-S (L5) — exact commands

Status: **ready to transfer** once local acceptance passes.  Nothing below has been
executed as production.  Placeholders `<ARCHIVE_ROOT>`, `<CRESST_DATA_ROOT>`,
`<RUN_ROOT>` and `<WORKERS>` are left **unresolved** on purpose: no host, scheduler,
data root or core count is invented here.

```
PKG=experiments.mlst2026.successor_20260913.cresst_s
PROTO=experiments/mlst2026/successor_20260913/cresst_s/PROTOCOL_CRESST_S_v1.json
```

Run everything from the EXP repository root with that root on `PYTHONPATH`
(`python -m` from the root does this already).

## Placeholders

| placeholder | what it must be | present here? |
|---|---|---|
| `<ARCHIVE_ROOT>` | the **manuscript repository's** `evidence/data` directory.  The package reads `cresst/sensitivity/20260713_channel_robustness_v1/*` and `paper1/mlst_2026-09-10/T13a_cresst_channel_bootstrap_v1/t13a_results.json` under it, verifying each against the SHA-256 frozen in the protocol.  **There is no default.** | yes, in the local container |
| `<CRESST_DATA_ROOT>` | raw CRESST pulse traces in the `cresst-s-pulse-root/1` layout of `SCHEMA.md` | **no — absent from this environment.**  The robustness cells fail cleanly without it. |
| `<RUN_ROOT>` | an explicit output directory, **not** under the repository's `results/` or `evidence/` (the runner refuses those) | chosen by the operator |
| `<WORKERS>` | parallelism, if the host offers any.  The ledger's append lock is an advisory `flock`, adequate for local disks and **not** assumed adequate for a network filesystem: on a network filesystem run one worker per ledger (`--cells` shards, one `<RUN_ROOT>` each) and reduce them separately. | unknown |

## 0. Inspect the design (computes nothing, opens no data)

```
python -m $PKG --help
python -m $PKG validate  --protocol $PROTO
python -m $PKG enumerate --protocol $PROTO --out <RUN_ROOT>/enumerate
```

`validate` prints the protocol hash, the matrix hash and the counts
(395 cells, 1056 fits, 0 solves, cap 1400).  `enumerate` writes
`run_matrix.jsonl`, `run_matrix_summary.json` and `protocol_snapshot.json`
and creates **no** `cells/`.  Neither needs `--archive-root`.

## 1. Archive reporting (parts A and B) — the deliverable that can run today

```
python -m $PKG run --protocol $PROTO --out <RUN_ROOT>/archive \
    --archive-root <ARCHIVE_ROOT> \
    --roles control algebraic
python -m $PKG reduce --protocol $PROTO --out <RUN_ROOT>/archive \
    --archive-root <ARCHIVE_ROOT> --allow-partial
```

`--allow-partial` is needed because this root deliberately holds a subset of the matrix
(the archive families only); the reduction is labelled partial and every scientific check
still runs.  A root covering every enumerated cell reduces without it (section 4).

This executes the 1 algebraic and 20 control cells: the point-estimate recomputation, the
ten interval constructions at the archived 10 000 replicates, the five cluster reports and
the six controls.  **Measured** at these exact production configurations on the real
archived inputs (`BENCHMARK.json`, and `RESOURCE_REPORT.md` for the tables): archive point
0.0003 s, the ten intervals 10 x 2.02 = **20.2 CPU-s**, the five cluster reports
5 x 4.76 = **23.8 CPU-s**, the four science controls 4 x 0.498 = **2.0 CPU-s**, the
calibration control **7.3 CPU-s** — **53.3 CPU-seconds** of cell compute for the 21 cells,
plus about 0.1 s of per-cell CLI harness.  These are measurements of the production cells,
not a fixture time scaled by a declared factor.  It fits nothing except the 6 fits of the
estimator control.

Acceptance: `REDUCED.json` must show
`archive_recomputation.comparison.all_within_tolerance = true` (max deviation `<= 1e-12`),
`archive_intervals.all_within_tolerance = true`, every control `passed`, and
`scientific_checks.leave_one_run_out_equals_run_stratified.passed = true`.
A deviation beyond tolerance is a **failure of the package**, not a new result: stop and
report it rather than quoting any recomputed number.

## 2. Calibration (part D)

```
python -m $PKG run --protocol $PROTO --out <RUN_ROOT>/calibration \
    --archive-root <ARCHIVE_ROOT> --families calib_pilot
python -m $PKG reduce --protocol $PROTO --out <RUN_ROOT>/calibration \
    --archive-root <ARCHIVE_ROOT> --allow-partial

python -m $PKG run --protocol $PROTO --out <RUN_ROOT>/calibration \
    --archive-root <ARCHIVE_ROOT> --families calib_confirm calib_control
python -m $PKG reduce --protocol $PROTO --out <RUN_ROOT>/calibration \
    --archive-root <ARCHIVE_ROOT> --allow-partial
```

The pilot arms are the same grid at a smaller Monte-Carlo size.  **They select nothing**:
every design constant is fixed in the protocol before any cell is drawn
(`inference.pilot_selection = NONE`), so there is no frozen-selection file to write or
read, and the confirmation arms may be run without waiting for the pilot.  Running the
pilot first is still the recommended order, because a pilot failure is cheaper.

`--allow-partial` is required here only because these commands deliberately run a subset
of the matrix; a full `<RUN_ROOT>` covering every enumerated cell reduces without it.

Cost, **measured** at the production replicate budgets (all 24 designs were run):
pilot 12 x 1.41 = **16.9 CPU-seconds**, confirmation 12 x 3.69 = **44.3 CPU-seconds**,
control **7.3 CPU-seconds**.  Zero fits.

## 3. Robustness grid (part C) — needs data that does not exist here

```
# Refused today, and this is the correct behaviour:
python -m $PKG run --protocol $PROTO --out <RUN_ROOT>/robustness \
    --archive-root <ARCHIVE_ROOT> --families robustness_pilot
#   -> FAILED <cell_id>: see ledger      (exit 1)
#   -> ledger error: "input not available: raw CRESST pulse data root (<CRESST_DATA_ROOT>) ..."
```

On a host that holds the release, once an adapter has put it into the
`cresst-s-pulse-root/1` layout of `SCHEMA.md`:

```
python -m $PKG run --protocol $PROTO --out <RUN_ROOT>/robustness \
    --archive-root <ARCHIVE_ROOT> --cresst-data-root <CRESST_DATA_ROOT> \
    --families robustness_pilot
python -m $PKG reduce --protocol $PROTO --out <RUN_ROOT>/robustness \
    --archive-root <ARCHIVE_ROOT> --allow-partial

python -m $PKG run --protocol $PROTO --out <RUN_ROOT>/robustness \
    --archive-root <ARCHIVE_ROOT> --cresst-data-root <CRESST_DATA_ROOT> \
    --families robustness_confirm
python -m $PKG reduce --protocol $PROTO --out <RUN_ROOT>/robustness \
    --archive-root <ARCHIVE_ROOT> --allow-partial
```

350 cells, 1050 fits, **118 CPU-seconds** at the declared production dimensions
(400 events x 128 samples per channel): 315 confirmation cells x 0.322 s = 101.4 CPU-s and
35 pilot cells x 0.481 s = 16.8 CPU-s.  Those are **measurements of the real
`fit_and_score` at the production dimensions**, made on a synthetic pulse root written by
the package's own `write_fixture_data_root` because the raw CRESST traces are not in this
container; they are labelled `measured-on-substitute` throughout, and section 4 says the
same thing.  EMPCA runs a fixed `niter = 25`, so the cost is set by the dimensions and the
iteration count rather than by the values, and it would move roughly linearly in `n_train`
if a real channel carries a different event count — which is worth re-checking on the
first channel that runs, though it is a check on a measurement, not a replacement for a
guess.  (The superseded flat estimate for this grid was about 405 CPU-seconds; the
protocol's `fixture_to_production_scale` survives only as `cross_check_flat_fixture_scale`
in `RESOURCE_ESTIMATE.json`.)

The grid is **not part of the current claim**.  Executing it produces paired
rank/shrinkage/time-cut contrasts within each (channel, time cut) group; it does not
produce a new-run or new-detector result, and it never can, because the seven clusters
still come from two runs.

## 4. Whole matrix in one root (if the host has the raw data)

```
python -m $PKG run --protocol $PROTO --out <RUN_ROOT>/all \
    --archive-root <ARCHIVE_ROOT> --cresst-data-root <CRESST_DATA_ROOT>
python -m $PKG reduce   --protocol $PROTO --out <RUN_ROOT>/all --archive-root <ARCHIVE_ROOT>
python -m $PKG estimate --protocol $PROTO --out <RUN_ROOT>/all --archive-root <ARCHIVE_ROOT>
```

Total **measured** estimate: **235 CPU-seconds** single-core (range 211–273).  That total
already contains everything: 232.7 s of cell compute, 1.8 s of `common/cli.py` harness
(ledger append under `flock`, schema and fit checks, atomic result write) and 0.61 s of
reduce — the three are not to be added to it again.  Every family except the robustness
grid was benchmarked at its exact production configuration on the real archived inputs;
the robustness cells cannot be executed here at all (no CRESST raw pulse traces in this
container) and were measured on a synthetic pulse root written at the production
dimensions, labelled `measured-on-substitute`.  Peak memory 202 MB RSS per worker
(measured in a fresh subprocess, `cluster_report` at 10 000 replicates), output about
2.0 MB.  Wall time 0.065 h at one worker, 0.0083 h at eight, 0.0022 h at thirty-two, each
**conditional** on that many cores existing; 0.75 s of that is a serial floor (ledger
appends plus the single-process reduce) that no worker count goes below.  See
`RESOURCE_REPORT.md` for the benchmark tables and `BENCHMARK.json` for the raw timings.
The protocol's flat `fixture_to_production_scale` gives 584 CPU-s, 2.5× the measured
total; it survives in `RESOURCE_ESTIMATE.json` only as `cross_check_flat_fixture_scale`.

## 5. Development fixture (never citable)

```
python -m $PKG run --protocol $PROTO --out <RUN_ROOT>/fixture --fixture \
    --archive-root <ARCHIVE_ROOT> [--fixture-data-root <RUN_ROOT>/fixture/fixture_data]
python -m $PKG reduce   --protocol $PROTO --out <RUN_ROOT>/fixture --archive-root <ARCHIVE_ROOT>
python -m $PKG estimate --protocol $PROTO --out <RUN_ROOT>/fixture --archive-root <ARCHIVE_ROOT>
```

29 cells, about 10 s.  Development seeds only (900000–900999); every output is marked
`development_only` and the reduced statement says so.  A fixture run refuses any pilot or
confirmation cell (exit 6).  The synthetic pulse fixture is generated by the runner if the
fixture data root does not already exist.

## Resume, restart and failure behaviour

* Re-issuing the same `run` command on the same `<RUN_ROOT>` **skips completed cells**
  (`skip <cell_id>: already completed`) and continues; this is the resume path.
* Every attempt is charged in `EXECUTION_LEDGER.jsonl` **when it starts**.  Failures,
  interruptions and retries are never refunded; a retry is a new `attempt_uuid`.
* A crash or kill leaves a `running` record; the next `run` on that root sweeps it to
  `interrupted` (still charged) and says how many it swept.
* A cell failure stops that `run` invocation with exit 1.  Fix the cause and re-issue the
  same command; completed cells are not repeated.
* The retry policy is at most two retries per cell; headroom under the cap is 344 fits.
* `run` refuses a `<RUN_ROOT>` under the repository's `results/` or `evidence/` (exit 6),
  and refuses a root that already holds a different matrix hash (exit 6) — start a new
  root for a new design rather than mixing.
* `reduce` refuses missing, extra, duplicated, unattributed or byte-changed cell files
  (exit 3) unless `--allow-partial`, which labels the output partial; it refuses a failed
  scientific check with exit 4 and that must not be overridden.

## Sharding (only if the host offers parallelism)

```
python -m $PKG enumerate --protocol $PROTO --out <RUN_ROOT>/enumerate
# split run_matrix.jsonl into <WORKERS> shards of cell ids, then per shard k:
python -m $PKG run --protocol $PROTO --out <RUN_ROOT>/shard-k \
    --archive-root <ARCHIVE_ROOT> --cresst-data-root <CRESST_DATA_ROOT> \
    --cells $(cat shard-k.ids)
```

One `<RUN_ROOT>` per worker keeps one ledger per process, which is the safe mode on a
network filesystem.  Reduce each shard with `--allow-partial`; a whole-matrix reduction
needs the cell files gathered into a single root together with their ledgers.

## What to return

`run_matrix.jsonl`, `run_matrix_summary.json`, `protocol_snapshot.json`, all of `cells/`,
`EXECUTION_LEDGER.jsonl`, `LEDGER_SUMMARY.json`, `REDUCED.json`, `RESOURCE_ESTIMATE.json`
and the console log — unedited.  Do not revise the scientific design remotely; a design
change is a new protocol file with a `supersedes` pointer, decided locally.
