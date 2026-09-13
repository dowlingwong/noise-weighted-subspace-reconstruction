# Remote execution from two complete repositories

No transfer tarball or Git bundle is required. Copy/clone the full manuscript and experiment repositories, including the ignored historical `results/mlst_2026-09-10/` and `results/t22_2026-09-11/` trees and the manuscript `evidence/` tree. A Git clone alone may omit ignored data. `MANIFEST.json` verifies the exact scientific inputs and code; never regenerate it remotely to hide a mismatch. Old bundles under `transfer/` predate the reviewed corrections and are obsolete for this round.

The manifest records the commit at freeze time as history; file hashes remain authoritative after committing the manifest itself. It excludes unrelated notebooks/noise-module work and includes the successor runner code, imported experiment/src modules, tests, frozen protocols and archived input files. It does not publish anything or grant permission to release held evidence.

## Server preparation

Use a Python 3.12 environment. The tested dependency pins are in `requirements-tested.txt`. Set these four paths on the server; `RUN_BASE` must be outside both repositories. No scheduler or host is assumed.

```sh
export EXP=/absolute/path/to/noise-weighted-subspace-reconstruction
export MS=/absolute/path/to/arXiv-v1-polished-extended-preprint
export RUN_BASE=/absolute/path/to/new/mlst-successor-runs
export PY=/absolute/path/to/python3.12-environment/bin/python
cd "$EXP"
"$PY" -m pip install -r experiments/mlst2026/successor_20260913/remote/requirements-tested.txt
"$PY" experiments/mlst2026/successor_20260913/remote/prepare.py verify --ms "$MS"
"$PY" experiments/mlst2026/successor_20260913/remote/prepare.py preflight --ms "$MS" --out "$RUN_BASE/preflight"
```

Preflight checks all manifest hashes and production matrix identities, verifies the manuscript evidence archive, runs the development/test suites with **zero skipped tests**, enumerates the five production matrices without executing them, and records the environment and development-test wall time. A failed preflight removes any old green receipt. Inspect `preflight/tests.log` for timings and `PREFLIGHT.json` for acceptance. This is required on the actual execution host, inside its allocation if a scheduler is used, even though local acceptance passed. A receipt from a login node does not authorize a different compute node. Keep resumptions on the same host; migration of running attempts between hosts needs an explicit interruption audit.

The launcher forces one BLAS thread and holds one process lock per output root. Run **one launcher on one host per root**. This mode does not claim cross-node lock safety; do not start concurrent jobs for the same root, shard cell files or combine independent ledgers manually. Request CPU resources, not GPUs. Scheduler wall time and memory must be chosen for the actual server; package fixture projections below are not measured production timings.

## Production commands, after preflight

Run the packages in separate persistent roots. The launcher stops on any failure, preserves logs, enforces at most two retries per cell, and never raises fit caps. T20/T21/T22 pilot reduction precedes confirmation and writes the existing package freeze artifacts. Each reduction is copied into the invocation's logs so a pilot report is not lost when the final report is written.

```sh
"$PY" experiments/mlst2026/successor_20260913/remote/prepare.py execute --ms "$MS" --receipt "$RUN_BASE/preflight/PREFLIGHT.json" --out "$RUN_BASE/t19s" --selection t19s
"$PY" experiments/mlst2026/successor_20260913/remote/prepare.py execute --ms "$MS" --receipt "$RUN_BASE/preflight/PREFLIGHT.json" --out "$RUN_BASE/t20s" --selection t20s
"$PY" experiments/mlst2026/successor_20260913/remote/prepare.py execute --ms "$MS" --receipt "$RUN_BASE/preflight/PREFLIGHT.json" --out "$RUN_BASE/t21s" --selection t21s
"$PY" experiments/mlst2026/successor_20260913/remote/prepare.py execute --ms "$MS" --receipt "$RUN_BASE/preflight/PREFLIGHT.json" --out "$RUN_BASE/t22s" --selection t22s
"$PY" experiments/mlst2026/successor_20260913/remote/prepare.py execute --ms "$MS" --receipt "$RUN_BASE/preflight/PREFLIGHT.json" --out "$RUN_BASE/cresst-archive" --selection cresst-archive
"$PY" experiments/mlst2026/successor_20260913/remote/prepare.py execute --ms "$MS" --receipt "$RUN_BASE/preflight/PREFLIGHT.json" --out "$RUN_BASE/cresst-calibration" --selection cresst-calibration
```

These are explicit production commands; preflight does not run them. Resume an interrupted selection with the identical command and root after inspecting the failure. Completed cells are skipped; failed/interrupted attempts remain charged. A stale protocol, changed source, changed environment, exhausted retries or mismatched matrix is a stop condition. Re-run preflight after a legitimate environment change; source/protocol amendments require a reviewed new manifest, not a silent remote edit.

| Package | Frozen cells | Fits | Existing fixture-scaled single-worker projection |
|---|---:|---:|---:|
| T19-S | 136 | 3,840 | ~14 minutes |
| T20-S | 964 | 4,800 | ~2.04 hours |
| T21-S | 2,499 | 2,496 | ~1.36 hours |
| T22-S simulation | 727 | 0 | **~105.23 hours** |
| CRESST-S whole design | 395 | 1,056 | ~10 minutes, unvalidated for real pulses |

Do not interpret zero fits as zero simulation cost. T22 may need multiple resumptions on queues shorter than this projection; use adequate termination grace and retain the run root. Development test timings are diagnostic, not a production scaling guarantee. Do not change scientific dimensions or use confirmation results for runtime tuning.

## Explicitly blocked portions

- **CRESST robustness:** requires raw pulse/noise/time arrays in `cresst-s-pulse-root/1` format. The full repositories contain archived residuals, which are insufficient. The release-to-schema adapter has not been validated on raw data. The launcher deliberately does not offer a full-robustness mode. On the server, locate real data, verify provenance and independent splits, implement/validate that adapter, then follow `../cresst_s/REMOTE_COMMANDS.md` under a reviewed data preflight. Do not substitute development fixtures for real pulses. Archive/calibration outputs are intentionally partial results.
- **T22 real instrument comparison:** not executable under this simulation protocol. It still needs a frozen matched design, feasible independent unexposed units and raw data. Do not treat the simulation run as an instrument verdict.

## Return and review

Return the whole run roots, preflight directory, manifest, stdout/stderr logs, frozen pilot artifacts, environment record and both repositories' commit IDs. Preserve every ledger record and cell file. `REDUCED.json` and successful process exits do not alone establish a claim. Reconcile matrix counts, checksums, failed attempts, complete/partial status and package scientific controls; rerun the reducer on a copy before updating manuscript tables or claims. Negative/inconclusive results must be retained.

Accounting limitation: directly instrumented estimators are guarded before invocation. Audited legacy ALS and T22 estimator batches report their counts after returning; an unexpected batch overrun is recorded in full and stops the run, but already-performed batch work cannot be prevented retrospectively. This is not an absolute CPU or internal-batch fit cap. No overruns are permitted to continue silently or be refunded.

Local acceptance: the final repository preflight passed **136 tests with zero failures/errors/skips**, and all five matrices were enumerated. `LOCAL_ACCEPTANCE.json` and `LOCAL_TESTS.txt` preserve that record. The local receipt is not a substitute for the target server's own `PREFLIGHT.json`.
