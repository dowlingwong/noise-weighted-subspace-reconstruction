# T22-S remote commands

Placeholders are left unresolved on purpose: `<RUN_ROOT>` (a writable directory
outside the repository's `results/` and `evidence/`), `<WORKERS>` (kept at **1**
in this document, see "Execution mode"), `<DATA_ROOT>` (NOT needed by part A;
part B is not to be run remotely under this protocol, see the last section).
Do not invent hosts, schedulers or hardware; audit the server first.

All commands run from the experiment repository root with the protocol
`experiments/mlst2026/successor_20260913/t22s/PROTOCOL_T22S_v2.json`
(`PROTO` below).  One run root, one ledger, one matrix; the pilot and the
confirmation are two stages of the same root.

```
PROTO=experiments/mlst2026/successor_20260913/t22s/PROTOCOL_T22S_v2.json
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## 0. Preparation (computes nothing)

```
python -m experiments.mlst2026.successor_20260913.t22s validate  --protocol $PROTO
python -m experiments.mlst2026.successor_20260913.t22s enumerate --protocol $PROTO --out <RUN_ROOT>
```

Expected: 727 cells (32 pilot, 690 confirmation, 5 control), 0 fits, 0 solves,
`fits_within_cap: true`.  Record `run_matrix_summary.json` (`matrix_sha256`)
and the protocol digest in the handoff.

## 1. Pilot stage (tuning scenarios, both replicate settings, plus controls)

```
python -m experiments.mlst2026.successor_20260913.t22s run --protocol $PROTO --out <RUN_ROOT> --families sim_pilot control
```

Serial: one process, one worker, cells in matrix order.  On interruption or
crash re-issue the same command: completed cells are skipped, the interrupted
attempt is swept into the ledger as `interrupted` (charged, never refunded),
and the cell is attempted again under a new `attempt_uuid`.  The failing cell
of a run that exits 1 is named in `EXECUTION_LEDGER.jsonl`; the retry is the
same command (at most two retries per cell under the protocol's retry policy).

## 2. Pilot reduce -> freeze

```
python -m experiments.mlst2026.successor_20260913.t22s reduce --protocol $PROTO --out <RUN_ROOT> --allow-partial
```

`--allow-partial` is required here because the confirmation cells are, by
construction, not yet present; the output is labelled `partial: true`.  The
reducer refuses (exit 4) if any control failed, and writes
`<RUN_ROOT>/FROZEN_CALIBRATION.json` only when **every** pilot cell is present.
It never overwrites an existing freeze (a differing one is exit 3).  Copy the
freeze's digest into the handoff record before step 3.

## 3. Confirmation stage (validation scenarios, frozen setting)

```
python -m experiments.mlst2026.successor_20260913.t22s run --protocol $PROTO --out <RUN_ROOT> --families sim_confirm
```

Refused (exit 6) if `FROZEN_CALIBRATION.json` is absent; each cell records the
freeze's digest.  Resume behaviour as in step 1.  690 cells; the resource
estimate (below) gives the expected CPU time.

## 4. Final reduce and estimate

```
python -m experiments.mlst2026.successor_20260913.t22s reduce   --protocol $PROTO --out <RUN_ROOT>
python -m experiments.mlst2026.successor_20260913.t22s estimate --protocol $PROTO --out <RUN_ROOT>
```

The final reduce takes no `--allow-partial`; it refuses missing, extra,
duplicated, unattributed or tampered cells (exit 3), any scientific-invariant
violation (exit 4) and any confirmation cell scored under another freeze
(exit 3).  Return `<RUN_ROOT>` whole: `run_matrix.jsonl`,
`run_matrix_summary.json`, `protocol_snapshot.json`, `EXECUTION_LEDGER.jsonl`,
`LEDGER_SUMMARY.json`, `FROZEN_CALIBRATION.json`, `cells/`, `REDUCED.json`,
`RESOURCE_ESTIMATE.json`, plus the stdout/stderr logs of every command.

## Execution mode: serial fallback preserved

Run **one worker per ledger** (`<WORKERS>` = 1).  The ledger serialises appends
with an advisory `flock` on a sidecar lock file; that is adequate on a local
disk and **unverified on network filesystems** (NFS, Lustre, CephFS, SMB): a
lock that is not honoured across clients would let two workers charge the same
cell or interleave ledger lines.  Nothing in this round tested `flock` on such
a filesystem, and a local macOS or Linux test is not evidence about the remote
mount.  Until a remote check of the mount's lock semantics exists, do not split
one run root over several workers; the historical T22 runs were serial and that
fallback is the mode this document prescribes.  If more throughput is needed,
use separate run roots per worker only with a separate protocol amendment that
declares how their ledgers are merged; this document does not authorise it.

## What is NOT run remotely under this protocol

Part B (`matched_stage.py`) opens instrument data only through the archived
`t22.load_stretch`/`open_instrument_file` path and is exercised here on a
synthetic HDF5 fixture (`matched_fixture` cell, `--fixture` only).  A matched
instrument run needs, first: the completed and validated confirmation
calibration; second, the exposure inventory (`EXPOSURE_INVENTORY.md`) and the
release file list, which the archive does not contain; third, the feasibility
decision (`matched_stage.feasibility`) with the number of files never opened by
any archived stage.  If feasibility fails the correct output is the failure
record.  A matched rerun on already exposed data must be labelled accordingly
and is not independent evidence.  No such command is issued here.
