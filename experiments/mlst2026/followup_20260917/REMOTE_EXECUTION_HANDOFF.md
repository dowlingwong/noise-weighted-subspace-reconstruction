# Follow-up round — remote execution handoff

Round directory: `experiments/mlst2026/followup_20260917/` (branch `codex/followup-20260917`).
All packages were designed **after** the September results were observed. No
production pilot or confirmation run has been executed.

## 1. Environment and preflight

Pins validated with the successor round: Python 3.12, `numpy==2.4.6`,
`scipy==1.17.1`, `pytest==9.1.0`, `h5py==3.16.0` (h5py only for R6's synthetic
fixtures). One BLAS thread. The runner imports the archived simulator/pipeline
modules under `experiments/mlst2026/successor_20260913/` and the archived
`experiments/mlst2026/t22_tidmad_narrowband.py`, `tidmad_endpoint.py`,
`tidmad_band.py`; those must be present and unchanged.

```sh
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=$PWD:$PWD/src
python -c "import numpy,scipy;print(numpy.__version__,scipy.__version__)"

# preflight: no command below draws a production cell.  validate/enumerate compute nothing.
for P in experiments/mlst2026/followup_20260917/r{1,2,3,4,5}_*/PROTOCOL_*.json; do
  pkg=$(python -c "import json,sys;print(json.load(open('$P'))['package'])")
  python -m experiments.mlst2026.followup_20260917.$pkg validate --protocol $P
done
```

Exit codes: 0 ok, 1 failure, 2 budget refusal, 3 reconciliation, 4 schema/science,
5 duplicate completion, 6 protocol/usage.

## 2. R0 — already complete, no remote compute

The correction is committed under
`r0_t22_h5/evidence_export/` and requires no remote run. If re-derivation is
wanted on the server:

```sh
python -m experiments.mlst2026.followup_20260917.r0_t22_h5 reduce \
  --run-root <archived t22s run root> --copy-to /var/tmp/r0_copy --out <export dir>
```

## 3. New-compute packages (run in this order)

**Execution-mode decision (A1):** every compute package is decided **serial**
(machine-readable in `EXECUTION_MODES.json` and each `EXECUTION_MODE.json`).
R1 is the borderline case — ~73 h serial versus the 72 h anchor (~3 days) — and is
decided **serial**: it does not materially exceed the reviewed budget, the
parallel alternative is unreviewed, and a failed locking claim would corrupt the
ledger. A parallel-safe ledger (old F8) that would cut this to ~5 h is a separate
reviewed engineering item, never a silent dependency. Do not parallelise a run
root unless the target filesystem's advisory-lock behaviour has been validated.

For every package the pattern is: **validate -> pilot -> reduce (freezes the
selection) -> confirmation -> reduce**. A confirmation cell refuses (exit 6)
without its freeze. Serial is the safe fallback.

### Preflight receipts to generate before any production run (A3, A4)

```sh
# R2 simulator-truth audit (production design)
python -m experiments.mlst2026.followup_20260917.r2_t20_theta012 truth-audit \
  --protocol experiments/mlst2026/followup_20260917/r2_t20_theta012/PROTOCOL_R2_T20_THETA012_v1.json \
  --out experiments/mlst2026/followup_20260917/r2_t20_theta012/TRUTH_AUDIT.json --production

# R3 ladder-preservation receipt at every rung (d0, 2d0, 4d0)
python -m experiments.mlst2026.followup_20260917.r3_t21_finite preservation-receipt \
  --protocol experiments/mlst2026/followup_20260917/r3_t21_finite/PROTOCOL_R3_T21_FINITE_v1.json \
  --out experiments/mlst2026/followup_20260917/r3_t21_finite/LADDER_PRESERVATION_RECEIPT.json --production

# R6 data-blocked inventory and input-gate refusal record
python -m experiments.mlst2026.followup_20260917.r6_instrument inventory \
  --out experiments/mlst2026/followup_20260917/r6_instrument/INPUT_READINESS_INVENTORY.md
python -m experiments.mlst2026.followup_20260917.r6_instrument check-inputs \
  > experiments/mlst2026/followup_20260917/r6_instrument/INPUT_GATE_RECEIPT.json   # exits 3
```

### R1 — T22 interval calibration (`r1_t22_calibration`)

```sh
P=experiments/mlst2026/followup_20260917/r1_t22_calibration/PROTOCOL_R1_T22_CALIBRATION_v2.json
RUN=/ceph/$USER/mlst-followup-20260917/r1_t22_calibration
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration run    --protocol $P --out $RUN --families sim_pilot control
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration reduce --protocol $P --out $RUN --allow-partial
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration run    --protocol $P --out $RUN --families sim_confirm
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration reduce --protocol $P --out $RUN
```

Stop after the pilot reduce if `FROZEN_CANDIDATE.json` chooses `"none"`; report
the calibration failure and do not run confirmation.

After confirmation, write R1's acceptance receipt (consumed by the R4 gate):

```sh
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration acceptance --run-root $RUN
# writes $RUN/R1_ACCEPTANCE.json with per-file-count coverage_ok / type1_ok
```

For R4, point the gate at that receipt:

```sh
export R4_R1_FREEZE=$RUN/FROZEN_CANDIDATE.json
export R4_R1_ACCEPTANCE=$RUN/R1_ACCEPTANCE.json
python -m experiments.mlst2026.followup_20260917.r4_t22_feasibility gate \
  --protocol experiments/mlst2026/followup_20260917/r4_t22_feasibility/PROTOCOL_R4_T22_FEASIBILITY_v2.json
```

The gate never re-scores R1; it only decides admissibility of R4's large-file
cells. On a cancel it emits an `R4_INFEASIBLE` receipt and the 32/64 cells are
omitted from R4's matrix, launch plan and cost report.

### R2 — T20 theta=0.12 (`r2_t20_theta012`)

See `r2_t20_theta012/REMOTE_COMMANDS.md`. Required matrix 966 cells; theta=0.40 is
an optional costed expansion (`R2_T20_INCLUDE_OPTIONAL=1`) not required for the
claimed scope.

### R3 — T21 finite size (`r3_t21_finite`)

See `r3_t21_finite/REMOTE_COMMANDS.md`. Run the pilot, freeze any alternative
edge band, then confirmation. If the orthogonal prerequisite fails, the loadings
step stays untested and the report says so.

### R4 — T22 feasibility (`r4_t22_feasibility`)

```sh
P=experiments/mlst2026/followup_20260917/r4_t22_feasibility/PROTOCOL_R4_T22_FEASIBILITY_v2.json
RUN=/ceph/$USER/mlst-followup-20260917/r4_t22_feasibility
python -m experiments.mlst2026.followup_20260917.r4_t22_feasibility run    --protocol $P --out $RUN --families eval_grid dev_grid control
python -m experiments.mlst2026.followup_20260917.r4_t22_feasibility reduce --protocol $P --out $RUN
```

Inspect `feasibility_by_n_files[*].screened_infeasible_by_withholding_bound`. If
true, **do not** run additional large-file production; report infeasibility.

### R5 — T19 reachability (optional) (`r5_t19_reach`)

See `r5_t19_reach/REMOTE_COMMANDS.md`. Price one cell before allocating the
multi-million-sample grid. `"not reachable"` is a valid result.

### R6 — instrument readiness (data-blocked)

No execution. `r6_instrument/` exposes `validate`, `enumerate`, `preflight`,
`inventory`, `check-inputs`; `check-inputs` and `preflight` exit 3 with a
`DATA-BLOCKED` status when real inputs are absent. Do not claim any real-data
adapter is validated until it is validated on real inputs.

## 4. Dependencies and cautions

- R0/R1/R4 import the archived T22-S pipeline; R2 imports the archived T20
  generator/analysis; R3 imports the archived T21 geometry/predictor; R5 imports
  the archived T19 synthetic/observables. None edits them.
- The ledger uses an advisory `flock`; the archived runs showed it is certified
  on local disk only. Keep a serial fallback. Failed cells are charged and are
  retried only with `--allow-repeat`, never by editing a freeze or a matrix.
- Frozen selections (`FROZEN_CANDIDATE.json`, `FROZEN_MODEL.json`,
  `FROZEN_EDGE.json`) are digest-checked before confirmation and must never be
  regenerated to hide a mismatch.
- A pilot-selected choice is never re-selected on confirmation data. A fixed
  design (R4, R5) has no selection authority.

## Round-2 correction addendum (17 September 2026)

- Use the **v2 protocols** (`PROTOCOL_R1_T22_CALIBRATION_v2.json`,
  `PROTOCOL_R4_T22_FEASIBILITY_v2.json`, `PROTOCOL_R5_T19_REACH_v2.json`); the
  v1 files are retained unchanged for the record and are **refused at
  validate/run** (R1/R4 by content binding, R5 by an explicit superseded-version
  check). A command that names a v1 protocol fails closed with exit 6 and must
  not be edited to bypass the refusal.
- R0 must be run with a **fresh `--copy-to`** destination outside the input
  trees; the CLI refuses in-place runs except the test-only `--allow-in-place`.
- R4 will **refuse to start production** while the R1 precondition is
  unresolved or blocked, and will omit the n=32/64 cells unless R1's acceptance
  receipt is a complete, explicit, hash-consistent pass. Do not edit a freeze or
  an acceptance receipt to bypass this.
- Each R4 record now names the applied inference version and the R1 receipt
  hash; a run that only tests the archived baseline says so.
- The full local suite (`132 passed`) is the pre-transfer gate; any failure
  refuses the transfer. `PREFLIGHT.json` is the hash-bound preflight record.
