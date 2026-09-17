# R4 — T22 feasibility without an unjustified evaluation-file escalation

Status: configured, locally tested on development fixtures; **ready to transfer**
for the screening run. Large-file production is conditional on the screening.

## Question

Which evaluation-file counts and development-data conditions jointly meet
coverage, power and withholding requirements — and does the development gate make
the criterion unreachable regardless of evaluation-file count?

## Design (new protocol)

- `PROTOCOL_R4_T22_FEASIBILITY_v1.json`, package `r4_t22_feasibility`, supersedes
  T22-S v2. Fixed design, **no selection authority, no pilot freeze**.
- Module (a): evaluation-file grid n_files = 4/8/16/32/64 at a fixed development
  design (n_dev = 4, n_cal = 64).
- Module (b): development-data adequacy over n_dev in {4, 8, 16} x n_cal in
  {64, 256} at a fixed evaluation design (n_files = 16).
- Separate development and evaluation random streams (`streams.py`) so changing
  one dimension does not alter the other's realisation. Recorded source change.
- Exact screening bound `P(supported) <= 1 - P(withheld)` with Wilson uncertainty:
  a design whose withholding cannot fall below 0.5 is screened infeasible and its
  large-file cells are omitted from the launch plan.
- Original feasibility requirements unchanged; conditional power is not a
  substitute for unconditional power.
- Seed namespaces: development [923000, 923999], pilot [27000, 27299],
  confirmation [28000, 28499].

## Commands

```sh
P=experiments/mlst2026/followup_20260917/r4_t22_feasibility/PROTOCOL_R4_T22_FEASIBILITY_v1.json
# local fixture
python -m experiments.mlst2026.followup_20260917.r4_t22_feasibility run    --protocol $P --out $OUT --fixture --families eval_grid dev_grid control
python -m experiments.mlst2026.followup_20260917.r4_t22_feasibility reduce --protocol $P --out $OUT
# production
python -m experiments.mlst2026.followup_20260917.r4_t22_feasibility run    --protocol $P --out $RUN --families eval_grid dev_grid control
python -m experiments.mlst2026.followup_20260917.r4_t22_feasibility reduce --protocol $P --out $RUN
```

## Tests

`tests/test_followup_20260917_r4.py` (7 tests): validate/enumerate, separate
streams, support-bound arithmetic, the withholding screen, fixture end-to-end for
both modules, and the no-selection-authority declaration.

## Limits and interpretation gate

No production run has been executed. A feasibility failure is the correct output
where it occurs. Changing the gate or a development dimension requires a new
protocol version with full type-I/coverage validation and fresh confirmation; it
is never forced in a scientific run. Nothing here is adequacy on real instrument
data.

## A2 — R1-failure → R4-cancellation gate

`r1_gate.py` evaluates a machine-checkable precondition before R4's large-file
cells (n_files = 32/64) are planned or priced. It reads R1's
`FROZEN_CANDIDATE.json` and `R1_ACCEPTANCE.json` (written by R1's `acceptance`
subcommand) and returns `pass`, `cancel`, or `unresolved`:

```sh
python -m experiments.mlst2026.followup_20260917.r4_t22_feasibility gate \
  --protocol PROTOCOL_R4_T22_FEASIBILITY_v1.json \
  --freeze $R1_RUN/FROZEN_CANDIDATE.json --acceptance $R1_RUN/R1_ACCEPTANCE.json
```

A cancel **omits** the 32/64 cells from the matrix, launch plan and cost report
and emits an `R4_INFEASIBLE` receipt naming the trigger; it is not a deferral.
R4 is not ready to transfer while the gate is unresolved or cancelled. See
`../../EXECUTION_MODES.json` for the serial execution-mode decision.
