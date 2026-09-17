# R1 remote execution commands

Dependency: `numpy==2.4.6`, `scipy==1.17.1`, `pytest==9.1.0`. One BLAS thread.
The package reuses the archived T22-S simulator and pipeline from
`experiments/mlst2026/successor_20260913/t22s/`; the repository must be at the
round's commit with that tree present and unchanged.

```sh
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=$PWD:$PWD/src

P=experiments/mlst2026/followup_20260917/r1_t22_calibration/PROTOCOL_R1_T22_CALIBRATION_v2.json
RUN=/ceph/$USER/mlst-followup-20260917/r1_t22_calibration

# 0. preflight (no compute)
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration validate --protocol $P

# 1. pilot -> freeze (serial; the freeze must exist before confirmation)
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration run    --protocol $P --out $RUN --families sim_pilot control
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration reduce --protocol $P --out $RUN --allow-partial
# inspect $RUN/FROZEN_CANDIDATE.json: chosen_candidate must be a candidate or "none"

# 2. confirmation (only if the freeze exists; refuses otherwise)
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration run    --protocol $P --out $RUN --families sim_confirm
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration reduce --protocol $P --out $RUN

# 3. resource projection from the run root
python -m experiments.mlst2026.followup_20260917.r1_t22_calibration estimate --protocol $P --out $RUN
```

Do **not** parallelise the run root unless the target filesystem's ledger locking
has been validated; serial is the safe fallback. Failed cells are charged and are
retried only with an explicit `--allow-repeat`. The frozen candidate is never
reselectable on confirmation data.
