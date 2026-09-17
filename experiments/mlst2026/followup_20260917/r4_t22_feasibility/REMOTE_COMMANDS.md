# R4 remote execution commands

Dependency: `numpy==2.4.6`, `pytest==9.1.0`. One BLAS thread. Reuses the archived
T22-S simulator/pipeline from `experiments/mlst2026/successor_20260913/t22s/`.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=$PWD:$PWD/src
P=experiments/mlst2026/followup_20260917/r4_t22_feasibility/PROTOCOL_R4_T22_FEASIBILITY_v2.json
RUN=/ceph/$USER/mlst-followup-20260917/r4_t22_feasibility

python -m experiments.mlst2026.followup_20260917.r4_t22_feasibility validate --protocol $P

# Screening first: run module (a) through n_files=16 and module (b), then decide
# whether the n_files=32/64 cells can be omitted.
python -m experiments.mlst2026.followup_20260917.r4_t22_feasibility run    --protocol $P --out $RUN --families eval_grid dev_grid control
python -m experiments.mlst2026.followup_20260917.r4_t22_feasibility reduce --protocol $P --out $RUN

# If feasibility_by_n_files shows screened_infeasible_by_withholding_bound=true,
# stop; do NOT run additional large-file production.
python -m experiments.mlst2026.followup_20260917.r4_t22_feasibility estimate --protocol $P --out $RUN
```

Serial is the safe fallback. Failed cells are charged and retried only with
`--allow-repeat`. A gate change or a development-dimension change requires a new
protocol version and fresh confirmation.
