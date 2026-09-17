# R5 remote execution commands (optional)

Dependency: `numpy==2.4.6`, `pytest==9.1.0`. One BLAS thread. Reuses the archived
T19-S `synthetic.py` and `observables.py` unchanged.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=$PWD:$PWD/src
P=experiments/mlst2026/followup_20260917/r5_t19_reach/PROTOCOL_R5_T19_REACH_v2.json
RUN=/ceph/$USER/mlst-followup-20260917/r5_t19_reach

python -m experiments.mlst2026.followup_20260917.r5_t19_reach validate --protocol $P

# Price one cell first, then run the fixed grid.
python -m experiments.mlst2026.followup_20260917.r5_t19_reach run    --protocol $P --out $RUN
python -m experiments.mlst2026.followup_20260917.r5_t19_reach reduce --protocol $P --out $RUN
```

Memory note: each cell holds one `d x d` accumulator (d = 256 -> 512 KiB) plus one
`chunk x d` block; the sample count `m` is never materialised. `"not reachable"`
is a valid outcome and is not a reason to enlarge `m` after the fact.
