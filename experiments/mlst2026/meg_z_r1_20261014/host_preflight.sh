#!/usr/bin/env bash
set -eu
cd /ceph/dwong/megz_r1_host_20261014
. /ceph/dwong/meg_env/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src TMPDIR=/tmp
unset PYTHONPYCACHEPREFIX
exec python -m experiments.mlst2026.meg_z_r1_20261014.runner preflight \
  --derived /ceph/dwong/meg_z/derived_timingfix_20261013 \
  --gates experiments/mlst2026/meg_z_20261013_dev_timingfix/gates.json \
  --out experiments/mlst2026/meg_z_r1_20261014_r2bfix/PREFLIGHT.json
