#!/bin/bash
set -euo pipefail
: "${ENL_CHECKOUT:?Frozen checkout required}"
: "${ENL_CONFIRMATION_AUTHORISED:?Exact confirmation authorization required}"
: "${ENL_OUTPUT:?Fresh output required}"
test "$ENL_CONFIRMATION_AUTHORISED" = yes
cd "$ENL_CHECKOUT"
unset PYTHONPYCACHEPREFIX
export PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
test ! -L experiments && test ! -L src
if find experiments src -type l -print -quit | read -r linked; then exit 75; fi
if find experiments src -name '*.pyc' -print -quit | read -r cached; then exit 75; fi
ulimit -v 8388608
ulimit -t 259200
exec timeout --kill-after=60 259140 /ceph/dwong/x3_env/bin/python -m experiments.mlst2026.enl_20261008.coordinator --out "$ENL_OUTPUT"
