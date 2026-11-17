#!/bin/bash
set -euo pipefail
: "${ENL_CONFIRMATION_AUTHORISED:?Exact once-only confirmation authorization required}"
test "$ENL_CONFIRMATION_AUTHORISED" = yes
: "${ENL_CHECKOUT:?Frozen checkout required}"
: "${ENL_FREEZE:?Exact candidate freeze required}"
: "${ENL_ACCEPTANCE:?Registered independent acceptance required}"
: "${ENL_OUTPUT:?Fresh external output required}"
: "${ENL_CAMPAIGN_STARTED_MONOTONIC:?Shared campaign clock required}"
: "${ENL_POST_SCIENCE_RECORD:?Shared post-science deadline record required}"
registry=/ceph/dwong/enl_runs/registry
mkdir -p "$registry"
exec 9>"$registry/paper1-enl-v1-confirm.lock"
flock -n 9 || exit 73
test ! -e "$registry/paper1-enl-v1-confirm.attempt.json"
test ! -e "$ENL_OUTPUT"
cd "$ENL_CHECKOUT"
unset PYTHONPYCACHEPREFIX
export PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
test ! -L experiments && test ! -L src
if find experiments src -type l -print -quit | read -r linked; then exit 75; fi
if find experiments src -name '*.pyc' -print -quit | read -r cached; then exit 75; fi
ulimit -v 8388608
ulimit -t 259200
exec /ceph/dwong/x3_env/bin/python -B -m experiments.mlst2026.enl_20261008.supervise
