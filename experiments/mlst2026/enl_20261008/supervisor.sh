#!/bin/bash
set -euo pipefail
: "${ENL_CONFIRMATION_AUTHORISED:?Exact once-only confirmation authorization required}"
test "$ENL_CONFIRMATION_AUTHORISED" = yes
: "${ENL_CHECKOUT:?Frozen checkout required}"
: "${ENL_FREEZE:?Exact candidate freeze required}"
: "${ENL_ACCEPTANCE:?Registered independent acceptance required}"
: "${ENL_OUTPUT:?Fresh external output required}"
registry=/ceph/dwong/enl_runs/registry
mkdir -p "$registry"
exec 9>"$registry/paper1-enl-v1-confirm.lock"
flock -n 9 || exit 73
test ! -e "$registry/paper1-enl-v1-confirm.attempt.json"
test ! -e "$ENL_OUTPUT"
cd "$ENL_CHECKOUT"
unset PYTHONPYCACHEPREFIX
export PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if find experiments src -name '*.pyc' -print -quit | read -r cached; then exit 75; fi
ulimit -v 8388608
ulimit -t 259200
started=$(date +%s)
set +e
timeout --kill-after=60 252000 /ceph/dwong/x3_env/bin/python -m experiments.mlst2026.enl_20261008.workflow run --mode confirmation --out "$ENL_OUTPUT" --freeze "$ENL_FREEZE" --acceptance "$ENL_ACCEPTANCE" --ledger "$registry/paper1-enl-v1-confirm.attempt.json"
rc=$?
set -e
printf '{"exit_code": %s}\n' "$rc" > "${ENL_OUTPUT}.supervisor_exit.json"
if test -e "$ENL_OUTPUT/ATTEMPT.json"; then
 remaining=$((259200 - $(date +%s) + started))
 test "$remaining" -gt 0 || exit 124
 timeout --kill-after=60 "$remaining" /ceph/dwong/x3_env/bin/python -m experiments.mlst2026.enl_20261008.workflow seal --out "$ENL_OUTPUT" --exit-code "$rc"
fi
exit "$rc"
