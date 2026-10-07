#!/bin/bash
# Candidate launch entrypoint. Review + registration precede any invocation.
set -euo pipefail
: "${TR09_CONFIRMATION_AUTHORISED:?Explicit authorisation required}"
test "$TR09_CONFIRMATION_AUTHORISED" = yes
: "${TR09_CHECKOUT:?Frozen isolated checkout required}"
: "${TR09_FREEZE:?Reviewed freeze.json required}"
: "${TR09_ACCEPTANCE:?Independent registered acceptance.json required}"
: "${TR09_OUTPUT:?Fresh external output required}"
registry=/ceph/dwong/tr09_runs/registry
mkdir -p "$registry"
exec 9>"$registry/paper1-tr09-v1-confirm.lock"
flock -n 9 || exit 73
test ! -e "$registry/paper1-tr09-v1-confirm.attempt.json"
test ! -e "$TR09_OUTPUT"
cd "$TR09_CHECKOUT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src
ulimit -v 8388608
set +e
timeout --kill-after=60 28800 /ceph/dwong/x3_env/bin/python -m experiments.mlst2026.tr09_20261007.workflow run --mode confirmation --out "$TR09_OUTPUT" --freeze "$TR09_FREEZE" --acceptance "$TR09_ACCEPTANCE" --ledger "$registry/paper1-tr09-v1-confirm.attempt.json"
rc=$?
set -e
printf '{"exit_code": %s}\n' "$rc" > "${TR09_OUTPUT}.supervisor_exit.json"
if test -e "$TR09_OUTPUT/ATTEMPT.json"; then
  /ceph/dwong/x3_env/bin/python -m experiments.mlst2026.tr09_20261007.workflow seal --out "$TR09_OUTPUT" --exit-code "$rc"
fi
exit "$rc"
