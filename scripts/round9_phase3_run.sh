#!/usr/bin/env bash
# Round 9 / Phase 3 - unattended CPU execution sequence.
#
# Order is the runbook's: E8 reproduction gate, E8 primary, E8 separable, E9, E10.
# An experiment-specific failure does not stop the independent experiments: E9 and
# E10 do not depend on the semantic code, so they run even if E8 fails. Every
# command gets a hard timeout and its return code is recorded.
#
# No scientific budget is ever reduced here. The caps below are wall-clock guards
# taken from the runbook, justified by the archived one-seed timing of ~226 s of
# fit time per seed across the five methods.

set -u
cd "$(dirname "$0")/.."
CODE_ROOT="$(pwd -P)"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PYTHON="$CODE_ROOT/.venv/bin/python"
R="$CODE_ROOT/results/rps2026"
LOGDIR="$R/Round9_remote/logs"
mkdir -p "$LOGDIR"

RC_FILE="$R/Round9_remote/phase3_returncodes.txt"
: > "$RC_FILE"

run_step () {
  local name="$1"; local cap="$2"; shift 2
  local log="$LOGDIR/${name}.log"
  echo "=== [$name] start $(date -u -Iseconds) cap=${cap} ===" | tee -a "$log"
  local t0=$SECONDS
  timeout "$cap" nice -n 10 "$PYTHON" "$@" >>"$log" 2>&1
  local rc=$?
  local dt=$(( SECONDS - t0 ))
  echo "=== [$name] end $(date -u -Iseconds) rc=${rc} wall_seconds=${dt} ===" | tee -a "$log"
  printf '%s\trc=%d\twall_seconds=%d\n' "$name" "$rc" "$dt" >> "$RC_FILE"
  return $rc
}

M=experiments.synthetic.nfpa_confirmatory

# ---------------------------------------------------------------- E8
E8_OK=1
run_step E8_gate_seed0 30m -m $M.run_semantics \
  --condition primary --seeds 0 --out-dir "$R/E8_semantics" --gate-only || E8_OK=0

if [ "$E8_OK" = "1" ]; then
  run_step E8_primary 2h -m $M.run_semantics \
    --condition primary --out-dir "$R/E8_semantics" || E8_OK=0
else
  echo "E8_primary SKIPPED: one-seed reproduction gate failed" | tee -a "$LOGDIR/E8_primary.log"
  printf 'E8_primary\trc=SKIPPED_GATE\twall_seconds=0\n' >> "$RC_FILE"
fi

if [ "$E8_OK" = "1" ]; then
  run_step E8_separable 2h -m $M.run_semantics \
    --condition separable --out-dir "$R/E8_semantics" || E8_OK=0
else
  echo "E8_separable SKIPPED: upstream E8 failure" | tee -a "$LOGDIR/E8_separable.log"
  printf 'E8_separable\trc=SKIPPED_UPSTREAM\twall_seconds=0\n' >> "$RC_FILE"
fi

# --------------------------------------- E9 and E10 are independent of E8
run_step E9_replication 4h -m $M.run_replication \
  --out-dir "$R/E9_blockA_replication"

run_step E10_estimated_covariance 3h -m $M.run_estimated_covariance \
  --out-dir "$R/E10_estimated_covariance"

echo "=== PHASE 3 COMPLETE $(date -u -Iseconds) ==="
cat "$RC_FILE"
