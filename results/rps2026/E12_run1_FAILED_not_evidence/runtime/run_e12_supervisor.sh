#!/bin/bash
# E12 resume supervisor.
# Takes over AFTER the original chain finishes (complete or halted) and resumes
# the sweep as many times as needed, then runs Steps 7-8.
# Never deletes a shard. Never passes --sizes. Never changes thread settings,
# nice level, or any scientific setting. Stops before Step 9 (commit/push).
cd /ceph/dwong/noise-weighted-subspace-reconstruction
export PYTHONPATH=src:experiments/synthetic
export R10_RUNTIME_ROOT="$RUNTIME"
unset NFPA_CONFIRMATORY_FAST
PY=.venv/bin/python
FREEZE_COMMIT=0c5e3095aa3510367ad96efc4bcba60372d6e9a7
CONTENT_COMMIT=75cbc40324293b8e7dfea65c01b0414027908997
OUT="$RUNTIME/E12_calibration_size_sweep"
MAX_ATTEMPTS=12

say() { echo "[$(date -Is)] $*" >> "$RUNTIME/supervisor.log"; }
fail() { say "SUPERVISOR_HALTED: $*"; echo "$*" > "$RUNTIME/supervisor_halted.txt"; exit 1; }

echo $$ > "$RUNTIME/supervisor.pid"
say "supervisor started, pid $$; waiting for the original chain to finish"

# Wait for the original chain to reach a terminal state, so we never race it.
while [ ! -f "$RUNTIME/chain_complete.txt" ] && [ ! -f "$RUNTIME/chain_halted.txt" ]; do
  sleep 120
done

if [ -f "$RUNTIME/chain_complete.txt" ]; then
  say "original chain completed everything; supervisor has nothing to do"
  echo done > "$RUNTIME/supervisor_complete.txt"; rm -f "$RUNTIME/supervisor.pid"; exit 0
fi
say "original chain halted: $(cat "$RUNTIME/chain_halted.txt" 2>/dev/null)"

sweep_complete() {
  [ -f "$OUT/sweep_summary.json" ] || return 1
  RUNTIME="$RUNTIME" $PY -c "
import json,os,sys
sm=json.load(open(os.environ['RUNTIME']+'/E12_calibration_size_sweep/sweep_manifest.json'))
sys.exit(0 if sm.get('status')=='complete' else 1)" 2>/dev/null
}

for attempt in $(seq 1 $MAX_ATTEMPTS); do
  if sweep_complete; then say "sweep already complete"; break; fi

  # never start a second runner alongside a live one
  while pgrep -f "run_calibration_sweep --implementation-freeze-commit" >/dev/null 2>&1; do
    say "a runner is still live; waiting"
    sleep 300
  done

  # the runner's gate is literally bool(git status --porcelain)
  git status --porcelain > "$RUNTIME/tree_state_resume_${attempt}.txt" 2>&1
  [ -s "$RUNTIME/tree_state_resume_${attempt}.txt" ] && fail "worktree dirty before resume $attempt: $(cat "$RUNTIME/tree_state_resume_${attempt}.txt")"
  [ "$(git rev-parse HEAD)" = "$CONTENT_COMMIT" ] || fail "HEAD moved off CONTENT_COMMIT before resume $attempt"

  say "resume attempt $attempt starting"
  START=$(date +%s)
  timeout 12h nice -n 10 $PY -m nfpa_confirmatory.run_calibration_sweep \
    --implementation-freeze-commit "$FREEZE_COMMIT" \
    --out "$OUT" --resume \
    > "$RUNTIME/e12_resume_${attempt}.log" 2>&1
  ST=$?
  ELAPSED=$(( $(date +%s) - START ))
  echo "RESUME_${attempt}_STATUS=$ST elapsed=${ELAPSED}s" >> "$RUNTIME/sweep_status.txt"
  say "resume attempt $attempt exit=$ST elapsed=${ELAPSED}s"

  if [ "$ST" -eq 0 ]; then
    say "runner exited 0"; break
  elif [ "$ST" -eq 124 ]; then
    say "hit the 12h cap; shards are registered, resuming again"
    continue
  else
    # Not a timeout: a refusal or a crash. That is a finding, not something to work around.
    fail "resume attempt $attempt exited $ST after ${ELAPSED}s (not a timeout). Refusal/crash is the finding; no shard deleted. See e12_resume_${attempt}.log"
  fi
done

if ! sweep_complete; then
  fail "sweep still not complete after $MAX_ATTEMPTS resume attempts; shards preserved"
fi
say "sweep complete"

# --- Step 7: the two gates ---
RUNTIME="$RUNTIME" $PY - > "$RUNTIME/gates.txt" 2>&1 <<'PY'
import json, pathlib, os
root = pathlib.Path(os.environ["RUNTIME"]) / "E12_calibration_size_sweep"
sm = json.load(open(root / "sweep_manifest.json"))
print("status:", sm["status"])
print("control gate:", json.dumps(sm["control_gate"], indent=1))
print("summary audit:", json.dumps(sm["summary_audit"], indent=1))
anchor = json.load(open(root / "n_cal_2048" / "manifest.json"))["anchor_reproduction"]
print("anchor reproduction:", json.dumps(anchor, indent=1))
assert sm["status"] == "complete"
assert sm["control_gate"]["passed"] is True
assert anchor["passed"] is True and anchor["comparisons"] == 20
print("GATES_OK")
PY
GATE_STATUS=$?
echo "GATE_STATUS=$GATE_STATUS" > "$RUNTIME/gate_status.txt"
say "Step 7 gates exit=$GATE_STATUS"

# --- Step 8: archive audit after (always runs, so tampering is always recorded) ---
RUNTIME="$RUNTIME" $PY -c "
from nfpa_confirmatory import r10_validation as V
import json, os
json.dump(V.audit_authoritative_archive(), open(os.environ['RUNTIME']+'/archive_after.json','w'), indent=1)"
RUNTIME="$RUNTIME" $PY -c "
import json, os
r=os.environ['RUNTIME']
a=json.load(open(r+'/archive_before.json'))['digests']
b=json.load(open(r+'/archive_after.json'))['digests']
print('archive unchanged:', a==b)
assert a==b" > "$RUNTIME/archive_compare.txt" 2>&1
echo "ARCHIVE_COMPARE_STATUS=$?" >> "$RUNTIME/archive_compare.txt"
git status --porcelain > "$RUNTIME/tree_state_post_sweep.txt" 2>&1
say "Step 8 done: $(tr '\n' ' ' < "$RUNTIME/archive_compare.txt")"

[ "$GATE_STATUS" -eq 0 ] || fail "GATES DID NOT PASS; evidence preserved, Step 9 not attempted"
say "SUPERVISOR_COMPLETE: steps 6-8 done. Step 9 left for a supervised session."
echo done > "$RUNTIME/supervisor_complete.txt"
rm -f "$RUNTIME/supervisor.pid"
