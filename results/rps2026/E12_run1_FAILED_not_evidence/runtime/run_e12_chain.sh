#!/bin/bash
# E12 unattended compute chain: Steps 5-8 of the runbook.
# Waits for the already-running Step 4 suite, then smoke, sweep, gates, archive audit.
# Deliberately STOPS before Step 9 (worktree copy, commit, push) - nothing
# outward-facing happens unattended.
cd /ceph/dwong/noise-weighted-subspace-reconstruction
export PYTHONPATH=src:experiments/synthetic
export R10_RUNTIME_ROOT="$RUNTIME"
unset NFPA_CONFIRMATORY_FAST
PY=.venv/bin/python
FREEZE_COMMIT=0c5e3095aa3510367ad96efc4bcba60372d6e9a7
CONTENT_COMMIT=75cbc40324293b8e7dfea65c01b0414027908997

say() { echo "[$(date -Is)] $*" >> "$RUNTIME/chain.log"; }
fail() { say "CHAIN_HALTED: $*"; echo "$*" > "$RUNTIME/chain_halted.txt"; exit 1; }

say "chain started, pid $$"
echo $$ > "$RUNTIME/chain.pid"

# --- wait for Step 4 (already running in screen e12_pytest) ---
say "waiting for Step 4 suite"
while [ ! -f "$RUNTIME/pytest_status.txt" ]; do
  sleep 30
  if ! pgrep -f 'pytest -q' >/dev/null 2>&1 && [ ! -f "$RUNTIME/pytest_status.txt" ]; then
    sleep 30
    [ -f "$RUNTIME/pytest_status.txt" ] || fail "pytest vanished without writing a status file"
  fi
done
PYTEST_STATUS=$(sed 's/.*=//' "$RUNTIME/pytest_status.txt")
say "Step 4 suite exit status: $PYTEST_STATUS"
[ "$PYTEST_STATUS" -eq 0 ] || fail "complete repository test suite FAILED (exit $PYTEST_STATUS); sweep not started"

# --- tree must still be clean; the runner gate is literally bool(git status --porcelain) ---
git status --porcelain > "$RUNTIME/tree_state_pre_sweep.txt" 2>&1
[ -s "$RUNTIME/tree_state_pre_sweep.txt" ] && fail "worktree dirty before sweep: $(cat "$RUNTIME/tree_state_pre_sweep.txt")"
[ "$(git rev-parse HEAD)" = "$CONTENT_COMMIT" ] || fail "HEAD moved off CONTENT_COMMIT"
say "tree clean, HEAD pinned"

# --- Step 5: smoke, out of tree ---
say "Step 5 smoke starting"
timeout 60m nice -n 10 $PY -m nfpa_confirmatory.run_calibration_sweep \
  --smoke --sizes 64 128 --out "$RUNTIME/Round10_smoke/E12" \
  > "$RUNTIME/e12_smoke.log" 2>&1
SMOKE_STATUS=$?
echo "SMOKE_STATUS=$SMOKE_STATUS" > "$RUNTIME/smoke_status.txt"
say "Step 5 smoke exit status: $SMOKE_STATUS"
[ "$SMOKE_STATUS" -eq 0 ] || fail "smoke FAILED (exit $SMOKE_STATUS)"

# --- Step 6: the sweep. No --sizes. Freeze commit required. ---
say "Step 6 sweep starting (attempt 1)"
date -Is > "$RUNTIME/sweep_start.txt"
timeout 12h nice -n 10 $PY -m nfpa_confirmatory.run_calibration_sweep \
  --implementation-freeze-commit "$FREEZE_COMMIT" \
  --out "$RUNTIME/E12_calibration_size_sweep" \
  > "$RUNTIME/e12.log" 2>&1
E12_STATUS=$?
echo "E12_STATUS_ATTEMPT1=$E12_STATUS" > "$RUNTIME/sweep_status.txt"
say "Step 6 sweep attempt 1 exit status: $E12_STATUS"

# At most ONE resume attempt, logged separately so nothing is hidden.
# Never deletes a shard. If resume refuses, that refusal is the finding.
if [ "$E12_STATUS" -ne 0 ]; then
  say "attempt 1 failed; trying exactly one --resume (no shard deleted)"
  timeout 12h nice -n 10 $PY -m nfpa_confirmatory.run_calibration_sweep \
    --implementation-freeze-commit "$FREEZE_COMMIT" \
    --out "$RUNTIME/E12_calibration_size_sweep" --resume \
    > "$RUNTIME/e12_resume.log" 2>&1
  E12_STATUS=$?
  echo "E12_STATUS_RESUME=$E12_STATUS" >> "$RUNTIME/sweep_status.txt"
  say "Step 6 resume exit status: $E12_STATUS"
fi
date -Is > "$RUNTIME/sweep_end.txt"
[ "$E12_STATUS" -eq 0 ] || fail "sweep FAILED (final exit $E12_STATUS); see e12.log / e12_resume.log"

# --- Step 7: verify the two gates fired ---
say "Step 7 gate verification"
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
say "Step 7 gate exit status: $GATE_STATUS"

# --- Step 8: archive audit AFTER (runs regardless, so tampering is always recorded) ---
say "Step 8 archive audit after"
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
say "Step 8 done: $(cat "$RUNTIME/archive_compare.txt" | tr '\n' ' ')"

git status --porcelain > "$RUNTIME/tree_state_post_sweep.txt" 2>&1
say "post-sweep tree state bytes: $(stat -c%s "$RUNTIME/tree_state_post_sweep.txt")"

[ "$GATE_STATUS" -eq 0 ] || fail "GATES DID NOT PASS; evidence preserved, Step 9 not attempted"

say "CHAIN_COMPLETE: steps 4-8 done. Step 9 (copy in, report, commit, push) deliberately left for a supervised step."
echo done > "$RUNTIME/chain_complete.txt"
rm -f "$RUNTIME/chain.pid"
