#!/bin/bash
# Step 4 driver: complete repository suite. Exit status captured to a file,
# never piped through tee (a pipeline would return tee's status).
cd /ceph/dwong/noise-weighted-subspace-reconstruction
export PYTHONPATH=src:experiments/synthetic
unset NFPA_CONFIRMATORY_FAST
echo $$ > "$RUNTIME/pytest.pid"
date -Is > "$RUNTIME/pytest_start.txt"
timeout 90m nice -n 10 .venv/bin/python -m pytest -q > "$RUNTIME/pytest_full.log" 2>&1
PYTEST_STATUS=$?
echo "PYTEST_STATUS=$PYTEST_STATUS" > "$RUNTIME/pytest_status.txt"
date -Is > "$RUNTIME/pytest_end.txt"
rm -f "$RUNTIME/pytest.pid"
