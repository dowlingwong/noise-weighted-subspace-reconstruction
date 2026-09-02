#!/bin/bash
# P1-E3 — local (CPU) rerun of the E12 calibration-size sweep.
# Uses the frozen runner unchanged; the anchor N_cal=2048 must reproduce E10b to 1e-9.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
PY=${PY:-$HOME/venv/bin/python}
OUT=${OUT:-results/paper1/P1-E3}
COMMIT=$(git rev-parse HEAD)
PYTHONPATH=src:experiments/synthetic $PY -m pytest -q experiments/synthetic/nfpa_confirmatory/tests/test_round12_sweep.py
PYTHONPATH=src:experiments/synthetic $PY -m nfpa_confirmatory.run_calibration_sweep --out "$OUT" --implementation-freeze-commit "$COMMIT" "$@"
