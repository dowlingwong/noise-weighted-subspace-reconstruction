# X3: Σ⁻¹ versus isotropic loss for a nonlinear class

Protocol: MS `plan/X3_NONLINEAR_METRIC_PROTOCOL_2026-10-05.md`. Launch kit: MS
`reviews/production_20261005/launch/` (`x3_host.sh`, README). Nothing here runs on import.

**Pinned dependency: torch == 2.9.1, CPU build** (checked at start-up; any other version refuses
to run). Install into the same interpreter as numpy/scipy/psutil/pytest/threadpoolctl:

```sh
# Linux (Deepthought): CPU wheel, avoids the multi-GB CUDA dependencies
$PY -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
# macOS (Apple silicon)
$PY -m pip install torch==2.9.1
```

- `guard.py`, `t25_accepted_sources.json` — the same accepted-T25 byte check as X2 (copied so
  X3 does not depend on the X2 package).
- `models.py` — `FlatAE`, Hutchinson divergence and SURE, ported from VERITAS
  `src/whitebox_subjects/{ladder,geometry}.py` (MIT, same author).
- `science.py` — data (T25 generator, oracle Σ, θ = 0.25), closed-form tied linear AE, frozen
  training recipe, evaluation (Σ⁻¹ and Euclidean clean risk, SURE), 54 contrasts, P1/P2 verdicts.
- `workflow.py` — the X2 process design (freeze, single supervised attempt, in-campaign
  re-evaluation plus training replay of the first seed, export, portable audit, qualification).
- `tests/` — 20 tests including the end-to-end fixture lifecycle and tamper checks.

```sh
export PYTHONPATH=$PWD/src:$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m experiments.mlst2026.x3_nonlinear_metric_20261005 torch-check
python -m pytest -q experiments/mlst2026/x3_nonlinear_metric_20261005/tests
```
