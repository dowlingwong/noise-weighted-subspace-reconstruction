# X2 (T25-NS): restriction × calibration under nonseparable noise

Protocol: MS `plan/T25NS_PROTOCOL_2026-10-05.md`. Launch kit: MS
`reviews/production_20261005/launch/` (`t25ns_host.sh`, README). Nothing here runs on import.

- `guard.py` — fail-closed check that every reused T25 file is byte-identical to the accepted
  T25 closure (`t25_accepted_sources.json`, identity `01f2f82f…`).
- `protocol.py`, `production.json`, `fixture.json` — the closed configurations.
- `science.py` — Σ_ρ, Van Loan–Pitsianis distance, estimator population limits, data (common
  random numbers across ρ), the T25 fit per cell, contrasts (31 per location + 12 cross-ρ = 384), population-limit
  clean-risk penalties,
  registered prediction verdicts.
- `workflow.py` — freeze/gates, supervised single attempt with in-campaign optimizer replay,
  reduction, export with tables, `EXPORT.json` and `SHA256SUMS`, portable audit (checks the export
  record and regenerates the tables), host qualification.
- `tests/` — unit, design and end-to-end fixture tests (39; qualification requires exactly 39 passed).

```sh
export PYTHONPATH=$PWD/src:$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m experiments.mlst2026.t25ns_20261005 identity          # X2 code identity
python -m pytest -q experiments/mlst2026/t25ns_20261005/tests   # fixtures
python -m experiments.mlst2026.t25ns_20261005 fixture --out /tmp/x2_fixture   # development lifecycle
```

Production runs only through the host launcher after independent acceptance.
Dependencies: numpy, scipy, psutil, pytest, threadpoolctl (the T25 environment).
