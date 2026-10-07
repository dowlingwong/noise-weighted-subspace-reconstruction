# TR09 preparation and guarded confirmation candidate

The accepted T25 dependency closure is checked before import and after execution.
The candidate protocol lives in the manuscript repo at `plan/TR09_PROTOCOL_2026-10-06.md`.
This package adds sparse eigenbasis latents, paired streams, oracle-only endpoints,
4 primary and 36 secondary contrasts, exact saved-map risk reevaluation and receipt audit.
The primary contrasts intentionally also appear in the secondary family.

Run from the EXP root with `PYTHONPATH=src` and OPENBLAS_NUM_THREADS,
OMP_NUM_THREADS and MKL_NUM_THREADS each set to 1. Fixture and benchmark modes use
separate shipped configurations. Output must be fresh. The benchmark is development
information only. No confirmation seed is drawn by tests or fixtures.

```
python -m experiments.mlst2026.tr09_20261007.workflow run --mode fixture --out /external/fresh_fixture
python -m experiments.mlst2026.tr09_20261007.workflow audit --out /external/received_fixture
python -m pytest experiments/mlst2026/tr09_20261007/tests -q
```

The audit is read-only unless its internal caller explicitly asks for a report.
It checks receipt SHA/census, snapshot bytes, current source identity, config/cell
census, deterministic training provenance, basis/intercept/restart factor maps,
objectives, rank and class geometry, exact risks, unrestricted solution and reductions.
Use the returned source snapshot with the bound Python/numpy/scipy environment for
portable audit; never patch a returned archive to match current source.

Confirmation requires an exact source/config/environment candidate freeze, separate
independent ACCEPTED registration binding the freeze SHA and namespace, explicit
TR09_CONFIRMATION_AUTHORISED=yes and an exclusive durable namespace ledger path.
A fixed supervised shell entrypoint provides timeout, flock and attempt ledger.
The registered ledger is never removed or relocated; a failed/incomplete run blocks
retry in that namespace. Correction after registration requires a new namespace.
The shell entrypoint, registration and reviewer records are part of the launch kit.

Intervals are approximate seed-mean Student-t intervals. Zero-SE is flagged; no
population certainty follows. Optimization-limited cells remain visible. No crossing,
nonlinear-model or second-order mechanism conclusion is produced.
