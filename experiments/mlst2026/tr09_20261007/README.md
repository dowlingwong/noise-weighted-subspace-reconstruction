# TR09 preparation and guarded confirmation candidate

The accepted T25 dependency closure is checked before import and after execution.
The candidate protocol lives in the manuscript repo at `plan/TR09_PROTOCOL_2026-10-06.md`.
This package adds sparse eigenbasis latents, paired streams, oracle-only endpoints,
4 primary and 36 secondary contrasts, exact saved-map risk reevaluation and receipt audit.
The primary contrasts intentionally also appear in the secondary family.

Run from the EXP root with `PYTHONPATH=src` and OPENBLAS_NUM_THREADS,
OMP_NUM_THREADS and MKL_NUM_THREADS each set to 1,
PYTHONDONTWRITEBYTECODE=1, and PYTHONPYCACHEPREFIX unset. Use a fresh checkout
without .pyc under experiments or src; the supervisor refuses caches before import. Fixture and benchmark modes use
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
For portable audit, enter the returned `source_snapshot` root and set
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src` before importing the package. Its exact
source set must match START.json. The receipt stores each raw training pool once
per seed/theta/p, at max N; all smaller N are prefixes. Training SHA verifies the
saved bytes, while regenerated records/moments/maps use declared tolerances so
BLAS rounding across audit hosts does not defeat portability. The confirmation
launch itself requires the exact frozen executable/distribution/BLAS identity.
Never patch a returned archive to match current source.

Confirmation requires an exact source/config/environment candidate freeze, separate
independent ACCEPTED registration binding the freeze SHA and namespace, explicit
TR09_CONFIRMATION_AUTHORISED=yes and an exclusive durable namespace ledger path.
The source-bound `supervisor.sh` provides timeout, flock and attempt ledger.
Invoke that exact absolute package path with bash; direct Python invocation is
refused without the registered Linux timeout/shell ancestry. The wrapper seals
its return code into the archive after the Python attempt exits. Per-cell alarms,
campaign elapsed-time checks and payload charging reserve half the 64 GiB
scratch cap for export; the wrapper applies an 8 GiB virtual-memory cap.
The registered ledger is never removed or relocated; a failed/incomplete run blocks
retry in that namespace. Correction after registration requires a new namespace.
The shell entrypoint, registration and reviewer records are part of the launch kit.

Intervals are approximate seed-mean Student-t intervals. Zero-SE is flagged; no
population certainty follows. Optimization-limited cells remain visible. No crossing,
nonlinear-model or second-order mechanism conclusion is produced.
