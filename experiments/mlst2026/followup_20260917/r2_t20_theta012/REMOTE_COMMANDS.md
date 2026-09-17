# R2 (T20-S v3 theta=0.12 scaling and crossover) — remote commands

Placeholders `<DATA_ROOT>`, `<RUN_ROOT>`, `<WORKERS>` are **unresolved** on
purpose: no host, scheduler or hardware is assumed here. `<DATA_ROOT>` is the
checkout of the experiment repository; `<RUN_ROOT>` must be outside `results/`
and `evidence/` (the runner refuses those). This package draws no external data.

Every command below is run from `<DATA_ROOT>` with

```
export PYTHONPATH=<DATA_ROOT>
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # timings were measured single-threaded
PKG="python -m experiments.mlst2026.followup_20260917.r2_t20_theta012"
PROTO=experiments/mlst2026/followup_20260917/r2_t20_theta012/PROTOCOL_R2_T20_THETA012_v1.json
```

## 0. Environment record and validation (computes nothing)

```
python -c "import numpy, scipy, sys; print(sys.version.split()[0], numpy.__version__, scipy.__version__)"   # compare with requirements.txt
$PKG validate  --protocol $PROTO
$PKG enumerate --protocol $PROTO --out <RUN_ROOT>
```

Expected from `validate`/`enumerate`: 966 cells = 3 algebraic + 3 control + 384
pilot + 576 confirmation; 4800 fits (5 per fit cell), 105 solves; cap 7000;
`matrix_sha256` as printed locally (record both in the handoff). If the counts
differ, stop: the protocol or code differs.

Optional four-geometry expansion (NOT required; enables `theta = 0.40` and prices
1287 cells / 6400 fits / 140 solves under the same cap):

```
R2_T20_INCLUDE_OPTIONAL=1 $PKG validate --protocol $PROTO
```

## 1. Algebraic + pilot phase (fresh pilot seeds 23000–23015)

```
$PKG run --protocol $PROTO --out <RUN_ROOT> --families algebraic
$PKG run --protocol $PROTO --out <RUN_ROOT> --families pilot
```

`run` is resumable and skips completed cells, so `<WORKERS>` may be started on
disjoint cell subsets with `--cells <id ...>` (ids from
`<RUN_ROOT>/run_matrix.jsonl`). All workers append to one `EXECUTION_LEDGER.jsonl`
with an advisory `flock`; this was verified on a local disk only. **Until the
remote filesystem's lock behaviour is checked, run one worker per ledger (serial
fallback)** or give each worker its own `--out` and reduce them separately.

## 2. Pilot reduce — writes FROZEN_MODEL.json (must precede any confirmation cell)

```
$PKG reduce --protocol $PROTO --out <RUN_ROOT> --allow-partial
```

`--allow-partial` is needed because the confirmation cells do not exist yet; the
package reducer still refuses (exit 3) unless **every** pilot and algebraic cell
is present, and it writes `<RUN_ROOT>/FROZEN_MODEL.json` exactly once.
`REDUCED.json` carries `phase: pilot`, `pilot_complete: true` and a
`partial_note`. Copy `FROZEN_MODEL.json` and `REDUCED.json` out now.

## 3. Confirmation phase (fresh seeds 24000–24023)

```
$PKG run --protocol $PROTO --out <RUN_ROOT> --families confirmation
```

The runner refuses this (exit 6, nothing charged) if `FROZEN_MODEL.json` is
missing or was frozen under a different protocol or matrix.

## 4. Confirmation reduce — scores both endpoints

```
$PKG reduce --protocol $PROTO --out <RUN_ROOT>
```

`REDUCED.json` then carries `phase: confirmation`, the two endpoints
(`confirmation.endpoint_a_old_prediction`, `confirmation.endpoint_b_new_model`),
the declared multiplicity rule and the `claim_supported` decision. It refuses
(exit 3) if the freeze is missing or the pilot digest changed, and (exit 4) if a
frozen prediction violates the offset convention.

## 5. Resource estimate

```
$PKG estimate --protocol $PROTO --out <RUN_ROOT>
```

No `BENCHMARK.json` is shipped with this protocol, so the estimate reports
`measured_projection_available: false`, marks every family `unmeasured`, and
computes only the flat cross-check block (which is labelled not-the-estimate).

## 6. Reproducibility

Every cell file records the code commit, source hashes, thread limits and the
protocol/matrix sha in its `provenance` block. Re-running `reduce` is
deterministic. A frozen `FROZEN_MODEL.json` is never overwritten; a conflicting
pilot reduce refuses (exit 3).
