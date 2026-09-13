# Local-build round — 13 September 2026 (L1–L5)

Governing documents (manuscript repository, MS):
`plan/MLST_COMPLETENESS_AND_LOCAL_BUILD_2026-09-13.md` (the amendment) and
`plan/MLST_UPGRADE_INSTRUCTIONS_2026-09-13.md`.

**Status vocabulary — use exactly these words in every report and manifest:**
`configured` → `locally tested` → `ready to transfer` → `executed` →
`results validated` → `claim supported`.  A package in this directory is at
most *ready to transfer*.  Nothing here is executed evidence, and no fixture
outcome may be quoted as a scientific result.

## Layout

```
successor_20260913/
  README.md                this file: conventions binding on every package
  common/                  shared protocol/matrix/ledger/results/CLI code
  t19s/                    L1  T19-S  covariance and observable validation
  t20s/                    L2  T20-S  geometry, bounds, learner contrasts
  t21s/                    L3  T21-S  independent geometry and excitation
  t22s/                    L4  T22-S  inference calibration + matched interface
  cresst_s/                L5  CRESST cluster-aware reporting (archive-only)
```

Historical freezes (`../successor_20260912/`, `results/`, `evidence/`) are
**immutable**.  New protocols are new files with a `supersedes`/`provenance`
pointer to the design input they correct; the historical files are not edited.

## Every package must contain

| item | file | rule |
|---|---|---|
| protocol | `PROTOCOL_<PKG>_v2.json` | schema `mlst-local-package/1` (see `common/protocol.py`); hypotheses, independent unit, estimands, controls, seed namespaces, inference/multiplicity rule, **fixed** stopping allocation, failure interpretation, budget |
| run matrix | produced by `enumerate` | one row per cell with role, seed, params, `n_fits` (estimator invocations on drawn data) and `n_solves` (algebraic/engineered solves that fit nothing); counts are derived mechanically from the matrix, never from "seeds × arms" in prose |
| runner | `runner.py` with `main()` built on `common.cli.main_template` | `validate`/`enumerate`/`--help` compute nothing; `run` needs an explicit `--out` outside `results/` and `evidence/`; `--fixture` uses the development seed namespace only and marks outputs `development_only` |
| ledger | `EXECUTION_LEDGER.jsonl` under the run root | written by `common.ledger`; every attempt charged at start; failures/retries/interrupts never refunded |
| reducer | `reduce` subcommand | deterministic (fixed bootstrap seeds, sorted keys); refuses missing/extra/duplicate/unattributed/tampered cells (exit 3) unless `--allow-partial`, which labels the output partial |
| result schema | `result_required` dict in the runner + `SCHEMA.md` | every per-cell field named with units and the formula it implements |
| tests | `tests/test_successor_20260913_<pkg>.py` | see "Required tests" |
| dependencies | `requirements.txt` (pins actually used locally) | numpy/scipy versions observed in the local run |
| resource estimate | `run_benchmarks.py` → `BENCHMARK.json`; `estimate` subcommand → `RESOURCE_ESTIMATE.json`; `make_resource_report.py` → `RESOURCE_REPORT.md` | **measured, scaling-aware**: the package's own dominant kernel timed at ≥ 4 sizes bracketing or reaching the production dimension (or at every production value of an axis that has fewer than four), fitted as `t = a n^b` and printed beside the algorithm's exponent, × the production matrix. Estimator calls, restarts, replicates, **the per-cell cost of the `common/cli.py` harness itself** (protocol load, charged ledger append under `flock`, schema and fit checks, atomic result write with fsync and sha256) and the reduce step are priced, and the harness is measured end to end by running fixture cells through the real CLI; peak memory, output storage, thread settings, transfer size and setup time are reported separately; wall time is a **range** at 1/8/32 workers as `wall = (cell_compute + parallel_harness)/W + serial_ledger + reduction`, with the ledger term measured because the appends serialise under one lock, and marked conditional because the server is unknown. The old flat fixture × declared-scale product survives only as a labelled cross-check, and it is compared with the **total**, not with kernel time alone. |
| commands | `REMOTE_COMMANDS.md` | exact commands for enumerate → run (pilot) → reduce → run (confirmation) → reduce, with resume behaviour; placeholders `<DATA_ROOT>`, `<RUN_ROOT>`, `<WORKERS>` left **unresolved** (never invent hosts, schedulers or hardware) |
| findings placeholder | `FINDINGS_<PKG>.md` | states what the package will and will not be able to establish; contains no outcome |

## Seeds

Three disjoint namespaces, declared in the protocol: `development`
(fixtures and smoke; **never** citable), `pilot` (design/selection decisions),
`confirmation` (scored against the frozen rule).  A fixture run is refused
outside the development namespace.  Pilot-selected choices (a scaling model, a
predictor, a band) are frozen into a small JSON written by the pilot reducer
before any confirmation cell is drawn, and the confirmation reducer reads it.

## Accounting

* one *fit* = one estimator invocation on drawn data (each restart counts;
  each learner counts; each rank counts);
* one *solve* = one eigen/least-squares solve on engineered or analytic
  matrices that fits no drawn data (counted, costed, not capped);
* inference replicates (bootstrap draws) are reported separately as
  `replicates`;
* the cap in the protocol is on fits; the matrix must fit under it with the
  declared retry headroom;
* fit counts are **observed, not restated**: each package instruments its own
  estimator call sites through `common/fitcount.py`, and `common.cli` refuses a
  cell (exit 4, charged) unless the count declared in the matrix, the count the
  cell reports and the count the instrumentation saw all agree.  Every cell file
  records the observed breakdown as `fits_observed`;
* a resource projection covers the production matrix with `development` cells
  excluded, and the estimate records that set as `priced_counts`; the handoff
  builder refuses to ship an estimate whose priced set differs from the
  enumerated production counts printed beside it.

## Required tests (per package, with the real reduced runner and synthetic fixtures)

1. `validate`/`enumerate` succeed and create no `cells/`;
2. a fixture end-to-end run (`run --fixture` → `reduce`) completes on a
   small size and the reduced output has the declared schema;
3. scientific controls behave: the package's named controls (zero
   perturbation, orthogonal geometry, all-zero signal, null simulation, …)
   produce the values the design says they must, within stated tolerance;
4. pilot and confirmation namespaces are disjoint in the enumerated matrix,
   and a fixture run refuses pilot/confirmation cells;
5. reduce refuses a partial run, a duplicated cell file, and a tampered cell
   file, each with the documented exit code;
6. an injected failure (`--fail-cell`) leaves a charged `failed` attempt and
   the retry accumulates;
7. **mutation**: an injected incorrect scientific output (wrong sign, wrong
   observable, wrong order, broken invariant) is rejected by the package's
   own scientific checks — not merely by the digest check;
8. a resource estimate can be produced from the fixture timings.

Keep fixture sizes small (each package's full test file under ~90 s on two
cores).  Do not run anything resembling production under a "smoke" label.

## What the packages are not

They are preparation.  "Ready for remote execution" means local acceptance
passed.  It does not mean any scientific claim is supported; the historical
verdicts (T19 slope prediction not met; T20 `c/N` refuted, `N*` bracketed not
located; T21 5/5 predictions not met; T22 verdicts withheld) stand until a
remote production run returns, is validated and is integrated.
