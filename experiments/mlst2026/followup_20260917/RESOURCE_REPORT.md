# Follow-up round — measured resource report

Generated 2026-09-17 from the actual implementations and local development
kernels. **No production cell was run.** Every number is labelled measured or
projected; the planning anchors in the follow-up proposals (72 h T22, 67 h R4,
2–8 h T20, ≤7 h T21, 3–6 h T19) are shown beside the re-estimate.

Environment: macOS arm64, Python 3.12.7, numpy 2.4.6, scipy 1.17.1, one BLAS
thread. The server's core count, per-core speed, filesystem locking and memory
are unknown; wall-time rows are conditional.

## Measured development kernels (this round)

| kernel | measurement | value |
|---|---|---|
| R0 overlay reduce | 690 stored confirmation cells | 9.5 s wall, 690 cells |
| R1 fixture sim cell | n_files=4, mc=20, file=40, hier=200, curve=30 | 1.06 s |
| R1 fixture sim cell | n_files=16, same fixture allocation | 2.21 s |
| R4 fixture sim cell | n_files=4, mc=6 | 0.42 s |
| R4 fixture sim cell | n_files=16, mc=6 | 1.12 s |
| R5 fixture reach cell | d=32, m=8192, 3 seeds | 0.007 s |
| R1 candidate augmentation | 5 candidates x up to 4 contrasts, per replicate | < 1 ms (negligible beside the bootstrap) |
| R4 separate-stream wrapper | per replicate vs archived single stream | identical model, no measurable overhead |

R3 measured its own kernels into `r3_t21_finite/BENCHMARK.json`: d=256 hi/lo
0.012/0.031 s, d=512 0.056/0.154 s, d=1024 0.328/0.830 s per fit; projected cell
compute 816.9 s for 3456 fits.

R2 measured its own kernels (see `r2_t20_theta012/README.md`): a production fit
cell at d=256, R=4, N=400/800/1600 takes 1.74/3.46/6.73 s (linear, ≈4.2 ms/N).

## Production projections

The archived measurement (`MEASURED_CELL_TIMES.txt`) is the anchor for the three
packages that reuse the T22-S simulator/pipeline with the same allocation:
T22-S confirmation cells are 203 s (n_files=4), 322 s (8) and 561 s (16) per
100 MC replicates. R0/R1/R4 run the same pipeline, so those per-cell times are
reused rather than re-derived from a smaller fixture.

| package | cells | per-cell basis | serial compute | old anchor |
|---|---|---|---|---|
| R0 | 0 new | read-only overlay, 9.5 s total | 0 | minutes |
| R1 confirmation | 690 | 203/322/561 s at n4/8/16 | **~70.7 h** | 72 h |
| R1 pilot | 32 | same | ~2.7 h | (in 72 h) |
| **R1 total** | 722 | | **~73 h serial** | 72 h |
| R4 module (a) eval grid | 320 | 102→1000 s at n4→64 (50 MC) | ~34.6 h | — |
| R4 module (b) dev grid | 240 | 280 s at n16 (50 MC) | ~18.7 h | — |
| **R4 total** | 560 | | **~53 h serial** (25 h removed if n=32/64 screened out) | 67 h (incomplete) |
| R2 required matrix | 966 | measured fits, see package | ~3.9 CPU-h | 2–8 h |
| R3 required matrix | 3456 fits | BENCHMARK, d=256→1024 | ~0.23 h cell compute | ≤7 h |
| R5 optional | 4 cells x 20 seeds | m up to 4e6, d=256 | projected ~0.2–2 h, unmeasured | 3–6 h |
| R6 | 0 executable | data-blocked | 0 | 7 CPU-min (unvalidated) |

### R4 estimate is now larger and split, as the instruction requires

The old 67 h proposal covered only evaluation-file extensions. R4 now contains a
distinct development-data gate question; module (b) adds ~19 h. The exact
withholding bound lets the screening step remove the n=32/64 extension (~25 h)
from the launch plan **before** it is run, which is the point of the package.

### R5 must be priced from a pilot cell

The 10.2-million-sample grid with d=256 cannot be priced from a d=32 fixture.
Memory is bounded (one 256x256 accumulator, 512 KiB, plus one 16384x256 block,
32 MiB) but the sample-streaming time must be measured on one real cell before
any allocation. The 3–6 h anchor is retained as a planning bound only.

## Setup, transfer, inference and storage

- **Setup**: create the pinned environment (numpy/scipy/pytest), validate each
  protocol (`validate` computes nothing), and run the preflight. No preflight or
  validation command draws a production cell.
- **Transfer**: the new package tree plus the unchanged
  `successor_20260913/t22s`, `t19s`, `t20s`, `t21s` modules it imports. Small
  (<1 MB) apart from R0's 0.5 MB report.
- **Inference replicates**: R1 69,000 confirmation replicates; R4 46,000;
  R2/R3/R5 as declared in their protocols.
- **Storage**: R1 ~0.9 GB of cell files at production (690 cells x ~1.3 MB,
  extrapolated from the archived t22s root); R4 larger in cell count but smaller
  per cell; R0 export 1.4 MB.
- **Conditional parallel wall time**: R1 is ~73 h serial; at 16 workers ~5 h
  *if* the target filesystem's ledger locking is validated (the archived runs
  showed advisory `flock` is certified on local disk only). Serial is the safe
  fallback and is sufficient for the budget.

## Execution-mode decisions (serial vs sharded)

Machine-readable in `EXECUTION_MODES.json` and each package's `EXECUTION_MODE.json`.
The round rule is **serial** unless a package's serial wall time exceeds its
reviewed budget **and** a bounded shard/merge design meets all five conditions
(disjoint cell ownership, immutable receipts, duplicate/missing detection, every
charged attempt preserved, target-filesystem locking validated). None meets them,
and no CephFS-safe ledger rewrite is implemented or silently depended on.

| package | serial estimate | anchor | decision | note |
|---|---|---|---|---|
| R1 | ~73 h (68–80) | 72 h | **serial** | borderline; justified in the package file |
| R2 | ~3.9 h (2–8) | 5 h (2–8) | **serial** | small required matrix |
| R3 | ~0.35 h (0.23–1) | ≤7 h | **serial** | targeted ladder |
| R4 | ~53 h (28 after screening) | 67 h | **serial** | screening removes ~25 h; cancellation-gated by R1 |
| R5 | ~1 h (0.2–2) projected | 3–6 h | **serial** | optional; price a pilot cell first |

A parallel-safe ledger that would turn R1's ~3 days into ~5 h remains a separate
reviewed engineering item (old F8); it is never a dependency of a science package.

## R4 cancellation effect on the cost report (A2)

R4's `n_files = 32/64` cells (~25 h at the measured trend) are included in the
~53 h total **only if** the R1 precondition passes. If R1 freezes `"none"`, fails
its coverage/type-I acceptance, or the withholding screen is universal, the gate
omits those cells and the R4 cost report is ~28 h. The cells are then cancelled,
not deferred, and are not staged as ready to transfer.

## What would change these numbers

- The server's per-core speed relative to the benchmark host.
- Whether concurrent workers contend for memory bandwidth / disk queue.
- Whether R4's screening removes the large-file cells.
- A measured R5 pilot cell, which is required before that allocation is priced.

## Round-2 correction note (17 September 2026)

The executable counts changed only where the review required it:

- **R4** production cells are now gate-dependent: **443** while R1 is
  unresolved/blocked (n=4/8/16 only), **563** on a clean R1 pass (adds the six
  n=32/64 scenarios). The conditional large-file cost in this report applies
  only to the 563-cell case; a blocked/unresolved launch must not price them.
- **R5** production is now one cell per seed group over the full m ladder (4
  cells: 1 reach + 3 controls) with exact accounting of 80 fits, so the priced
  matrix matches the drawn matrix.
- **R1/R2/R3/R6** counts are unchanged in substance; exact figures and hashes
  are recorded in `PREFLIGHT.json` (R1 726/0, R2 966/4800, R3 3460/3456, R6
  readiness only).
