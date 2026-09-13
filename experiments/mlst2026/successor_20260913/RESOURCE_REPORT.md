# Resource report — MLST successor round (local build, 2026-09-13)

Benchmarks generated 2026-09-13T11:09:05+00:00; rendered by `experiments/mlst2026/successor_20260913/make_resource_report.py` from the `BENCHMARK.json` and `RESOURCE_ESTIMATE.json` of each package. Every number below — the tables and the prose — is read out of those files, and `make_resource_report.py --check` fails if this file is not what they currently produce.

## What this replaces

The previous `RESOURCE_ESTIMATE.json` files multiplied one measured *fixture* wall time per family by a constant declared in the protocol (`budget.fixture_to_production_scale`). That is a linear extrapolation of a tiny fixture with no account of the production dimension and no account of the algorithmic scaling of the kernel. For t21s it overstated the package by a factor of about 60.2: 4891.29 CPU-s declared against 81.2218 CPU-s measured for everything a production run of that package costs. (Against the *kernel* time alone, 68.99 CPU-s, the ratio is 70.9; an earlier version of this report quoted that as the correction, which compared a whole-package estimate with a part of a package. The like-for-like number is the first one.)

This pass also fixed a defect on the measured side. `common/ledger.py` re-parsed the entire execution ledger twice inside every `start()`, so a run was quadratic in its own cell count: on this hardware `start` cost 1.2 ms at the first of 2 499 cells and 83 ms at the last, 69.6 s in total for one package's matrix, none of it in any estimate. The ledger now folds its state incrementally and re-reads only what other processes have appended; the same 2 499 cells cost 2.9 s. What remains is measured and priced below as **harness**, per package, and its serialised part is carried explicitly in the wall-time model.

What is here instead: each package's dominant kernel, identified by reading its runner, benchmarked **with the package's own functions** at four or more problem sizes that bracket or reach the production dimension (or at every production value of an axis that has fewer than four, which is the case only for `replicate_by_n_files`, whose axis is n_files in {4, 8, 16}); the scaling fitted as `t = a n^b` by least squares on the logs and printed beside the exponent the algorithm predicts; the per-cell seconds multiplied by the enumerated production matrix. The per-cell cost of the CLI harness around the kernel is measured end to end and priced beside it, and its serialised part appears again in the wall-time model. Peak memory, output storage, transfer size and setup time are measured and reported separately from all of that.

**Every number is labelled `measured` or `projected`.** A projected one names the fitted exponent it used and the extrapolation factor in the dimension. In this round the production dimension turned out to be reachable for every package, so almost everything below is a measurement; the one substitution (CRESST raw pulse traces, which are not in this container) is called out where it occurs and is labelled `measured-on-substitute`.

## Benchmark hardware

```
Intel(R) Xeon(R) Processor @ 2.10GHz, 2 vCPU (2 cores, 1 thread/core), 7 GiB RAM
Linux container; python 3.11.15, numpy 2.4.4, scipy-openblas 0.3.31.188.0
Benchmarks run single-threaded: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Summary

| package | cells | fits | solves | cell compute CPU-s | harness CPU-s | reduce CPU-s | total CPU-s (low - point - high) | wall h @1 | wall h @8 | wall h @32 | peak RSS/cell | output | flat cross-check CPU-s | flat/total |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `t19s` | 136 | 3840 | 3605 | 835.808 | 0.66 | 7.441 | 838.562 - **843.909** - 847.461 | 0.234 | 0.0311 | 0.00934 | 183 MB | 10.6 MB | 817.446 | 0.969 |
| `t20s` | 964 | 4800 | 140 | 6229.24 | 4.009 | 18.16 | 6213.9 - **6251.42** - 6277.26 | 1.74 | 0.222 | 0.0593 | 154 MB | 4.0 MB | 7329.77 | 1.17 |
| `t21s` | 2499 | 2496 | 2498 | 68.9921 | 8.604 | 3.626 | 80.2748 - **81.2218** - 81.6619 | 0.0226 | 0.00399 | 0.00201 | 60.5 MB | 15.7 MB | 4891.29 | 60.2 |
| `t22s` | 727 | 0 | 0 | 337533 | 16.81 | 4.022 | 335997 - **337554** - 341619 | 93.8 | 11.7 | 2.93 | 47.5 MB | 723.3 MB | 378823 | 1.12 |
| `cresst_s` | 395 | 1056 | 0 | 232.678 | 1.804 | 0.6112 | 210.809 - **235.093** - 272.784 | 0.0653 | 0.00835 | 0.00224 | 202 MB | 2.0 MB | 583.973 | 2.48 |
| **all five** | 4721 | 12192 | 6243 | | | | **344966** (= 95.82 CPU-hours) | 95.82 | 11.98 | 2.99 | | | | |

`total` is cell compute + the per-cell CLI harness + the reduction; the last column is the old flat estimate divided by that total, which is the like-for-like comparison (the flat product was an estimate of a whole package, so it must be compared with a whole package). The all-five wall-time cells divide the summed CPU seconds by the worker count and are therefore optimistic across packages twice over: they assume the five packages share one pool, and they drop each package's serial floor. Per package, use the rows above.

### The historical 6-8 hour allocation

This round's honest total is **95.8 CPU-hours**, of which 93.8 is T22-S alone and 2.06 is the other four packages combined. That is neither the historical 6-8 hour allocation nor anything near it, and the allocation was not used as an input anywhere in this document. At 8 workers the round is about 12 wall-hours; at 32 workers, about 3. Whether either worker count exists on the receiving machine is not known here. The number is dominated by one package: if T22-S's pilot freezes the cheaper replicate setting, the round roughly halves.

## Transfer and setup (separate from computation)

- **Transfer.** The handoff archive is **7.76 MB** (`/tmp/hf5/mlst-successor_20260913-handoff-20260913T111611Z.tar.gz`). The transfer *time* depends on the link, which is unknown; only the size is a fact. Successive builds of the same handoff differ by a few tens of kB (timestamps, this report and the estimate files).
- **`verify_handoff.py`** on the extracted archive: 3.59-3.64 s (manifest, environment, validate and enumerate for all five packages; computes no cells). MEASURED, 2 runs.
- **`verify_handoff.py --tests`**: 57.34-58.53 s. MEASURED, 2 runs. This is the receiving side's first real step and it is a bounded, one-off cost.
- **Environment.** The pinned requirements are `h5py==3.16.0`, `numpy==2.4.4`, `pytest==9.1.1`, `scipy==1.17.1`. This container **already satisfies them**, so no real `pip install` was timed - that would have measured a no-op. The honest figures are the import check, 0.139 s, and a `pip install --dry-run --no-deps` resolution, 0.51 s, both MEASURED. On a server that does *not* have them, the cost is the download and install of the numpy and scipy wheels (of order 90 MB) plus h5py and pytest; that is link- and index-dependent and is **unknown** from here.

## What remains unknown

- **Server core count.** Every wall-time row below is given at 1, 8 and 32 workers
  and is conditional on those cores existing. This round does not know how many
  the receiving machine has, and has not assumed a number.
- **Per-core speed relative to this Xeon.** All timings are on an Intel Xeon
  @ 2.10 GHz (see the hardware block). A machine 2x faster per core halves every
  CPU-second figure; a machine on older silicon does the reverse. No scaling
  factor has been applied.
- **BLAS threading policy on the server.** Everything here is single-threaded
  (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`). A multithreaded
  BLAS makes each cell faster and each worker wider, in opposite directions; the
  net effect on the wall time is not predictable from here. Whoever runs this
  should pin threads to 1 and parallelise over cells, which is what these numbers
  assume.
- **Filesystem speed and lock behaviour.** `reduce` reads every cell file once,
  and the execution ledger is appended under an advisory `flock`. The ledger
  append is measured and is carried in the wall-time model as an explicit serial
  term (see each package's wall-time section); on a network filesystem the
  advisory lock may not hold at all, and the safe fallback is one run root per
  worker, which this round cannot cost from here.
- **Contention between concurrent workers.** Every per-cell time in this document
  was measured on a 2-vCPU container running **one process at a time**. Nothing
  here contains cache pressure, memory-bandwidth sharing or disk-queue contention
  between workers. Running 8 or 32 cells concurrently makes each of them slower by
  an amount this benchmark cannot see and which grows with the worker count, so
  every 8- and 32-worker row below is optimistic by an unmeasured factor.
- **Link bandwidth for the transfer.** Only the archive size is a fact.
- **Whether the pinned requirements are already installed on the server.**
- **Memory available per worker.** The per-cell peak RSS figures below are
  measured; the server's memory budget is not known, so the maximum useful worker
  count from a memory standpoint cannot be computed here.

---

## T19-S — covariance-weighted PCA on the G1 generator

The production matrix is 136 cells: 120 **paired** cells and 16 deterministic
(algebraic/control) cells. A paired cell draws one training set at `d = 256`,
`N = 512`, runs 32 CW-PCA fits and 29 eigen solves, and — this is where the time
actually goes — streams a noise-only calibration pool of `m/d = 2048`, i.e.
`2048 x 256 = 524 288` rows of dimension 256, accumulating `Z^T Z` in 8192-row
chunks. The accumulation is `O(m d^2) = O(2048 d^3)`, so the calibration stream
alone is about two thousand times the cost of one `d x d` eigendecomposition.

### Benchmarks

**Kernel `deterministic_cell`** — `t19s.runner.run_deterministic_cell on the G1 population operator (8 eps points, one eigen solve each)`

> every d = 256 deterministic/control family runs this same code path with a different direction

| d | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 0.0075776 | 0.0074152 | 0.0083842 | 3 | - | - | - |
| 64 | 0.018628 | 0.018371 | 0.019615 | 3 | - | - | - |
| 128 | 0.067379 | 0.06692 | 0.068507 | 3 | - | - | - |
| 256 | 0.30453 | 0.30207 | 0.3053 | 3 | 7.21 MB | 55.1 MB | subprocess |

Scaling fit `t = a * d^b` (least squares on the logs): **a = 1.332e-05**, **b = 1.784**, residual (log RMS) 0.1533, worst |log residual| 0.1767, R² 0.9879 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **1.784** |
| theoretical (eigendecomposition of a d x d matrix is O(d^3)) | **3** |

**Disagreement:** the measured exponent is BELOW the algorithm's by 1.22: at these sizes the kernel is not yet flop-bound (Python/allocation overhead dominates), so a projection made with the fitted exponent UNDERSTATES the cost at larger sizes and one made with the theoretical exponent overstates it at these ones.

Production value: **measured** — measured at the production size.

**Kernel `deterministic_cell_2d`**

| n | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0.0018019 | 0.0017709 | 0.0018112 | 3 | - | - | - |

Production value: **measured** — measured at the production size.

**Kernel `paired_cell`** — `t19s.synthetic.run_paired_cell (32 cw_pca fits + 29 eigen solves per cell; the m/d = 2048 noise-only calibration stream of 2048*d rows dominates)`

> N is held at the production aspect ratio N = 2d so the d = 256 point IS one production paired cell

| d | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 0.094657 | 0.094078 | 0.095902 | 3 | - | - | - |
| 64 | 0.31722 | 0.31583 | 0.37846 | 3 | - | - | - |
| 128 | 1.3886 | 1.3772 | 1.3998 | 3 | - | - | - |
| 256 | 6.927 | 6.8827 | 6.9565 | 3 | 83.4 MB | 183 MB | subprocess |

Scaling fit `t = a * d^b` (least squares on the logs): **a = 6.498e-05**, **b = 2.071**, residual (log RMS) 0.1006, worst |log residual| 0.1199, R² 0.9961 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **2.071** |
| theoretical (eigendecomposition of a d x d matrix is O(d^3)) | **3** |

**Disagreement:** the measured exponent is BELOW the algorithm's by 0.93: at these sizes the kernel is not yet flop-bound (Python/allocation overhead dominates), so a projection made with the fitted exponent UNDERSTATES the cost at larger sizes and one made with the theoretical exponent overstates it at these ones.

Production value: **measured** — measured at the production size.

### Projection arithmetic

| family | cells | s/cell (median) | s/cell (min-max) | basis | fitted b | theory b | extrap. factor in dimension | cells x s/cell = CPU-s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| `ctrl_additive_generic` | 3 | 0.30453 | 0.30207 - 0.3053 | measured | 1.78 | 3 | 1 | 3 x 0.30453 = **0.913601** |
| `ctrl_additive_pure_coupling` | 3 | 0.30453 | 0.30207 - 0.3053 | measured | 1.78 | 3 | 1 | 3 x 0.30453 = **0.913601** |
| `ctrl_block_diagonal` | 2 | 0.30453 | 0.30207 - 0.3053 | measured | 1.78 | 3 | 1 | 2 x 0.30453 = **0.609068** |
| `ctrl_zero` | 1 | 0.30453 | 0.30207 - 0.3053 | measured | 1.78 | 3 | 1 | 1 x 0.30453 = **0.304534** |
| `det_generic` | 3 | 0.30453 | 0.30207 - 0.3053 | measured | 1.78 | 3 | 1 | 3 x 0.30453 = **0.913601** |
| `det_pure_coupling` | 3 | 0.30453 | 0.30207 - 0.3053 | measured | 1.78 | 3 | 1 | 3 x 0.30453 = **0.913601** |
| `det_two_d` | 1 | 0.0018019 | 0.0017709 - 0.0018112 | measured | - | - | - | 1 x 0.0018019 = **0.00180193** |
| `paired` | 120 | 6.927 | 6.8827 - 6.9565 | measured | 2.07 | 3 | 1 | 120 x 6.927 = **831.238** |
| **cell compute** | 136 | | | | | | | **835.808** (range 830.461 - 839.36) |
| **harness** (`common/cli.py` per cell) | 136 | 0.004853 | | measured end to end, priced per cell | | | | **0.66** |
| **reduction** | | | | measured rate, projected | | | | **7.441** |
| **total** | | | | | | | | **843.909** (range 838.562 - 847.461) |

Per-cell figures are the measured median over repeats where a family's cells are identical, and the mean over the family's own cells of their measured medians where they are not. The low/high columns are the corresponding measured extremes: they are measurements, not margins.

**Reduction work.** Measured on the fixture run root: 0.5724 s of reduce work over 12 cells = **0.0477 s per cell** (interpreter and import, 0.109 s, subtracted). Measured input throughput 0.54 MB/s against a warm page cache, so **at that size** (12 small fixture cells) the step is **compute-bound**: it reads every cell file once but spends its time re-deriving summaries and bootstrapping. That classification is a measurement of the fixture scale and is not asserted of the production scale; see the sentence after this table. Priced as 20 cells (pilot reduce) x 0.0477 s = 0.954 s + 136 cells (final reduce over the whole run root) x 0.0477 s = 6.487 s = **7.441 s**.

**Reduce at the production input size.** At the production scale reduce reads 10 MB in 7.44 s, i.e. 1.4 MB/s, still far below the 50 MB/s at which I/O would take over, so the compute-bound classification measured on the fixture also holds at production for this package.

**Harness work.** A production cell goes through `common/cli.py`, not through the kernel alone: ledger `start` under an advisory `flock`, the kernel, the schema and fit-accounting checks, the atomic result write (json encode + `fsync` + sha256) and ledger `complete`. Measured end to end by running 12 fixture cells through the real CLI (4.74 ms per cell), then re-priced at the production ledger length and the production cell-file size: **4.853 ms per cell**, 0.00485258 s x 136 cells = **0.66 CPU-s**. Of that, **0.04925 CPU-s** is ledger append, which every worker serialises through one lock; it appears again as the serial term in the wall-time model below. Per-worker interpreter and import start-up is 0.114 s, measured; it is paid once per worker **process** and is therefore reported here rather than added to the per-cell figure.

### Wall time

Cells are independent processes, so the cell compute and the parallel part of the per-cell harness are taken as perfectly parallel over W workers.  Two terms are NOT parallel and are added whole: the ledger appends, which every worker serialises through one advisory flock, and the reduction, which is a single process.  wall = (cell_compute + parallel_harness) / W + serial_ledger + reduction.  The serial ledger term is a floor no worker count beats.

Measured terms: cell compute 835.808 s (parallel), harness 0.6107 s (parallel), ledger append 0.04925 s (**serial**), reduce 7.441 s (**serial**). The serial floor is 7.49 s (0.002081 h) and no worker count goes below it.

| workers | wall hours (low - high) | serial floor is | status |
|---:|---|---|---|
| 1 | 0.2329 - 0.2354 | 0.9% of the point figure | CONDITIONAL on 1 usable cores; the server specification is not known |
| 8 | 0.03094 - 0.03125 | 6.7% of the point figure | CONDITIONAL on 8 usable cores; the server specification is not known |
| 32 | 0.009295 - 0.009372 | 22.3% of the point figure | CONDITIONAL on 32 usable cores; the server specification is not known |

### Peak memory and output storage

Peak memory per cell at the production dimension: **183 MB process RSS** (`paired_cell` at 256, measured in a subprocess), of which 83.4 MB is Python-allocator peak for the call itself. The RSS figure includes the interpreter and numpy, so it is what one worker actually needs. MEASURED.

| item | cells | bytes/cell | total | basis |
|---|---:|---:|---:|---|
| `ctrl_additive_generic` | 3 | 17,090 | 51.3 kB | measured |
| `ctrl_additive_pure_coupling` | 3 | 17,090 | 51.3 kB | measured |
| `ctrl_block_diagonal` | 2 | 17,090 | 34.2 kB | measured |
| `ctrl_zero` | 1 | 17,090 | 17.1 kB | measured |
| `det_generic` | 3 | 17,090 | 51.3 kB | measured |
| `det_pure_coupling` | 3 | 17,090 | 51.3 kB | measured |
| `det_two_d` | 1 | 11,946 | 11.9 kB | measured |
| `paired` | 120 | 81,506 | 9.78 MB | measured |
| execution ledger | 136 | | 258 kB | projected from a measured per-cell rate |
| `REDUCED.json` | | | 320 kB | projected linearly from the fixture |
| **total** | | | **10.6 MB** | |

### Cross-check against the protocol's flat scale factor

The old estimate (one fixture wall time x `budget.fixture_to_production_scale`) gives **817.446 CPU-s**. It estimated what a whole production run of the package costs, so the like-for-like comparison is against the measured **total**, **843.909 CPU-s** (cell compute 835.808 + harness 0.66 + reduce 7.441). Ratio measured total / flat = **1.032**; the declared flat scale is within a factor 2 of the measured total. Against cell compute alone the ratio would be 1.022, which is not the correction: it compares a whole-package estimate with part of a package.

### Findings

- **The production paired cell was run directly, so the paired number is a measurement.** `d = 256` with `N = 2d = 512` is the production configuration and it is one of the 4 benchmarked sizes. Nothing about the 120 paired cells is extrapolated; only the multiplication by 120 is arithmetic.
- **The fitted exponent (2.07) is well below the algorithmic one (3).** This is a statement about the *small* end of the benchmark, not about the production point. At `d = 32` the cell is 0.095 s of mostly Python and allocation; at `d = 256` it is 6.93 s of BLAS. A power law fitted across that range averages the two regimes. Because the production size was measured, the fit is used for description only - no projection uses it.
- **The protocol's own scale basis names the right dominant term.** `budget.scale_basis` already said the `m = 2048 d` calibration stream dominates, and the measurement agrees. The declared factor 200 for the paired family also lands close: the flat cross-check gives 817 CPU-s against a measured 836 CPU-s of cell compute, a ratio of 1.03. t19s is the one package whose declared flat scale was about right.
- **The CLI harness around the kernel costs 4.85 ms per cell, 0.66 CPU-s over the package, 0.078% of its total.** A production cell is not a kernel call. `common/cli.py` appends a charged `running` record to the execution ledger under an advisory `flock`, runs the kernel inside the fit-count instrumentation, checks the cell schema and the three fit counts, writes the result atomically (json encode + `fsync` + sha256) and appends the completion record. None of that is in the kernel benchmarks. It was measured end to end by running 12 fixture cells through the real CLI (4.74 ms per cell there) and re-priced at the production ledger length (136 records, 0.362 ms per cell) and the production cell-file size (73 889 bytes, 1.55 ms to write). Of the total, 0.0493 CPU-s is ledger append, which every worker serialises through one lock: that part is a floor on the wall time at any worker count. The end-to-end figure also carries the once-per-run costs amortised over 12 fixture cells, so it overstates rather than understates.

---

## T20-S — separable ALS against closed-form PCA

The production matrix is 964 cells: 960 **fit** cells and 4 algebraic cells.
A fit cell draws `N` rows at `d = C*T = 256`, runs one closed-form PCA (a single
`eigh` of a 256x256 second moment) and then `R = 4` restarts of the separable
ALS learner at up to 150 sweeps each. The ALS is the cost; the PCA is not.
`N` runs over the six-point production grid {400, 800, 1000, 1250, 1600, 3200},
and every `(seed, theta)` contributes exactly one cell at each `N`.

### Benchmarks

**Kernel `algebraic_cell`**

| n | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2.2099 | 2.2022 | 2.2437 | 3 | - | - | - |

Production value: **measured** — measured at the production size.

**Kernel `fit_cell`** — `t20s.runner.run_fit_cell -> np.linalg.eigh(S) (1 PCA fit) + experiments.synthetic.nfpa_confirmatory.methods.als_fit (R = 4 restarts x 150 sweeps)`

> d = C.T = 256 is fixed by the design; N is the only production size that varies, and every production N was run

| N | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 400 | 1.9768 | 1.9642 | 1.9798 | 3 | - | - | - |
| 800 | 3.7877 | 3.7616 | 3.8077 | 3 | - | - | - |
| 1000 | 4.795 | 4.7493 | 4.8298 | 3 | - | - | - |
| 1250 | 5.836 | 5.7959 | 5.8661 | 3 | - | - | - |
| 1600 | 7.4486 | 7.436 | 7.4495 | 3 | - | - | - |
| 3200 | 15.033 | 14.936 | 15.105 | 3 | 29 MB | 154 MB | subprocess |

Scaling fit `t = a * N^b` (least squares on the logs): **a = 0.005672**, **b = 0.975**, residual (log RMS) 0.0125, worst |log residual| 0.0158, R² 0.9996 over 6 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **0.975** |
| theoretical (an ALS sweep is O(iter . N . d . k), so linear in N at fixed d, k, iter) | **1** |

Agreement: the measured exponent agrees with the algorithm within +/- 0.35.

Production value: **measured** — measured at the production size.

### Projection arithmetic

| family | cells | s/cell (median) | s/cell (min-max) | basis | fitted b | theory b | extrap. factor in dimension | cells x s/cell = CPU-s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| `algebraic` | 4 | 2.2099 | 2.2022 - 2.2437 | measured | - | - | 1 | 4 x 2.2099 = **8.83961** |
| `confirmation` | 576 | 6.4796 | 6.4405 - 6.5064 | measured | 0.975 | 1 | 1 | 576 x 6.4796 = **3732.24** |
| `pilot` | 384 | 6.4796 | 6.4405 - 6.5064 | measured | 0.975 | 1 | 1 | 384 x 6.4796 = **2488.16** |
| **cell compute** | 964 | | | | | | | **6229.24** (range 6191.73 - 6255.09) |
| **harness** (`common/cli.py` per cell) | 964 | 0.004159 | | measured end to end, priced per cell | | | | **4.009** |
| **reduction** | | | | measured rate, projected | | | | **18.16** |
| **total** | | | | | | | | **6251.42** (range 6213.9 - 6277.26) |

Per-cell figures are the measured median over repeats where a family's cells are identical, and the mean over the family's own cells of their measured medians where they are not. The low/high columns are the corresponding measured extremes: they are measurements, not margins.

**Reduction work.** Measured on the fixture run root: 0.2425 s of reduce work over 18 cells = **0.01347 s per cell** (interpreter and import, 0.832 s, subtracted). Measured input throughput 0.31 MB/s against a warm page cache, so **at that size** (18 small fixture cells) the step is **compute-bound**: it reads every cell file once but spends its time re-deriving summaries and bootstrapping. That classification is a measurement of the fixture scale and is not asserted of the production scale; see the sentence after this table. Priced as 384 cells (pilot reduce) x 0.01347 s = 5.174 s + 964 cells (final reduce over the whole run root) x 0.01347 s = 12.99 s = **18.16 s**.

**Reduce at the production input size.** At the production scale reduce reads 1 MB in 18.2 s, i.e. 0.1 MB/s, still far below the 50 MB/s at which I/O would take over, so the compute-bound classification measured on the fixture also holds at production for this package.

**Harness work.** A production cell goes through `common/cli.py`, not through the kernel alone: ledger `start` under an advisory `flock`, the kernel, the schema and fit-accounting checks, the atomic result write (json encode + `fsync` + sha256) and ledger `complete`. Measured end to end by running 18 fixture cells through the real CLI (4.13 ms per cell), then re-priced at the production ledger length and the production cell-file size: **4.159 ms per cell**, 0.00415898 s x 964 cells = **4.009 CPU-s**. Of that, **0.5236 CPU-s** is ledger append, which every worker serialises through one lock; it appears again as the serial term in the wall-time model below. Per-worker interpreter and import start-up is 0.85 s, measured; it is paid once per worker **process** and is therefore reported here rather than added to the per-cell figure.

### Wall time

Cells are independent processes, so the cell compute and the parallel part of the per-cell harness are taken as perfectly parallel over W workers.  Two terms are NOT parallel and are added whole: the ledger appends, which every worker serialises through one advisory flock, and the reduction, which is a single process.  wall = (cell_compute + parallel_harness) / W + serial_ledger + reduction.  The serial ledger term is a floor no worker count beats.

Measured terms: cell compute 6229.24 s (parallel), harness 3.486 s (parallel), ledger append 0.5236 s (**serial**), reduce 18.16 s (**serial**). The serial floor is 18.69 s (0.005191 h) and no worker count goes below it.

| workers | wall hours (low - high) | serial floor is | status |
|---:|---|---|---|
| 1 | 1.726 - 1.744 | 0.3% of the point figure | CONDITIONAL on 1 usable cores; the server specification is not known |
| 8 | 0.2203 - 0.2225 | 2.3% of the point figure | CONDITIONAL on 8 usable cores; the server specification is not known |
| 32 | 0.05897 - 0.05952 | 8.8% of the point figure | CONDITIONAL on 32 usable cores; the server specification is not known |

### Peak memory and output storage

Peak memory per cell at the production dimension: **154 MB process RSS** (`fit_cell` at 3200, measured in a subprocess), of which 29 MB is Python-allocator peak for the call itself. The RSS figure includes the interpreter and numpy, so it is what one worker actually needs. MEASURED.

| item | cells | bytes/cell | total | basis |
|---|---:|---:|---:|---|
| `algebraic` | 4 | 1,835 | 7.34 kB | measured |
| `confirmation` | 576 | 1,409 | 812 kB | measured |
| `pilot` | 384 | 1,409 | 541 kB | measured |
| execution ledger | 964 | | 1.83 MB | projected from a measured per-cell rate |
| `REDUCED.json` | | | 827 kB | projected linearly from the fixture |
| **total** | | | **4.0 MB** | |

### Cross-check against the protocol's flat scale factor

The old estimate (one fixture wall time x `budget.fixture_to_production_scale`) gives **7329.77 CPU-s**. It estimated what a whole production run of the package costs, so the like-for-like comparison is against the measured **total**, **6251.42 CPU-s** (cell compute 6229.24 + harness 4.009 + reduce 18.16). Ratio measured total / flat = **0.8529**; the declared flat scale is within a factor 2 of the measured total. Against cell compute alone the ratio would be 0.8499, which is not the correction: it compares a whole-package estimate with part of a package.

### Findings

- **Every production `N` was run, so the whole fit family is measured.** The 6-point `N` grid (400, 800, 1000, 1250, 1600, 3200) is the production grid; the per-cell figure is the unweighted mean of the 6 measured medians, which is exactly how the cells are distributed.
- **The fitted exponent (0.975) agrees with the ALS sweep cost `O(iter . N . d . k)`.** This is the one package where the benchmark is squarely in the flop-bound regime across the whole measured range.
- **The measured per-cell time excludes a per-worker cost the protocol never mentions.** `run_fit_cell` memoises the population reference per `(process, theta)` with an LRU cache. The benchmark discards a warm-up, so the quoted per-cell times are warm-cache times. The cold reference costs 0.567 s (measured). A long-lived worker pays it four times in total (one per theta). A scheduler that starts a fresh process per cell pays it on all 960 fit cells, which adds about 545 CPU-seconds, i.e. about 9% on top of the total. This is recorded in `work_accounting.per_worker_cold_population_reference_seconds_measured`.
- **The CLI harness around the kernel costs 4.16 ms per cell, 4.01 CPU-s over the package, 0.064% of its total.** A production cell is not a kernel call. `common/cli.py` appends a charged `running` record to the execution ledger under an advisory `flock`, runs the kernel inside the fit-count instrumentation, checks the cell schema and the three fit counts, writes the result atomically (json encode + `fsync` + sha256) and appends the completion record. None of that is in the kernel benchmarks. It was measured end to end by running 18 fixture cells through the real CLI (4.13 ms per cell there) and re-priced at the production ledger length (964 records, 0.543 ms per cell) and the production cell-file size (1 410 bytes, 0.641 ms to write). Of the total, 0.524 CPU-s is ledger append, which every worker serialises through one lock: that part is a floor on the wall time at any worker count. The end-to-end figure also carries the once-per-run costs amortised over 18 fixture cells, so it overstates rather than understates.

---

## T21-S — rank-k PCA against the DGJ spiked predictor

The production matrix is 2 499 cells: 2 496 **data** cells and 3 algebraic
cells. A data cell computes the population quantities for its configuration
(`GEO.directions` — a QR of a `d x J` Gaussian; `signal_covariance`; the
predictor's `eigh` of the `d x d` matrix `K`), draws `N` rows, and fits one
closed-form rank-k PCA (a thin SVD of the centred `N x d` draw). Two aspect
ratios are in the production design: the `hi` arm at `N = 128` (`gamma = 2`) and
the `lo` arm at `N = 1024` (`gamma = 0.25`), both at `d = 256`.

### Benchmarks

**Kernel `algebraic_cell`**

| what | n | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `geometry_invariants` | 256 | 0.003379 | 0.0032966 | 0.0034701 | 5 | - | - | - |
| `predictor_sanity` | 256 | 0.0032356 | 0.0031826 | 0.0034504 | 5 | - | - | - |

Production value: **measured** — measured at the production size.

**Kernel `data_cell_correlated_hi`** — `t21s.runner.run_data_cell -> compute_predictions (eigh of the d x d population K, one solve) then fit_pca (thin SVD of the centred N x d draw, one fit)`

> the correlated geometry arm; N is held at the production aspect ratio for the hi arm (gamma = d/N fixed), so the d = 256 point IS a production correlated/hi cell

| d | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 24 | 0.0010817 | 0.0010586 | 0.0012915 | 5 | - | - | - |
| 64 | 0.0020219 | 0.0018417 | 0.0022987 | 5 | - | - | - |
| 128 | 0.0042434 | 0.0041455 | 0.004549 | 5 | - | - | - |
| 256 | 0.014247 | 0.014072 | 0.014324 | 5 | 2.36 MB | 49.6 MB | subprocess |

Scaling fit `t = a * d^b` (least squares on the logs): **a = 2.999e-05**, **b = 1.066**, residual (log RMS) 0.2239, worst |log residual| 0.2499, R² 0.9456 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **1.066** |
| theoretical (thin SVD of an N x d matrix is O(N d min(N, d)); at fixed aspect ratio that is O(d^3)) | **3** |

**Disagreement:** the measured exponent is BELOW the algorithm's by 1.93: at these sizes the kernel is not yet flop-bound (Python/allocation overhead dominates), so a projection made with the fitted exponent UNDERSTATES the cost at larger sizes and one made with the theoretical exponent overstates it at these ones.

Production value: **measured** — measured at the production size.

**Kernel `data_cell_correlated_lo`** — `t21s.runner.run_data_cell -> compute_predictions (eigh of the d x d population K, one solve) then fit_pca (thin SVD of the centred N x d draw, one fit)`

> the correlated geometry arm; N is held at the production aspect ratio for the lo arm (gamma = d/N fixed), so the d = 256 point IS a production correlated/lo cell

| d | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 24 | 0.0012511 | 0.0011754 | 0.0013147 | 5 | - | - | - |
| 64 | 0.0032084 | 0.0031197 | 0.0033037 | 5 | - | - | - |
| 128 | 0.010277 | 0.010182 | 0.010414 | 5 | - | - | - |
| 256 | 0.04137 | 0.04055 | 0.041619 | 5 | 7.57 MB | 60.5 MB | subprocess |

Scaling fit `t = a * d^b` (least squares on the logs): **a = 9.363e-06**, **b = 1.473**, residual (log RMS) 0.2245, worst |log residual| 0.2902, R² 0.9706 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **1.473** |
| theoretical (thin SVD of an N x d matrix is O(N d min(N, d)); at fixed aspect ratio that is O(d^3)) | **3** |

**Disagreement:** the measured exponent is BELOW the algorithm's by 1.53: at these sizes the kernel is not yet flop-bound (Python/allocation overhead dominates), so a projection made with the fitted exponent UNDERSTATES the cost at larger sizes and one made with the theoretical exponent overstates it at these ones.

Production value: **measured** — measured at the production size.

**Kernel `data_cell_orthogonal_hi`** — `t21s.runner.run_data_cell -> compute_predictions (eigh of the d x d population K, one solve) then fit_pca (thin SVD of the centred N x d draw, one fit)`

> the orthogonal geometry arm; N is held at the production aspect ratio for the hi arm (gamma = d/N fixed), so the d = 256 point IS a production orthogonal/hi cell

| d | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 24 | 0.0010926 | 0.0010179 | 0.0014816 | 5 | - | - | - |
| 64 | 0.0018956 | 0.0018521 | 0.0020328 | 5 | - | - | - |
| 128 | 0.004219 | 0.0040976 | 0.0043482 | 5 | - | - | - |
| 256 | 0.013932 | 0.013903 | 0.014015 | 5 | 2.36 MB | 49.5 MB | subprocess |

Scaling fit `t = a * d^b` (least squares on the logs): **a = 3.033e-05**, **b = 1.059**, residual (log RMS) 0.2386, worst |log residual| 0.2700, R² 0.9378 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **1.059** |
| theoretical (thin SVD of an N x d matrix is O(N d min(N, d)); at fixed aspect ratio that is O(d^3)) | **3** |

**Disagreement:** the measured exponent is BELOW the algorithm's by 1.94: at these sizes the kernel is not yet flop-bound (Python/allocation overhead dominates), so a projection made with the fitted exponent UNDERSTATES the cost at larger sizes and one made with the theoretical exponent overstates it at these ones.

Production value: **measured** — measured at the production size.

**Kernel `data_cell_orthogonal_lo`** — `t21s.runner.run_data_cell -> compute_predictions (eigh of the d x d population K, one solve) then fit_pca (thin SVD of the centred N x d draw, one fit)`

> the orthogonal geometry arm; N is held at the production aspect ratio for the lo arm (gamma = d/N fixed), so the d = 256 point IS a production orthogonal/lo cell

| d | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 24 | 0.0012582 | 0.0012023 | 0.0012981 | 5 | - | - | - |
| 64 | 0.0031874 | 0.0031225 | 0.0034724 | 5 | - | - | - |
| 128 | 0.010264 | 0.010229 | 0.010334 | 5 | - | - | - |
| 256 | 0.041 | 0.040506 | 0.041295 | 5 | 7.57 MB | 60.5 MB | subprocess |

Scaling fit `t = a * d^b` (least squares on the logs): **a = 9.555e-06**, **b = 1.468**, residual (log RMS) 0.2259, worst |log residual| 0.2954, R² 0.9700 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **1.468** |
| theoretical (thin SVD of an N x d matrix is O(N d min(N, d)); at fixed aspect ratio that is O(d^3)) | **3** |

**Disagreement:** the measured exponent is BELOW the algorithm's by 1.53: at these sizes the kernel is not yet flop-bound (Python/allocation overhead dominates), so a projection made with the fitted exponent UNDERSTATES the cost at larger sizes and one made with the theoretical exponent overstates it at these ones.

Production value: **measured** — measured at the production size.

### Projection arithmetic

| family | cells | s/cell (median) | s/cell (min-max) | basis | fitted b | theory b | extrap. factor in dimension | cells x s/cell = CPU-s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| `algebraic` | 3 | 0.0033312 | 0.0031826 - 0.0034701 | measured | - | - | 1 | 3 x 0.0033312 = **0.00999355** |
| `confirmation-correlated-hi` | 520 | 0.014247 | 0.014072 - 0.014324 | measured | 1.07 | 3 | 1 | 520 x 0.014247 = **7.40827** |
| `confirmation-correlated-lo` | 520 | 0.04137 | 0.04055 - 0.041619 | measured | 1.47 | 3 | 1 | 520 x 0.04137 = **21.5123** |
| `confirmation-orthogonal-hi` | 520 | 0.013932 | 0.013903 - 0.014015 | measured | 1.06 | 3 | 1 | 520 x 0.013932 = **7.24466** |
| `confirmation-orthogonal-lo` | 520 | 0.041 | 0.040506 - 0.041295 | measured | 1.47 | 3 | 1 | 520 x 0.041 = **21.3198** |
| `pilot-correlated-hi` | 104 | 0.014247 | 0.014072 - 0.014324 | measured | 1.07 | 3 | 1 | 104 x 0.014247 = **1.48165** |
| `pilot-correlated-lo` | 104 | 0.04137 | 0.04055 - 0.041619 | measured | 1.47 | 3 | 1 | 104 x 0.04137 = **4.30247** |
| `pilot-orthogonal-hi` | 104 | 0.013932 | 0.013903 - 0.014015 | measured | 1.06 | 3 | 1 | 104 x 0.013932 = **1.44893** |
| `pilot-orthogonal-lo` | 104 | 0.041 | 0.040506 - 0.041295 | measured | 1.47 | 3 | 1 | 104 x 0.041 = **4.26397** |
| **cell compute** | 2499 | | | | | | | **68.9921** (range 68.0451 - 69.4322) |
| **harness** (`common/cli.py` per cell) | 2499 | 0.003443 | | measured end to end, priced per cell | | | | **8.604** |
| **reduction** | | | | measured rate, projected | | | | **3.626** |
| **total** | | | | | | | | **81.2218** (range 80.2748 - 81.6619) |

Per-cell figures are the measured median over repeats where a family's cells are identical, and the mean over the family's own cells of their measured medians where they are not. The low/high columns are the corresponding measured extremes: they are measurements, not margins.

**Reduction work.** Measured on the fixture run root: 0.5212 s of reduce work over 419 cells = **0.001244 s per cell** (interpreter and import, 0.108 s, subtracted). Measured input throughput 3.71 MB/s against a warm page cache, so **at that size** (419 small fixture cells) the step is **compute-bound**: it reads every cell file once but spends its time re-deriving summaries and bootstrapping. That classification is a measurement of the fixture scale and is not asserted of the production scale; see the sentence after this table. Priced as 416 cells (pilot reduce) x 0.001244 s = 0.5175 s + 2499 cells (final reduce over the whole run root) x 0.001244 s = 3.109 s = **3.626 s**.

**Reduce at the production input size.** At the production scale reduce reads 6 MB in 3.63 s, i.e. 1.7 MB/s, still far below the 50 MB/s at which I/O would take over, so the compute-bound classification measured on the fixture also holds at production for this package.

**Harness work.** A production cell goes through `common/cli.py`, not through the kernel alone: ledger `start` under an advisory `flock`, the kernel, the schema and fit-accounting checks, the atomic result write (json encode + `fsync` + sha256) and ledger `complete`. Measured end to end by running 419 fixture cells through the real CLI (3.45 ms per cell), then re-priced at the production ledger length and the production cell-file size: **3.443 ms per cell**, 0.00344284 s x 2499 cells = **8.604 CPU-s**. Of that, **1.206 CPU-s** is ledger append, which every worker serialises through one lock; it appears again as the serial term in the wall-time model below. Per-worker interpreter and import start-up is 0.106 s, measured; it is paid once per worker **process** and is therefore reported here rather than added to the per-cell figure.

### Wall time

Cells are independent processes, so the cell compute and the parallel part of the per-cell harness are taken as perfectly parallel over W workers.  Two terms are NOT parallel and are added whole: the ledger appends, which every worker serialises through one advisory flock, and the reduction, which is a single process.  wall = (cell_compute + parallel_harness) / W + serial_ledger + reduction.  The serial ledger term is a floor no worker count beats.

Measured terms: cell compute 68.9921 s (parallel), harness 7.397 s (parallel), ledger append 1.206 s (**serial**), reduce 3.626 s (**serial**). The serial floor is 4.832 s (0.001342 h) and no worker count goes below it.

| workers | wall hours (low - high) | serial floor is | status |
|---:|---|---|---|
| 1 | 0.0223 - 0.02268 | 5.9% of the point figure | CONDITIONAL on 1 usable cores; the server specification is not known |
| 8 | 0.003962 - 0.00401 | 33.6% of the point figure | CONDITIONAL on 8 usable cores; the server specification is not known |
| 32 | 0.001997 - 0.002009 | 66.9% of the point figure — **this row is dominated by the serial floor** | CONDITIONAL on 32 usable cores; the server specification is not known |

### Peak memory and output storage

Peak memory per cell at the production dimension: **60.5 MB process RSS** (`data_cell_orthogonal_lo` at 256, measured in a subprocess), of which 7.57 MB is Python-allocator peak for the call itself. The RSS figure includes the interpreter and numpy, so it is what one worker actually needs. MEASURED.

| item | cells | bytes/cell | total | basis |
|---|---:|---:|---:|---|
| `algebraic` | 3 | 963 | 2.89 kB | measured |
| `confirmation-correlated-hi` | 520 | 2,583 | 1.34 MB | measured |
| `confirmation-correlated-lo` | 520 | 2,519 | 1.31 MB | measured |
| `confirmation-orthogonal-hi` | 520 | 2,544 | 1.32 MB | measured |
| `confirmation-orthogonal-lo` | 520 | 2,496 | 1.3 MB | measured |
| `pilot-correlated-hi` | 104 | 2,583 | 269 kB | measured |
| `pilot-correlated-lo` | 104 | 2,519 | 262 kB | measured |
| `pilot-orthogonal-hi` | 104 | 2,544 | 265 kB | measured |
| `pilot-orthogonal-lo` | 104 | 2,496 | 260 kB | measured |
| execution ledger | 2499 | | 4.75 MB | projected from a measured per-cell rate |
| `REDUCED.json` | | | 4.65 MB | projected linearly from the fixture |
| **total** | | | **15.7 MB** | |

### Cross-check against the protocol's flat scale factor

The old estimate (one fixture wall time x `budget.fixture_to_production_scale`) gives **4891.29 CPU-s**. It estimated what a whole production run of the package costs, so the like-for-like comparison is against the measured **total**, **81.2218 CPU-s** (cell compute 68.9921 + harness 8.604 + reduce 3.626). Ratio measured total / flat = **0.01661**; the declared flat scale disagrees with the measured TOTAL by a factor 60.2; the flat factor is wrong, not the measurement. Against cell compute alone the ratio would be 0.01411, which is not the correction: it compares a whole-package estimate with part of a package.

### Findings

- **The declared flat scale overstates the package by a factor of about 60, measured like for like. This is still the largest single correction in this round.** The like-for-like comparison is the old flat number against everything a production run of this package costs, which is cell compute **plus** the CLI harness **plus** the reduction: 4891 CPU-s declared against 81.22 CPU-s measured (69 cell compute + 8.6 harness + 3.63 reduce), a factor 60.2. Against cell compute alone the ratio is 70.9, and an earlier version of this report quoted that larger number as if it were the correction: it is not, because the flat estimate was an estimate of the whole package and the kernel time is not. `budget.fixture_to_production_scale` is the ratio of the PCA flop counts `N d min(N,d)` between the production and the fixture cell; multiplied by a fixture cell of about 1.6 ms it gives about 2.0 s per production cell. The measured production cells are 0.0142 s (`hi`) and 0.0414 s (`lo`).
- **The dominant cost is not the estimator the flat factor was derived from.** The scale was derived from the PCA flop count. At `d = 256, N = 128` the thin SVD of a 128x256 matrix is a few hundred microseconds; the cell takes 14.2 ms. The time is in `compute_predictions` - the `d x d` eigendecomposition of `K`, the QR of the direction frame, and the per-direction predictor arithmetic - all of which are independent of `N` and were not in the flop ratio at all. The `hi`/`lo` split confirms it: raising `N` from 128 to 1024 multiplies the cell time by 2.9, so the `N`-dependent part is real but it sits on top of a large `N`-independent base.
- **Both geometry arms were timed, and they differ by less than the arithmetic needs.** The correlated arm is 0.01425 s (`hi`) and 0.04137 s (`lo`); the orthogonal arm is 0.01393 s and 0.041 s, i.e. 2.2% and 0.89% away. An earlier pass timed only the correlated arm and copied its number onto the orthogonal families, which labelled 1248 cells `measured` for a configuration that had never been run. The number barely moved; the label was wrong, and all four families are now timed.
- **The fitted exponents (1.07 `hi`, 1.47 `lo`) are far below the SVD's 3.** Again a small-size effect, and again the production size was measured so nothing is projected with the fitted exponent. It does say that a flop-ratio argument between `d = 24` and `d = 256` cannot be trusted for this package, which is precisely how the flat factor's error arose.
- **At 69 CPU-seconds of cell compute, the two things the kernel benchmark does not measure are 15% of the package.** The measured reduce rate is 1.24 ms per cell over 419 fixture cells; over 2 499 production cells and two reduces (pilot and final) that is 3.63 s. The CLI harness adds 8.6 s. Against 69 s of cell compute they are not a rounding error, and an estimate that omitted them - the previous one omitted both - would be materially wrong.
- **The CLI harness around the kernel costs 3.44 ms per cell, 8.6 CPU-s over the package, 11% of its total.** A production cell is not a kernel call. `common/cli.py` appends a charged `running` record to the execution ledger under an advisory `flock`, runs the kernel inside the fit-count instrumentation, checks the cell schema and the three fit counts, writes the result atomically (json encode + `fsync` + sha256) and appends the completion record. None of that is in the kernel benchmarks. It was measured end to end by running 419 fixture cells through the real CLI (3.45 ms per cell there) and re-priced at the production ledger length (2 499 records, 0.483 ms per cell) and the production cell-file size (2 533 bytes, 0.835 ms to write). Of the total, 1.21 CPU-s is ledger append, which every worker serialises through one lock: that part is a floor on the wall time at any worker count. The end-to-end figure also carries the once-per-run costs amortised over 419 fixture cells, so it overstates rather than understates.

---

## T22-S — Monte Carlo over the archived T22 inference pipeline

The production matrix is 727 cells: 690 confirmation simulation cells, 32
pilot simulation cells and 5 controls. **No cell fits anything** — the simulator
generates the learned estimates, so the production fit count is 0 and all the
work is simulation and bootstrap replicates. One simulation cell is
`mc_per_cell = 100` Monte-Carlo replicates of `pipeline.run_replicate`, and each
replicate runs the archived inference pipeline end to end: block-variance curves
on the development files, `te.joint_bootstrap` once per evaluation file
(`R_file = 2000` at the archived setting), and the two-stage hierarchical file
bootstrap (`R_hier = 10 000`).

### Benchmarks

**Kernel `control_cell`**

| what | n | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ar1_autocorrelation` | 1 | 0.090712 | 0.090421 | 0.091003 | 2 | - | - | - |
| `dependence_gate` | 1 | 4.9749 | 4.9602 | 4.9896 | 2 | - | - | - |
| `holm_arithmetic` | 1 | 5.735e-05 | 2.4707e-05 | 8.9992e-05 | 2 | - | - | - |
| `iid_bootstrap_se` | 1 | 0.54817 | 0.54646 | 0.54988 | 2 | - | - | - |
| `simulator_estimand` | 1 | 0.037446 | 0.037259 | 0.037632 | 2 | - | - | - |

Production value: **measured** — measured at the production configuration.

**Kernel `pipeline_replicate`** — `t22s.pipeline.run_replicate: te.joint_bootstrap per evaluation file (R_file), t22.combine_stage two-stage hierarchical bootstrap (R_hier = 5 R_file) and te.block_variance_curve on the development files (R_curve = R_file/5)`

> the three replicate counts are moved together at the archived ratio 1 : 5 : 0.2, so the size axis is the whole replicate budget of one Monte-Carlo replicate; the production (archived) setting was run directly

| R_file | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 60 | 0.091758 | 0.090154 | 0.1025 | 3 | - | - | - |
| 250 | 0.35927 | 0.33557 | 0.35931 | 3 | - | - | - |
| 1000 | 1.3088 | 1.3026 | 1.3118 | 3 | - | - | - |
| 2000 | 2.5761 | 2.5463 | 2.6614 | 6 | 2.53 MB | 47.5 MB | subprocess |

Scaling fit `t = a * R_file^b` (least squares on the logs): **a = 0.001895**, **b = 0.948**, residual (log RMS) 0.0087, worst |log residual| 0.0130, R² 1.0000 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **0.948** |
| theoretical (a bootstrap is linear in the number of replicates) | **1** |

Agreement: the measured exponent agrees with the algorithm within +/- 0.35.

Production value: **measured** — measured at the production size.

**Kernel `replicate_by_n_files`**

| n_files | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 2.5761 | 2.5463 | 2.6614 | 6 | - | - | - |
| 8 | 4.1509 | 4.1306 | 4.1652 | 3 | - | - | - |
| 16 | 7.2422 | 7.2276 | 7.2952 | 3 | - | - | - |

Scaling fit `t = a * n_files^b` (least squares on the logs): **a = 0.9042**, **b = 0.746**, residual (log RMS) 0.0188, worst |log residual| 0.0265, R² 0.9980 over 3 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **0.746** |
| theoretical (the joint bootstrap is run once per evaluation file, so the replicate is linear in n_files above a fixed simulation cost) | **1** |

Agreement: the measured exponent agrees with the algorithm within +/- 0.35.

Production value: **measured** — every production n_files (4, 8, 16) was run.

### Projection arithmetic

| family | cells | s/cell (median) | s/cell (min-max) | basis | fitted b | theory b | extrap. factor in dimension | cells x s/cell = CPU-s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| `control` | 5 | 1.1303 | 1.1269 - 1.1336 | measured | - | - | 1 | 5 x 1.1303 = **5.65128** |
| `sim_confirm` | 690 | 474.43 | 472.25 - 480.15 | measured | 0.948 | 1 | 1 | 690 x 474.43 = **327358** |
| `sim_pilot` | 32 | 317.81 | 316.16 - 321.52 | measured | 0.948 | 1 | 1 | 32 x 317.81 = **10169.9** |
| **cell compute** | 727 | | | | | | | **337533** (range 335976 - 341598) |
| **harness** (`common/cli.py` per cell) | 727 | 0.02313 | | measured end to end, priced per cell | | | | **16.81** |
| **reduction** | | | | measured rate, projected | | | | **4.022** |
| **total** | | | | | | | | **337554** (range 335997 - 341619) |

Per-cell figures are the measured median over repeats where a family's cells are identical, and the mean over the family's own cells of their measured medians where they are not. The low/high columns are the corresponding measured extremes: they are measurements, not margins.

**Reduction work.** Measured on the fixture run root: 0.07418 s of reduce work over 14 cells = **0.005299 s per cell** (interpreter and import, 0.114 s, subtracted). Measured input throughput 4.08 MB/s against a warm page cache, so **at that size** (14 small fixture cells) the step is **compute-bound**: it reads every cell file once but spends its time re-deriving summaries and bootstrapping. That classification is a measurement of the fixture scale and is not asserted of the production scale; see the sentence after this table. Priced as 32 cells (pilot reduce) x 0.005299 s = 0.1696 s + 727 cells (final reduce over the whole run root) x 0.005299 s = 3.852 s = **4.022 s**.

**Reduce at the production input size.** At the production scale reduce must read 719 MB in the 4.02 s the measured per-cell rate allows, i.e. 179 MB/s.  That is above the 50 MB/s at which a warm page cache stops being free, so at production this step is expected to be I/O-DOMINATED and the reduce figure above is a LOWER bound: the compute-bound classification measured on the fixture does not carry to this scale.

**Harness work.** A production cell goes through `common/cli.py`, not through the kernel alone: ledger `start` under an advisory `flock`, the kernel, the schema and fit-accounting checks, the atomic result write (json encode + `fsync` + sha256) and ledger `complete`. Measured end to end by running 14 fixture cells through the real CLI (18.6 ms per cell), then re-priced at the production ledger length and the production cell-file size: **23.13 ms per cell**, 0.023127 s x 727 cells = **16.81 CPU-s**. Of that, **0.2865 CPU-s** is ledger append, which every worker serialises through one lock; it appears again as the serial term in the wall-time model below. Per-worker interpreter and import start-up is 0.114 s, measured; it is paid once per worker **process** and is therefore reported here rather than added to the per-cell figure.

### Wall time

Cells are independent processes, so the cell compute and the parallel part of the per-cell harness are taken as perfectly parallel over W workers.  Two terms are NOT parallel and are added whole: the ledger appends, which every worker serialises through one advisory flock, and the reduction, which is a single process.  wall = (cell_compute + parallel_harness) / W + serial_ledger + reduction.  The serial ledger term is a floor no worker count beats.

Measured terms: cell compute 337533 s (parallel), harness 16.53 s (parallel), ledger append 0.2865 s (**serial**), reduce 4.022 s (**serial**). The serial floor is 4.308 s (0.001197 h) and no worker count goes below it.

| workers | wall hours (low - high) | serial floor is | status |
|---:|---|---|---|
| 1 | 93.33 - 94.89 | 0.0% of the point figure | CONDITIONAL on 1 usable cores; the server specification is not known |
| 8 | 11.67 - 11.86 | 0.0% of the point figure | CONDITIONAL on 8 usable cores; the server specification is not known |
| 32 | 2.918 - 2.967 | 0.0% of the point figure | CONDITIONAL on 32 usable cores; the server specification is not known |

### Peak memory and output storage

Peak memory per cell at the production dimension: **47.5 MB process RSS** (`pipeline_replicate` at 2000, measured in a subprocess), of which 2.53 MB is Python-allocator peak for the call itself. The RSS figure includes the interpreter and numpy, so it is what one worker actually needs. MEASURED.

| item | cells | bytes/cell | total | basis |
|---|---:|---:|---:|---|
| `control` | 5 | 964 | 4.82 kB | measured |
| `sim_confirm` | 690 | 998,378 | 689 MB | projected |
| `sim_pilot` | 32 | 928,862 | 29.7 MB | projected |
| execution ledger | 727 | | 1.38 MB | projected from a measured per-cell rate |
| `REDUCED.json` | | | 3.29 MB | projected linearly from the fixture |
| **total** | | | **723.3 MB** | |

### Cross-check against the protocol's flat scale factor

The old estimate (one fixture wall time x `budget.fixture_to_production_scale`) gives **378823 CPU-s**. It estimated what a whole production run of the package costs, so the like-for-like comparison is against the measured **total**, **337554 CPU-s** (cell compute 337533 + harness 16.81 + reduce 4.022). Ratio measured total / flat = **0.8911**; the declared flat scale is within a factor 2 of the measured total. Against cell compute alone the ratio would be 0.891, which is not the correction: it compares a whole-package estimate with part of a package.

### Findings

- **The production replicate setting was run directly.** `file 2000 / hier 10000 / curve 400` is one of the 4 benchmarked replicate budgets, and all three production `n_files` values (4, 8, 16) were run at it. Only the multiplication by `mc = 100` and by the cell count is arithmetic.
- **This package is 98% of the whole round's CPU cost, and that survives the correction.** 337 554 CPU-s (93.8 CPU-hours) against 7 412 CPU-s (2.06 CPU-hours) for the other four packages combined. The flat cross-check said 378 823 CPU-s; the measurement is 11% lower. The order of magnitude was right, so unlike t21s the flat factor here was not badly wrong - but it was still a product of a synthetic work proxy, and the measurement replaces it.
- **The two timings of the same call are reconciled rather than averaged away.** `pipeline_replicate` at `R_file = 2000` and `replicate_by_n_files` at `n_files = 4` are one and the same call, and they came out 1.8% apart - about three times the min/max spread either of them quoted. The repeats of both are now pooled: the per-replicate figure is 2.5761 s over 6 repeats and the quoted range 2.5463 - 2.6614 s is BETWEEN-run variability of the identical call (4.5% of the median), not the jitter inside one timing.
- **The confirmation total is an upper bound, for a reason the protocol states.** The confirmation cells run the replicate setting the pilot freezes, which is not known before the pilot. Pricing uses the archived setting, the more expensive of the two. The measured ratio reduced/archived is 0.467 at `n_files = 4`, so if the pilot freezes `reduced` the confirmation cost roughly halves, to about 42 CPU-hours for that family.
- **Output storage, not CPU, is the constraint a receiving site is most likely to trip over - and at that size the reduce step stops being compute-bound.** The projected cell files come to 723 MB, essentially all of it the 690 confirmation cells at about 1 MB each (100 Monte-Carlo records per cell). That is 22x the other four packages combined (32 MB) and about 93x the transferred archive itself. At the production scale reduce must read 719 MB in the 4.02 s the measured per-cell rate allows, i.e. 179 MB/s.  That is above the 50 MB/s at which a warm page cache stops being free, so at production this step is expected to be I/O-DOMINATED and the reduce figure above is a LOWER bound: the compute-bound classification measured on the fixture does not carry to this scale.
- **The CLI harness around the kernel costs 23.1 ms per cell, 16.8 CPU-s over the package, 0.005% of its total.** A production cell is not a kernel call. `common/cli.py` appends a charged `running` record to the execution ledger under an advisory `flock`, runs the kernel inside the fit-count instrumentation, checks the cell schema and the three fit counts, writes the result atomically (json encode + `fsync` + sha256) and appends the completion record. None of that is in the kernel benchmarks. It was measured end to end by running 14 fixture cells through the real CLI (18.6 ms per cell there) and re-priced at the production ledger length (727 records, 0.394 ms per cell) and the production cell-file size (988 458 bytes, 5.18 ms to write). Of the total, 0.286 CPU-s is ledger append, which every worker serialises through one lock: that part is a floor on the wall time at any worker count. The end-to-end figure also carries the once-per-run costs amortised over 14 fixture cells, so it overstates rather than understates.

---

## CRESST-S — archive recomputation, cluster reporting, EMPCA robustness, calibration

The production matrix is 395 cells across nine families: 1 archive-point
cell, 10 archived interval reconstructions, 5 cluster reports, 4 science
controls, 1 calibration control, 350 EMPCA robustness cells (35 pilot + 315
confirmation) and 24 calibration cells. The robustness cells are the only
estimator work (3 fits each: 1 classic PCA + 2 EMPCA restarts); everything else
is bootstrap replicates over the archived residual table or over simulated
clusters.

### Benchmarks

**Kernel `archive_interval`** — `cresst_s.archive_recompute.recompute_interval: a cluster bootstrap over the 19335 archived evaluation events in 7 channels`

> the production replicate count was run directly

| R | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1200 | 0.19921 | 0.18972 | 0.20496 | 3 | - | - | - |
| 2500 | 0.41249 | 0.40044 | 0.46619 | 3 | - | - | - |
| 5000 | 0.81576 | 0.80503 | 0.87177 | 3 | - | - | - |
| 10000 | 1.6548 | 1.6462 | 1.6626 | 3 | 64.3 MB | 141 MB | subprocess |

Scaling fit `t = a * R^b` (least squares on the logs): **a = 0.000169**, **b = 0.997**, residual (log RMS) 0.0061, worst |log residual| 0.0096, R² 0.9999 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **0.997** |
| theoretical (a bootstrap is linear in the number of replicates) | **1** |

Agreement: the measured exponent agrees with the algorithm within +/- 0.35.

Production value: **measured** — measured at the production size.

**Kernel `archive_point`**

| n | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.00027667 | 0.00024444 | 0.00028327 | 3 | - | - | - |

Production value: **measured** — measured at the production size.

**Kernel `calibration_control`**

| what | n | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `null_dependence` | 1 | 7.3144 | 7.2047 | 7.4241 | 2 | - | - | - |

Production value: **measured** — measured at the production configuration.

**Kernel `calibration_coverage`** — `cresst_s.calibration.coverage: mc Monte-Carlo datasets x bootstrap_replicates cluster bootstrap draws each`

> at the reference design (n_runs 2, channels_per_run 2); the production mc was run directly

| mc | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 50 | 0.12041 | 0.11488 | 0.12281 | 3 | - | - | - |
| 100 | 0.23311 | 0.23155 | 0.23852 | 3 | - | - | - |
| 200 | 0.45945 | 0.45908 | 0.46345 | 3 | - | - | - |
| 400 | 0.92698 | 0.92278 | 0.93024 | 3 | 1.56 MB | 71.5 MB | subprocess |

Scaling fit `t = a * mc^b` (least squares on the logs): **a = 0.002565**, **b = 0.981**, residual (log RMS) 0.0104, worst |log residual| 0.0111, R² 0.9998 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **0.981** |
| theoretical (a bootstrap is linear in the number of replicates) | **1** |

Agreement: the measured exponent agrees with the algorithm within +/- 0.35.

Production value: **measured** — measured at the production size.

**Kernel `cluster_report`** — `cresst_s.runner._cell_cluster_report -> cluster_report.* (cluster bootstrap per report)`

> the production replicate count was run directly

| R | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1200 | 0.46191 | 0.46077 | 0.46304 | 2 | - | - | - |
| 2500 | 1.0019 | 0.97178 | 1.0319 | 2 | - | - | - |
| 5000 | 2.1665 | 2.0628 | 2.2702 | 2 | - | - | - |
| 10000 | 4.1849 | 4.1238 | 4.2461 | 2 | 99.9 MB | 202 MB | subprocess |

Scaling fit `t = a * R^b` (least squares on the logs): **a = 0.0002791**, **b = 1.047**, residual (log RMS) 0.0250, worst |log residual| 0.0411, R² 0.9991 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **1.047** |
| theoretical (a bootstrap is linear in the number of replicates) | **1** |

Agreement: the measured exponent agrees with the algorithm within +/- 0.35.

Production value: **measured** — measured at the production size.

**Kernel `empca_robustness`** — `cresst_s.robustness_design.fit_and_score -> src.canonical.empca.classic_pca (1 fit) + empca (EMPCA_RESTARTS = 2 fits, niter = 25)`

> the real CRESST pulse traces are NOT in this container; the synthetic pulse root written by the package's own write_fixture_data_root is used AT THE PRODUCTION DIMENSIONS (n_events 400, n_samples 128, n_noise 400). The EMPCA cost is set by those dimensions and by the fixed niter, not by the values, but the substitution is a substitution and is labelled here.

| rank k | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0.29346 | 0.29173 | 0.32678 | 3 | - | - | - |
| 4 | 0.39672 | 0.3571 | 0.43294 | 3 | - | - | - |
| 6 | 0.45474 | 0.44693 | 0.47436 | 3 | - | - | - |
| 8 | 0.53809 | 0.53 | 0.55358 | 3 | - | - | - |
| 12 | 0.73545 | 0.73182 | 0.78218 | 3 | 3.9 MB | 76 MB | subprocess |

Scaling fit `t = a * rank k^b` (least squares on the logs): **a = 0.2011**, **b = 0.493**, residual (log RMS) 0.0503, worst |log residual| 0.0729, R² 0.9730 over 5 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **0.493** |
| theoretical (a weighted-EM sweep is O(niter . n_train . n_components . k), so linear in the rank k) | **1** |

**Disagreement:** the measured exponent is BELOW the algorithm's by 0.51: at these sizes the kernel is not yet flop-bound (Python/allocation overhead dominates), so a projection made with the fitted exponent UNDERSTATES the cost at larger sizes and one made with the theoretical exponent overstates it at these ones.

Production value: **measured** — measured at the production size.

**Kernel `science_control`**

| what | n | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `block1_equals_iid` | 1 | 0.0022484 | 0.0020837 | 0.0024132 | 2 | - | - | - |
| `empca_uniform_weight_limit` | 1 | 0.61955 | 0.61847 | 0.62062 | 2 | - | - | - |
| `event_weight_constant` | 1 | 0.036347 | 0.035942 | 0.036752 | 2 | - | - | - |
| `zero_effect_identity` | 1 | 1.3353 | 1.3104 | 1.3602 | 2 | - | - | - |

Production value: **measured** — measured at the production configuration.

### Projection arithmetic

| family | cells | s/cell (median) | s/cell (min-max) | basis | fitted b | theory b | extrap. factor in dimension | cells x s/cell = CPU-s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| `archive_interval` | 10 | 2.0211 | 1.3865 - 2.6009 | measured | 0.997 | 1 | 1 | 10 x 2.0211 = **20.2107** |
| `archive_point` | 1 | 0.00027667 | 0.00024444 - 0.00028327 | measured | - | - | 1 | 1 x 0.00027667 = **0.000276674** |
| `calib_confirm` | 12 | 3.6901 | 3.6783 - 3.7019 | measured | 0.981 | 1 | 1 | 12 x 3.6901 = **44.2815** |
| `calib_control` | 1 | 7.3144 | 7.2047 - 7.4241 | measured | - | - | 1 | 1 x 7.3144 = **7.3144** |
| `calib_pilot` | 12 | 1.4061 | 1.3964 - 1.4158 | measured | 0.981 | 1 | 1 | 12 x 1.4061 = **16.8729** |
| `cluster_report` | 5 | 4.7551 | 1.854 - 9.5812 | measured | 1.05 | 1 | 1 | 5 x 4.7551 = **23.7756** |
| `robustness_confirm` | 315 | 0.32185 | 0.31899 - 0.33319 | measured-on-substitute | 0.493 | 1 | 1 | 315 x 0.32185 = **101.384** |
| `robustness_pilot` | 35 | 0.48129 | 0.47617 - 0.49199 | measured-on-substitute | 0.493 | 1 | 1 | 35 x 0.48129 = **16.8452** |
| `science_control` | 4 | 0.49837 | 0.0020837 - 1.3602 | measured | - | - | 1 | 4 x 0.49837 = **1.99348** |
| **cell compute** | 395 | | | | | | | **232.678** (range 208.393 - 270.369) |
| **harness** (`common/cli.py` per cell) | 395 | 0.004568 | | measured end to end, priced per cell | | | | **1.804** |
| **reduction** | | | | measured rate, projected | | | | **0.6112** |
| **total** | | | | | | | | **235.093** (range 210.809 - 272.784) |

Per-cell figures are the measured median over repeats where a family's cells are identical, and the mean over the family's own cells of their measured medians where they are not. The low/high columns are the corresponding measured extremes: they are measurements, not margins.

**Reduction work.** Measured on the fixture run root: 0.0401 s of reduce work over 29 cells = **0.001383 s per cell** (interpreter and import, 0.297 s, subtracted). Measured input throughput 4.52 MB/s against a warm page cache, so **at that size** (29 small fixture cells) the step is **compute-bound**: it reads every cell file once but spends its time re-deriving summaries and bootstrapping. That classification is a measurement of the fixture scale and is not asserted of the production scale; see the sentence after this table. Priced as 47 cells (pilot reduce) x 0.001383 s = 0.065 s + 395 cells (final reduce over the whole run root) x 0.001383 s = 0.5463 s = **0.6112 s**.

**Reduce at the production input size.** At the production scale reduce reads 0 MB in 0.611 s, i.e. 0.4 MB/s, still far below the 50 MB/s at which I/O would take over, so the compute-bound classification measured on the fixture also holds at production for this package.

**Harness work.** A production cell goes through `common/cli.py`, not through the kernel alone: ledger `start` under an advisory `flock`, the kernel, the schema and fit-accounting checks, the atomic result write (json encode + `fsync` + sha256) and ledger `complete`. Measured end to end by running 29 fixture cells through the real CLI (4.56 ms per cell), then re-priced at the production ledger length and the production cell-file size: **4.568 ms per cell**, 0.0045676 s x 395 cells = **1.804 CPU-s**. Of that, **0.1393 CPU-s** is ledger append, which every worker serialises through one lock; it appears again as the serial term in the wall-time model below. Per-worker interpreter and import start-up is 0.295 s, measured; it is paid once per worker **process** and is therefore reported here rather than added to the per-cell figure.

### Wall time

Cells are independent processes, so the cell compute and the parallel part of the per-cell harness are taken as perfectly parallel over W workers.  Two terms are NOT parallel and are added whole: the ledger appends, which every worker serialises through one advisory flock, and the reduction, which is a single process.  wall = (cell_compute + parallel_harness) / W + serial_ledger + reduction.  The serial ledger term is a floor no worker count beats.

Measured terms: cell compute 232.678 s (parallel), harness 1.665 s (parallel), ledger append 0.1393 s (**serial**), reduce 0.6112 s (**serial**). The serial floor is 0.7506 s (0.0002085 h) and no worker count goes below it.

| workers | wall hours (low - high) | serial floor is | status |
|---:|---|---|---|
| 1 | 0.05856 - 0.07577 | 0.3% of the point figure | CONDITIONAL on 1 usable cores; the server specification is not known |
| 8 | 0.007502 - 0.009654 | 2.5% of the point figure | CONDITIONAL on 8 usable cores; the server specification is not known |
| 32 | 0.002032 - 0.00257 | 9.3% of the point figure | CONDITIONAL on 32 usable cores; the server specification is not known |

### Peak memory and output storage

Peak memory per cell at the production dimension: **202 MB process RSS** (`cluster_report` at 10000, measured in a subprocess), of which 99.9 MB is Python-allocator peak for the call itself. The RSS figure includes the interpreter and numpy, so it is what one worker actually needs. MEASURED.

| item | cells | bytes/cell | total | basis |
|---|---:|---:|---:|---|
| `archive_interval` | 10 | 3,028 | 30.3 kB | measured |
| `archive_point` | 1 | 9,042 | 9.04 kB | measured |
| `calib_confirm` | 12 | 1,255 | 15.1 kB | measured |
| `calib_control` | 1 | 2,555 | 2.56 kB | measured |
| `calib_pilot` | 12 | 1,255 | 15.1 kB | measured |
| `cluster_report` | 5 | 6,988 | 34.9 kB | measured |
| `robustness_confirm` | 315 | 377 | 119 kB | measured |
| `robustness_pilot` | 35 | 377 | 13.2 kB | measured |
| `science_control` | 4 | 675 | 2.7 kB | measured |
| execution ledger | 395 | | 755 kB | projected from a measured per-cell rate |
| `REDUCED.json` | | | 984 kB | projected linearly from the fixture |
| **total** | | | **2.0 MB** | |

### Cross-check against the protocol's flat scale factor

The old estimate (one fixture wall time x `budget.fixture_to_production_scale`) gives **583.973 CPU-s**. It estimated what a whole production run of the package costs, so the like-for-like comparison is against the measured **total**, **235.093 CPU-s** (cell compute 232.678 + harness 1.804 + reduce 0.6112). Ratio measured total / flat = **0.4026**; the declared flat scale disagrees with the measured TOTAL by a factor 2.48; the flat factor is wrong, not the measurement. Against cell compute alone the ratio would be 0.3984, which is not the correction: it compares a whole-package estimate with part of a package.

### Findings

- **Every family except the robustness grid was benchmarked at its exact production configuration on the real archived inputs.** The 10 interval reconstructions, the 5 cluster reports, the 4 science controls, the calibration control and all 24 calibration designs were each run at their production replicate budgets. Those numbers are measurements of the production cell, not projections.
- **The robustness cells cannot be executed here at all, and the substitution is on the data, not on the dimensions.** They need raw CRESST pulse traces, which are not in this container and may not be sought here; `run_cell` raises `input not available` for every one of them. The benchmark therefore used the package's own `write_fixture_data_root` at the *production* dimensions and ran the real `fit_and_score`. EMPCA runs a fixed `niter = 25`, so the cost is set by the dimensions and not by the values; the numbers are labelled `measured-on-substitute` and would move linearly in `n_train` if a real channel has a different event count. Measured: 0.3219 s per confirmation cell and 0.4813 s per pilot cell, 101.4 + 16.85 = 118.2 CPU-s for the grid.
- **The EMPCA cost is only weakly a function of the rank: fitted exponent 0.49 against a theoretical 1.** An EM sweep is `O(niter . n_train . n_components . k)`, so the rank should be linear. It is not, because a robustness cell also pays a rank-independent base: the stationary PSD estimate, the real-FFT representation of every pulse, and the classic-PCA fit. Going from rank 2 to rank 12 only takes the cell from 0.293 s to 0.735 s. The protocol's cost proxy has no such base term, which is why its rank-6 estimate (1.2-1.6 s/cell) is several times the measured 0.32-0.48 s.
- **The declared flat scale overstates the package by a factor 2.48.** 584 CPU-s declared against 235.1 CPU-s measured for the whole package (cell compute 232.7 + harness 1.8 + reduce 0.611). Most of the gap is the robustness grid (350 of the 395 cells).
- **The 5 cluster-report cells differ by a factor 5.2 among themselves.** `leave_one_channel_out` is 9.58 s and `variance_decomposition` is 1.85 s at the production replicate count, because the reports draw 1x to 14x the replicate count. The family figure is their mean and the low/high bounds are the extremes; the per-report table is in `BENCHMARK.json` under `measured_tables.cluster_report_by_report`.
- **The CLI harness around the kernel costs 4.57 ms per cell, 1.8 CPU-s over the package, 0.77% of its total.** A production cell is not a kernel call. `common/cli.py` appends a charged `running` record to the execution ledger under an advisory `flock`, runs the kernel inside the fit-count instrumentation, checks the cell schema and the three fit counts, writes the result atomically (json encode + `fsync` + sha256) and appends the completion record. None of that is in the kernel benchmarks. It was measured end to end by running 29 fixture cells through the real CLI (4.56 ms per cell there) and re-priced at the production ledger length (395 records, 0.353 ms per cell) and the production cell-file size (611 bytes, 0.629 ms to write). Of the total, 0.139 CPU-s is ledger append, which every worker serialises through one lock: that part is a floor on the wall time at any worker count. The end-to-end figure also carries the once-per-run costs amortised over 29 fixture cells, so it overstates rather than understates.

---

## Method and reproduction

```
python3 experiments/mlst2026/successor_20260913/run_benchmarks.py \
    --ms-evidence <MS>/evidence/data \
    [--handoff-dir <extracted handoff> --handoff-tar <archive>]
python3 -m experiments.mlst2026.successor_20260913.<pkg> estimate \
    --protocol <PROTOCOL> --out <FIXTURE RUN ROOT>
python3 experiments/mlst2026/successor_20260913/make_resource_report.py
```

`--check` on the last command regenerates this file and diffs it against what is on disk; it is run by `tests/test_successor_20260913_common.py`, so a benchmark re-run that is not followed by a re-render fails the test suite rather than leaving stale prose beside fresh tables.

The measurement harness is `experiments/mlst2026/successor_20260913/common/benchmark.py`: `timed` (warm-up discarded, median and min/max over repeats), `peak_memory` and `probe_peak_memory_subprocess` (tracemalloc plus the process's `VmHWM`, in a fresh process so the high-water mark belongs to one size - `VmHWM` and not `ru_maxrss`, which Linux inherits across `fork` and which therefore reported the benchmark driver's own peak for every probe until this pass), `fit_power_law` (least squares on the logs, returning `a`, `b` and the residual), `exponent_verdict` (fitted against theoretical, with the disagreement stated), `extrapolation` (which labels anything spanning more than a decade in the dimension as a projection) and `wall_time_range` (parallel seconds / W plus the serial ledger and reduce terms). The package benchmarks took 12 minutes on the hardware above, plus about three more for the transfer and setup measurements.

`cli._estimate` reads each package's `BENCHMARK.json`, multiplies by the enumerated production matrix, adds the measured harness and reduction rates, and writes `RESOURCE_ESTIMATE.json`. If `BENCHMARK.json` is absent it says so (`measured_projection_available: false`) rather than falling back to a guess. `make_handoff.py` refuses to build a handoff whose estimate is not the measured one, whose harness is unmeasured, whose total is not cell compute + harness + reduction, or whose wall model has no serial ledger term; and a test regenerates each `RESOURCE_ESTIMATE.json` from the shipped `BENCHMARK.json` and requires them to match, because the handoff's own check is structural and cannot see a stale estimate.

