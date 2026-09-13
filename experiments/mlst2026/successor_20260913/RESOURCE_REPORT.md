# Resource report — MLST successor round (local build, 2026-09-13)

Generated 2026-09-13 09:47 UTC by `experiments/mlst2026/successor_20260913/make_resource_report.py` from the `BENCHMARK.json` and `RESOURCE_ESTIMATE.json` of each package. Every number below comes out of those files.

## What this replaces

The previous `RESOURCE_ESTIMATE.json` files multiplied one measured *fixture* wall time per family by a constant declared in the protocol (`budget.fixture_to_production_scale`). That is a linear extrapolation of a tiny fixture with no account of the production dimension and no account of the algorithmic scaling of the kernel. For t21s it was wrong by a factor of 64.

What is here instead: each package's dominant kernel, identified by reading its runner, benchmarked **with the package's own functions** at four or more problem sizes that bracket or reach the production dimension; the scaling fitted as `t = a n^b` by least squares on the logs and printed beside the exponent the algorithm predicts; the per-cell seconds multiplied by the enumerated production matrix. Reduction work, peak memory, output storage, transfer size and setup time are measured and reported separately from cell compute.

**Every number is labelled `measured` or `projected`.** A projected one names the fitted exponent it used and the extrapolation factor in the dimension. In this round the production dimension turned out to be reachable for every package, so almost everything below is a measurement; the one substitution (CRESST raw pulse traces, which are not in this container) is called out where it occurs and is labelled `measured-on-substitute`.

## Benchmark hardware

```
Intel(R) Xeon(R) Processor @ 2.10GHz, 2 vCPU (2 cores, 1 thread/core), 7 GiB RAM
Linux container; python 3.11.15, numpy 2.4.4, scipy-openblas 0.3.31.188.0
Benchmarks run single-threaded: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Summary

| package | cells | fits | solves | CPU-s (low - point - high) | wall h @1 | wall h @8 | wall h @32 | peak RSS/cell | output | flat cross-check CPU-s | measured/flat |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `t19s` | 136 | 3840 | 3605 | 858.259 - **864.734** - 867.253 | 0.24 | 0.0319 | 0.00958 | 183 MB | 10.6 MB | 817.446 | 1.05 |
| `t20s` | 964 | 4800 | 140 | 6875.06 - **6900.37** - 6963.58 | 1.92 | 0.244 | 0.0643 | 153 MB | 4.0 MB | 7329.77 | 0.939 |
| `t21s` | 2499 | 2496 | 2498 | 92.6771 - **93.1675** - 95.4264 | 0.0259 | 0.00718 | 0.00517 | 58.9 MB | 15.8 MB | 4891.29 | 0.0157 |
| `t22s` | 727 | 0 | 0 | 342240 - **343137** - 345249 | 95.3 | 11.9 | 2.98 | 45.9 MB | 723.3 MB | 378823 | 0.906 |
| `cresst_s` | 395 | 1056 | 0 | 208.605 - **231.173** - 266.747 | 0.0642 | 0.00818 | 0.00218 | 170 MB | 2.0 MB | 583.973 | 0.395 |
| **all five** | 4721 | 12192 | 6243 | **351226** (= 97.56 CPU-hours) | 97.56 | 12.20 | 3.05 | | | | |

The all-five wall-time cells divide the summed CPU seconds by the worker count and are therefore optimistic across packages: they assume the five packages are run together on one pool. Per package, use the rows above, which add each package's serial reduction back in.

### The historical 6-8 hour allocation

This round's honest total is **97.6 CPU-hours**, of which 95.3 is T22-S alone and 2.25 is the other four packages combined. That is neither the historical 6-8 hour allocation nor anything near it, and the allocation was not used as an input anywhere in this document. At 8 workers the round is about 12 wall-hours; at 32 workers, about 3. Whether either worker count exists on the receiving machine is not known here. The number is dominated by one package: if T22-S's pilot freezes the cheaper replicate setting, the round roughly halves.

## Transfer and setup (separate from computation)

- **Transfer.** The handoff archive is **7.41 MB** (`/tmp/hf3/mlst-successor_20260913-handoff-20260913T093439Z.tar.gz`). The transfer *time* depends on the link, which is unknown; only the size is a fact. Successive builds of the same handoff differ by a few kB (timestamps and this report).
- **`verify_handoff.py`** on the extracted archive: 3.61-3.63 s (manifest, environment, validate and enumerate for all five packages; computes no cells). MEASURED, 2 runs.
- **`verify_handoff.py --tests`**: 67.68-68.09 s. MEASURED, 2 runs. This is the receiving side's first real step and it is a bounded, one-off cost.
- **Environment.** The pinned requirements are `h5py==3.16.0`, `numpy==2.4.4`, `pytest==9.1.1`, `scipy==1.17.1`. This container **already satisfies them**, so no real `pip install` was timed - that would have measured a no-op. The honest figures are the import check, 0.147 s, and a `pip install --dry-run --no-deps` resolution, 0.532 s, both MEASURED. On a server that does *not* have them, the cost is the download and install of the numpy and scipy wheels (of order 90 MB) plus h5py and pytest; that is link- and index-dependent and is **unknown** from here.

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
  and the execution ledger is appended under an advisory `flock`. On this
  container `reduce` was compute-bound at well under 4 MB/s of input, so
  filesystem speed did not matter here; on a network filesystem the advisory lock
  may not hold and the ledger falls back to serial appends, which caps useful
  parallelism at a point this round cannot compute.
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
| 32 | 0.0073903 | 0.0073078 | 0.0073962 | 3 | - | - | - |
| 64 | 0.017838 | 0.017746 | 0.017845 | 3 | - | - | - |
| 128 | 0.066775 | 0.066583 | 0.067149 | 3 | - | - | - |
| 256 | 0.30207 | 0.30137 | 0.30545 | 3 | 6.35 MB | 183 MB | this process |

Scaling fit `t = a * d^b` (least squares on the logs): **a = 1.234e-05**, **b = 1.796**, residual (log RMS) 0.1595, worst |log residual| 0.1945, R² 0.9870 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **1.796** |
| theoretical (eigendecomposition of a d x d matrix is O(d^3)) | **3** |

**Disagreement:** the measured exponent is BELOW the algorithm's by 1.20: at these sizes the kernel is not yet flop-bound (Python/allocation overhead dominates), so a projection made with the fitted exponent UNDERSTATES the cost at larger sizes and one made with the theoretical exponent overstates it at these ones.

Production value: **measured** — measured at the production size.

**Kernel `deterministic_cell_2d`**

| n | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0.0017657 | 0.0017144 | 0.0017998 | 3 | - | - | - |

Production value: **measured** — measured at the production size.

**Kernel `paired_cell`** — `t19s.synthetic.run_paired_cell (32 cw_pca fits + 29 eigen solves per cell; the m/d = 2048 noise-only calibration stream of 2048*d rows dominates)`

> N is held at the production aspect ratio N = 2d so the d = 256 point IS one production paired cell

| d | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 0.097511 | 0.095664 | 0.098701 | 3 | - | - | - |
| 64 | 0.33455 | 0.31553 | 0.34897 | 3 | - | - | - |
| 128 | 1.4012 | 1.393 | 1.4246 | 3 | - | - | - |
| 256 | 7.104 | 7.0502 | 7.1246 | 3 | 83.4 MB | 183 MB | subprocess |

Scaling fit `t = a * d^b` (least squares on the logs): **a = 6.947e-05**, **b = 2.063**, residual (log RMS) 0.0976, worst |log residual| 0.0989, R² 0.9963 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **2.063** |
| theoretical (eigendecomposition of a d x d matrix is O(d^3)) | **3** |

**Disagreement:** the measured exponent is BELOW the algorithm's by 0.94: at these sizes the kernel is not yet flop-bound (Python/allocation overhead dominates), so a projection made with the fitted exponent UNDERSTATES the cost at larger sizes and one made with the theoretical exponent overstates it at these ones.

Production value: **measured** — measured at the production size.

### Projection arithmetic

| family | cells | s/cell (median) | s/cell (min-max) | basis | fitted b | theory b | extrap. factor in dimension | cells x s/cell = CPU-s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| `ctrl_additive_generic` | 3 | 0.30207 | 0.30137 - 0.30545 | measured | 1.8 | 3 | 1 | 3 x 0.30207 = **0.906203** |
| `ctrl_additive_pure_coupling` | 3 | 0.30207 | 0.30137 - 0.30545 | measured | 1.8 | 3 | 1 | 3 x 0.30207 = **0.906203** |
| `ctrl_block_diagonal` | 2 | 0.30207 | 0.30137 - 0.30545 | measured | 1.8 | 3 | 1 | 2 x 0.30207 = **0.604136** |
| `ctrl_zero` | 1 | 0.30207 | 0.30137 - 0.30545 | measured | 1.8 | 3 | 1 | 1 x 0.30207 = **0.302068** |
| `det_generic` | 3 | 0.30207 | 0.30137 - 0.30545 | measured | 1.8 | 3 | 1 | 3 x 0.30207 = **0.906203** |
| `det_pure_coupling` | 3 | 0.30207 | 0.30137 - 0.30545 | measured | 1.8 | 3 | 1 | 3 x 0.30207 = **0.906203** |
| `det_two_d` | 1 | 0.0017657 | 0.0017144 - 0.0017998 | measured | - | - | - | 1 x 0.0017657 = **0.00176574** |
| `paired` | 120 | 7.104 | 7.0502 - 7.1246 | measured | 2.06 | 3 | 1 | 120 x 7.104 = **852.486** |
| **cell compute** | 136 | | | | | | | **857.019** (range 850.544 - 859.538) |
| **reduction** | | | | measured rate, projected | | | | **7.715** |
| **total** | | | | | | | | **864.734** (range 858.259 - 867.253) |

Per-cell figures are the measured median over repeats where a family's cells are identical, and the mean over the family's own cells of their measured medians where they are not. The low/high columns are the corresponding measured extremes: they are measurements, not margins.

**Reduction work.** Measured on the fixture run root: 0.5935 s of reduce work over 12 cells = **0.04946 s per cell** (interpreter and import, 0.114 s, subtracted). Measured input throughput 0.52 MB/s against a warm page cache, so the step is **compute-bound**, not I/O-bound: it reads every cell file once but spends its time re-deriving summaries and bootstrapping. Priced as 20 cells (pilot reduce) x 0.04946 s = 0.9891 s + 136 cells (final reduce over the whole run root) x 0.04946 s = 6.726 s = **7.715 s**.

### Wall time

cells are independent processes that share only the append-only ledger (advisory flock), so the cell compute is taken as perfectly parallel over W workers; the reduction is a single process and is added serially.  wall = cell_compute / W + reduction.

| workers | wall hours (low - high) | status |
|---:|---|---|
| 1 | 0.2384 - 0.2409 | CONDITIONAL on 1 usable cores; the server specification is not known |
| 8 | 0.03168 - 0.03199 | CONDITIONAL on 8 usable cores; the server specification is not known |
| 32 | 0.009526 - 0.009604 | CONDITIONAL on 32 usable cores; the server specification is not known |

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

The old estimate (one fixture wall time x `budget.fixture_to_production_scale`) gives **817.446 CPU-s**. The measured cell compute is **857.019 CPU-s**. Ratio measured/flat = **1.048**. the declared flat scale is within a factor 2 of the measurement.

### Findings

- **The production paired cell was run directly, so the paired number is a measurement.** `d = 256` with `N = 2d = 512` is the production configuration and it is one of the four benchmarked sizes. Nothing about the 120 paired cells is extrapolated; only the multiplication by 120 is arithmetic.
- **The fitted exponent (2.06) is well below the algorithmic one (3).** This is a statement about the *small* end of the benchmark, not about the production point. At `d = 32` the cell is 0.098 s of mostly Python and allocation; at `d = 256` it is 7.1 s of BLAS. A power law fitted across that range averages the two regimes and lands near 2. Because the production size was measured, the fit is used for description only - no projection uses it.
- **The protocol's own scale basis names the right dominant term.** `budget.scale_basis` already said the `m = 2048 d` calibration stream dominates, and the measurement agrees. The declared factor 200 for the paired family also lands close: the flat cross-check gives 817 CPU-s against a measured 857 CPU-s of cell compute. t19s is the one package whose declared flat scale was about right.

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
| 1 | 2.2801 | 2.2171 | 3.4707 | 3 | - | - | - |

Production value: **measured** — measured at the production size.

**Kernel `fit_cell`** — `t20s.runner.run_fit_cell -> np.linalg.eigh(S) (1 PCA fit) + experiments.synthetic.nfpa_confirmatory.methods.als_fit (R = 4 restarts x 150 sweeps)`

> d = C.T = 256 is fixed by the design; N is the only production size that varies, and every production N was run

| N | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 400 | 1.9615 | 1.9443 | 1.9737 | 3 | - | - | - |
| 800 | 3.8216 | 3.8144 | 3.8439 | 3 | - | - | - |
| 1000 | 4.8159 | 4.7678 | 4.8932 | 3 | - | - | - |
| 1250 | 6.6934 | 6.6722 | 6.725 | 3 | - | - | - |
| 1600 | 8.7393 | 8.7175 | 8.8937 | 3 | - | - | - |
| 3200 | 16.937 | 16.896 | 17.004 | 3 | 29 MB | 153 MB | subprocess |

Scaling fit `t = a * N^b` (least squares on the logs): **a = 0.003419**, **b = 1.057**, residual (log RMS) 0.0403, worst |log residual| 0.0500, R² 0.9964 over 6 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **1.057** |
| theoretical (an ALS sweep is O(iter . N . d . k), so linear in N at fixed d, k, iter) | **1** |

Agreement: the measured exponent agrees with the algorithm within +/- 0.35.

Production value: **measured** — measured at the production size.

### Projection arithmetic

| family | cells | s/cell (median) | s/cell (min-max) | basis | fitted b | theory b | extrap. factor in dimension | cells x s/cell = CPU-s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| `algebraic` | 4 | 2.2801 | 2.2171 - 3.4707 | measured | - | - | 1 | 4 x 2.2801 = **9.12048** |
| `confirmation` | 576 | 7.1615 | 7.1354 - 7.2223 | measured | 1.06 | 1 | 1 | 576 x 7.1615 = **4125** |
| `pilot` | 384 | 7.1615 | 7.1354 - 7.2223 | measured | 1.06 | 1 | 1 | 384 x 7.1615 = **2750** |
| **cell compute** | 964 | | | | | | | **6884.12** (range 6858.81 - 6947.33) |
| **reduction** | | | | measured rate, projected | | | | **16.25** |
| **total** | | | | | | | | **6900.37** (range 6875.06 - 6963.58) |

Per-cell figures are the measured median over repeats where a family's cells are identical, and the mean over the family's own cells of their measured medians where they are not. The low/high columns are the corresponding measured extremes: they are measurements, not margins.

**Reduction work.** Measured on the fixture run root: 0.217 s of reduce work over 18 cells = **0.01206 s per cell** (interpreter and import, 0.84 s, subtracted). Measured input throughput 0.35 MB/s against a warm page cache, so the step is **compute-bound**, not I/O-bound: it reads every cell file once but spends its time re-deriving summaries and bootstrapping. Priced as 384 cells (pilot reduce) x 0.01206 s = 4.63 s + 964 cells (final reduce over the whole run root) x 0.01206 s = 11.62 s = **16.25 s**.

### Wall time

cells are independent processes that share only the append-only ledger (advisory flock), so the cell compute is taken as perfectly parallel over W workers; the reduction is a single process and is added serially.  wall = cell_compute / W + reduction.

| workers | wall hours (low - high) | status |
|---:|---|---|
| 1 | 1.91 - 1.934 | CONDITIONAL on 1 usable cores; the server specification is not known |
| 8 | 0.2427 - 0.2457 | CONDITIONAL on 8 usable cores; the server specification is not known |
| 32 | 0.06405 - 0.06482 | CONDITIONAL on 32 usable cores; the server specification is not known |

### Peak memory and output storage

Peak memory per cell at the production dimension: **153 MB process RSS** (`fit_cell` at 3200, measured in a subprocess), of which 29 MB is Python-allocator peak for the call itself. The RSS figure includes the interpreter and numpy, so it is what one worker actually needs. MEASURED.

| item | cells | bytes/cell | total | basis |
|---|---:|---:|---:|---|
| `algebraic` | 4 | 1,835 | 7.34 kB | measured |
| `confirmation` | 576 | 1,409 | 812 kB | measured |
| `pilot` | 384 | 1,409 | 541 kB | measured |
| execution ledger | 964 | | 1.83 MB | projected from a measured per-cell rate |
| `REDUCED.json` | | | 827 kB | projected linearly from the fixture |
| **total** | | | **4.0 MB** | |

### Cross-check against the protocol's flat scale factor

The old estimate (one fixture wall time x `budget.fixture_to_production_scale`) gives **7329.77 CPU-s**. The measured cell compute is **6884.12 CPU-s**. Ratio measured/flat = **0.9392**. the declared flat scale is within a factor 2 of the measurement.

### Findings

- **Every production `N` was run, so the whole fit family is measured.** The six-point `N` grid is the production grid; the per-cell figure is the unweighted mean of the six measured medians, which is exactly how the cells are distributed.
- **The fitted exponent (1.06) agrees with the ALS sweep cost `O(iter . N . d . k)`.** This is the one package where the benchmark is squarely in the flop-bound regime across the whole measured range.
- **The measured per-cell time excludes a per-worker cost the protocol never mentions.** `run_fit_cell` memoises the population reference per `(process, theta)` with an LRU cache. The benchmark discards a warm-up, so the quoted per-cell times are warm-cache times. The cold reference costs 0.594 s (measured). A long-lived worker pays it four times in total (one per theta). A scheduler that starts a fresh process per cell pays it on all 960 fit cells, which adds about 570 CPU-seconds, i.e. about 8% on top of the total. This is recorded in `work_accounting.per_worker_cold_population_reference_seconds_measured`.

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
| `geometry_invariants` | 256 | 0.0034659 | 0.003338 | 0.0039966 | 5 | - | - | - |
| `predictor_sanity` | 256 | 0.0032926 | 0.0032049 | 0.0033557 | 5 | - | - | - |

Production value: **measured** — measured at the production size.

**Kernel `data_cell_hi`** — `t21s.runner.run_data_cell -> compute_predictions (eigh of the d x d population K, one solve) then fit_pca (thin SVD of the centred N x d draw, one fit)`

> N is held at the production aspect ratio for the hi arm (gamma = d/N fixed), so the d = 256 point IS a production hi-arm cell

| d | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 24 | 0.0011637 | 0.0010763 | 0.0012166 | 5 | - | - | - |
| 64 | 0.0019121 | 0.0018777 | 0.0020333 | 5 | - | - | - |
| 128 | 0.0048109 | 0.0046728 | 0.0050952 | 5 | - | - | - |
| 256 | 0.015369 | 0.015252 | 0.015548 | 5 | 2.36 MB | 47.9 MB | subprocess |

Scaling fit `t = a * d^b` (least squares on the logs): **a = 2.902e-05**, **b = 1.086**, residual (log RMS) 0.2516, worst |log residual| 0.3292, R² 0.9345 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **1.086** |
| theoretical (thin SVD of an N x d matrix is O(N d min(N, d)); at fixed aspect ratio that is O(d^3)) | **3** |

**Disagreement:** the measured exponent is BELOW the algorithm's by 1.91: at these sizes the kernel is not yet flop-bound (Python/allocation overhead dominates), so a projection made with the fitted exponent UNDERSTATES the cost at larger sizes and one made with the theoretical exponent overstates it at these ones.

Production value: **measured** — measured at the production size.

**Kernel `data_cell_lo`** — `t21s.runner.run_data_cell -> compute_predictions (eigh of the d x d population K, one solve) then fit_pca (thin SVD of the centred N x d draw, one fit)`

> N is held at the production aspect ratio for the lo arm (gamma = d/N fixed), so the d = 256 point IS a production lo-arm cell

| d | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 24 | 0.0012351 | 0.0011949 | 0.0012561 | 5 | - | - | - |
| 64 | 0.0031601 | 0.003083 | 0.0032981 | 5 | - | - | - |
| 128 | 0.011503 | 0.011356 | 0.011557 | 5 | - | - | - |
| 256 | 0.046277 | 0.046002 | 0.047907 | 5 | 7.57 MB | 58.9 MB | subprocess |

Scaling fit `t = a * d^b` (least squares on the logs): **a = 7.448e-06**, **b = 1.536**, residual (log RMS) 0.2377, worst |log residual| 0.3373, R² 0.9697 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **1.536** |
| theoretical (thin SVD of an N x d matrix is O(N d min(N, d)); at fixed aspect ratio that is O(d^3)) | **3** |

**Disagreement:** the measured exponent is BELOW the algorithm's by 1.46: at these sizes the kernel is not yet flop-bound (Python/allocation overhead dominates), so a projection made with the fitted exponent UNDERSTATES the cost at larger sizes and one made with the theoretical exponent overstates it at these ones.

Production value: **measured** — measured at the production size.

### Projection arithmetic

| family | cells | s/cell (median) | s/cell (min-max) | basis | fitted b | theory b | extrap. factor in dimension | cells x s/cell = CPU-s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| `algebraic` | 3 | 0.0034081 | 0.0032049 - 0.0039966 | measured | - | - | 1 | 3 x 0.0034081 = **0.0102244** |
| `confirmation-correlated-hi` | 520 | 0.015369 | 0.015252 - 0.015548 | measured | 1.09 | 3 | 1 | 520 x 0.015369 = **7.99205** |
| `confirmation-correlated-lo` | 520 | 0.046277 | 0.046002 - 0.047907 | measured | 1.54 | 3 | 1 | 520 x 0.046277 = **24.0641** |
| `confirmation-orthogonal-hi` | 520 | 0.015369 | 0.015252 - 0.015548 | measured | 1.09 | 3 | 1 | 520 x 0.015369 = **7.99205** |
| `confirmation-orthogonal-lo` | 520 | 0.046277 | 0.046002 - 0.047907 | measured | 1.54 | 3 | 1 | 520 x 0.046277 = **24.0641** |
| `pilot-correlated-hi` | 104 | 0.015369 | 0.015252 - 0.015548 | measured | 1.09 | 3 | 1 | 104 x 0.015369 = **1.59841** |
| `pilot-correlated-lo` | 104 | 0.046277 | 0.046002 - 0.047907 | measured | 1.54 | 3 | 1 | 104 x 0.046277 = **4.81281** |
| `pilot-orthogonal-hi` | 104 | 0.015369 | 0.015252 - 0.015548 | measured | 1.09 | 3 | 1 | 104 x 0.015369 = **1.59841** |
| `pilot-orthogonal-lo` | 104 | 0.046277 | 0.046002 - 0.047907 | measured | 1.54 | 3 | 1 | 104 x 0.046277 = **4.81281** |
| **cell compute** | 2499 | | | | | | | **76.9449** (range 76.4546 - 79.2038) |
| **reduction** | | | | measured rate, projected | | | | **16.22** |
| **total** | | | | | | | | **93.1675** (range 92.6771 - 95.4264) |

Per-cell figures are the measured median over repeats where a family's cells are identical, and the mean over the family's own cells of their measured medians where they are not. The low/high columns are the corresponding measured extremes: they are measurements, not margins.

**Reduction work.** Measured on the fixture run root: 2.332 s of reduce work over 419 cells = **0.005565 s per cell** (interpreter and import, 0.11 s, subtracted). Measured input throughput 0.83 MB/s against a warm page cache, so the step is **compute-bound**, not I/O-bound: it reads every cell file once but spends its time re-deriving summaries and bootstrapping. Priced as 416 cells (pilot reduce) x 0.005565 s = 2.315 s + 2499 cells (final reduce over the whole run root) x 0.005565 s = 13.91 s = **16.22 s**.

### Wall time

cells are independent processes that share only the append-only ledger (advisory flock), so the cell compute is taken as perfectly parallel over W workers; the reduction is a single process and is added serially.  wall = cell_compute / W + reduction.

| workers | wall hours (low - high) | status |
|---:|---|---|
| 1 | 0.02574 - 0.02651 | CONDITIONAL on 1 usable cores; the server specification is not known |
| 8 | 0.007161 - 0.007256 | CONDITIONAL on 8 usable cores; the server specification is not known |
| 32 | 0.00517 - 0.005194 | CONDITIONAL on 32 usable cores; the server specification is not known |

### Peak memory and output storage

Peak memory per cell at the production dimension: **58.9 MB process RSS** (`data_cell_lo` at 256, measured in a subprocess), of which 7.57 MB is Python-allocator peak for the call itself. The RSS figure includes the interpreter and numpy, so it is what one worker actually needs. MEASURED.

| item | cells | bytes/cell | total | basis |
|---|---:|---:|---:|---|
| `algebraic` | 3 | 963 | 2.89 kB | measured |
| `confirmation-correlated-hi` | 520 | 2,583 | 1.34 MB | measured |
| `confirmation-correlated-lo` | 520 | 2,519 | 1.31 MB | measured |
| `confirmation-orthogonal-hi` | 520 | 2,583 | 1.34 MB | measured |
| `confirmation-orthogonal-lo` | 520 | 2,519 | 1.31 MB | measured |
| `pilot-correlated-hi` | 104 | 2,583 | 269 kB | measured |
| `pilot-correlated-lo` | 104 | 2,519 | 262 kB | measured |
| `pilot-orthogonal-hi` | 104 | 2,583 | 269 kB | measured |
| `pilot-orthogonal-lo` | 104 | 2,519 | 262 kB | measured |
| execution ledger | 2499 | | 4.75 MB | projected from a measured per-cell rate |
| `REDUCED.json` | | | 4.65 MB | projected linearly from the fixture |
| **total** | | | **15.8 MB** | |

### Cross-check against the protocol's flat scale factor

The old estimate (one fixture wall time x `budget.fixture_to_production_scale`) gives **4891.29 CPU-s**. The measured cell compute is **76.9449 CPU-s**. Ratio measured/flat = **0.01573**. the declared flat scale disagrees with the measurement by a factor 63.6; the flat factor is wrong, not the measurement.

### Findings

- **The declared flat scale overstates the cost by a factor of about 64. This is the largest single correction in this round.** `budget.fixture_to_production_scale = 1214` is the ratio of the PCA flop counts `N d min(N,d)` between the production and the fixture cell. Multiplied by a fixture cell of about 1.6 ms it gives about 2.0 s per production cell and 4 891 CPU-s for the package. The measured production cells are 0.0154 s (`hi`) and 0.0463 s (`lo`), giving 77 CPU-s of cell compute. The protocol's own note called the flat factor an upper bound; it is an upper bound by 64x, which is not a useful bound.
- **The dominant cost is not the estimator the flat factor was derived from.** The scale was derived from the PCA flop count. At `d = 256, N = 128` the thin SVD of a 128x256 matrix is a few hundred microseconds; the cell takes 15 ms. The time is in `compute_predictions` - the `d x d` eigendecomposition of `K`, the QR of the direction frame, and the per-direction predictor arithmetic - all of which are independent of `N` and were not in the flop ratio at all. The `hi`/`lo` split confirms it: tripling `N` from 128 to 1024 triples the cell time, so the `N`-dependent part is real but it sits on top of a large `N`-independent base.
- **The fitted exponents (1.09 `hi`, 1.54 `lo`) are far below the SVD's 3.** Again a small-size effect, and again the production size was measured so nothing is projected with the fitted exponent. It does say that a flop-ratio argument between `d = 24` and `d = 256` cannot be trusted for this package, which is precisely how the 64x error arose.
- **At 77 CPU-seconds of cell compute, reduce is a fifth of the package's cost.** The measured reduce rate is 5.57 ms per cell over 419 fixture cells; over 2 499 production cells and two reduces (pilot and final) that is 16.2 s against 77 s of cell compute. For this package the reduction is no longer a rounding error, and an estimate that omitted it (the previous one did) would be materially wrong.

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
| `ar1_autocorrelation` | 1 | 0.10448 | 0.10249 | 0.10647 | 2 | - | - | - |
| `dependence_gate` | 1 | 5.3442 | 5.2861 | 5.4024 | 2 | - | - | - |
| `holm_arithmetic` | 1 | 5.4722e-05 | 2.238e-05 | 8.7064e-05 | 2 | - | - | - |
| `iid_bootstrap_se` | 1 | 0.58833 | 0.58076 | 0.5959 | 2 | - | - | - |
| `simulator_estimand` | 1 | 0.040415 | 0.038817 | 0.042013 | 2 | - | - | - |

Production value: **measured** — measured at the production configuration.

**Kernel `pipeline_replicate`** — `t22s.pipeline.run_replicate: te.joint_bootstrap per evaluation file (R_file), t22.combine_stage two-stage hierarchical bootstrap (R_hier = 5 R_file) and te.block_variance_curve on the development files (R_curve = R_file/5)`

> the three replicate counts are moved together at the archived ratio 1 : 5 : 0.2, so the size axis is the whole replicate budget of one Monte-Carlo replicate; the production (archived) setting was run directly

| R_file | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 60 | 0.093693 | 0.091952 | 0.096941 | 3 | - | - | - |
| 250 | 0.34048 | 0.33524 | 0.3427 | 3 | - | - | - |
| 1000 | 1.3839 | 1.358 | 1.3919 | 3 | - | - | - |
| 2000 | 2.6454 | 2.6433 | 2.6616 | 3 | 2.53 MB | 45.9 MB | subprocess |

Scaling fit `t = a * R_file^b` (least squares on the logs): **a = 0.001798**, **b = 0.959**, residual (log RMS) 0.0310, worst |log residual| 0.0518, R² 0.9994 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **0.959** |
| theoretical (a bootstrap is linear in the number of replicates) | **1** |

Agreement: the measured exponent agrees with the algorithm within +/- 0.35.

Production value: **measured** — measured at the production size.

**Kernel `replicate_by_n_files`**

| n_files | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 2.5775 | 2.5645 | 2.5859 | 3 | - | - | - |
| 8 | 4.141 | 4.124 | 4.1495 | 3 | - | - | - |
| 16 | 7.4394 | 7.4299 | 7.5026 | 3 | - | - | - |

Scaling fit `t = a * n_files^b` (least squares on the logs): **a = 0.8766**, **b = 0.765**, residual (log RMS) 0.0263, worst |log residual| 0.0373, R² 0.9963 over 3 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **0.765** |
| theoretical (the joint bootstrap is run once per evaluation file, so the replicate is linear in n_files above a fixed simulation cost) | **1** |

Agreement: the measured exponent agrees with the algorithm within +/- 0.35.

Production value: **measured** — every production n_files (4, 8, 16) was run.

### Projection arithmetic

| family | cells | s/cell (median) | s/cell (min-max) | basis | fitted b | theory b | extrap. factor in dimension | cells x s/cell = CPU-s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| `control` | 5 | 1.2155 | 1.2016 - 1.2294 | measured | - | - | 1 | 5 x 1.2155 = **6.0775** |
| `sim_confirm` | 690 | 481.99 | 480.74 - 484.97 | measured | 0.959 | 1 | 1 | 690 x 481.99 = **332573** |
| `sim_pilot` | 32 | 329.8 | 328.77 - 331.47 | measured | 0.959 | 1 | 1 | 32 x 329.8 = **10553.6** |
| **cell compute** | 727 | | | | | | | **343132** (range 342236 - 345245) |
| **reduction** | | | | measured rate, projected | | | | **4.238** |
| **total** | | | | | | | | **343137** (range 342240 - 345249) |

Per-cell figures are the measured median over repeats where a family's cells are identical, and the mean over the family's own cells of their measured medians where they are not. The low/high columns are the corresponding measured extremes: they are measurements, not margins.

**Reduction work.** Measured on the fixture run root: 0.07817 s of reduce work over 14 cells = **0.005584 s per cell** (interpreter and import, 0.115 s, subtracted). Measured input throughput 3.87 MB/s against a warm page cache, so the step is **compute-bound**, not I/O-bound: it reads every cell file once but spends its time re-deriving summaries and bootstrapping. Priced as 32 cells (pilot reduce) x 0.005584 s = 0.1787 s + 727 cells (final reduce over the whole run root) x 0.005584 s = 4.059 s = **4.238 s**.

### Wall time

cells are independent processes that share only the append-only ledger (advisory flock), so the cell compute is taken as perfectly parallel over W workers; the reduction is a single process and is added serially.  wall = cell_compute / W + reduction.

| workers | wall hours (low - high) | status |
|---:|---|---|
| 1 | 95.07 - 95.9 | CONDITIONAL on 1 usable cores; the server specification is not known |
| 8 | 11.88 - 11.99 | CONDITIONAL on 8 usable cores; the server specification is not known |
| 32 | 2.972 - 2.998 | CONDITIONAL on 32 usable cores; the server specification is not known |

### Peak memory and output storage

Peak memory per cell at the production dimension: **45.9 MB process RSS** (`pipeline_replicate` at 2000, measured in a subprocess), of which 2.53 MB is Python-allocator peak for the call itself. The RSS figure includes the interpreter and numpy, so it is what one worker actually needs. MEASURED.

| item | cells | bytes/cell | total | basis |
|---|---:|---:|---:|---|
| `control` | 5 | 964 | 4.82 kB | measured |
| `sim_confirm` | 690 | 998,378 | 689 MB | projected |
| `sim_pilot` | 32 | 928,862 | 29.7 MB | projected |
| execution ledger | 727 | | 1.38 MB | projected from a measured per-cell rate |
| `REDUCED.json` | | | 3.29 MB | projected linearly from the fixture |
| **total** | | | **723.3 MB** | |

### Cross-check against the protocol's flat scale factor

The old estimate (one fixture wall time x `budget.fixture_to_production_scale`) gives **378823 CPU-s**. The measured cell compute is **343132 CPU-s**. Ratio measured/flat = **0.9058**. the declared flat scale is within a factor 2 of the measurement.

### Findings

- **The production replicate setting was run directly.** `file 2000 / hier 10000 / curve 400` is one of the four benchmarked replicate budgets, and all three production `n_files` values (4, 8, 16) were run at it. Only the multiplication by `mc = 100` and by the cell count is arithmetic.
- **This package is 98% of the whole round's CPU cost, and that survives the correction.** 343 137 CPU-s (95.3 CPU-hours) against 8 089 CPU-s (2.25 CPU-hours) for the other four packages combined. The flat cross-check said 378 823 CPU-s; the measurement is 9% lower. The order of magnitude was right, so unlike t21s the flat factor here was not badly wrong - but it was still a product of a synthetic work proxy, and the measurement replaces it.
- **The confirmation total is an upper bound, for a reason the protocol states.** The confirmation cells run the replicate setting the pilot freezes, which is not known before the pilot. Pricing uses the archived setting, the more expensive of the two. The measured ratio reduced/archived is 0.506 at `n_files = 4`, so if the pilot freezes `reduced` the confirmation cost roughly halves to about 48 CPU-hours.
- **Output storage, not CPU, is the constraint a receiving site is most likely to trip over.** The projected cell files come to 723 MB, essentially all of it the 690 confirmation cells at about 1.0 MB each (100 Monte-Carlo records per cell). That is 22x the other four packages combined (32 MB) and about 100x the transferred archive itself.

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
| 1200 | 0.21744 | 0.20816 | 0.22088 | 3 | - | - | - |
| 2500 | 0.4509 | 0.44694 | 0.46676 | 3 | - | - | - |
| 5000 | 0.8687 | 0.83097 | 0.87293 | 3 | - | - | - |
| 10000 | 1.8087 | 1.7255 | 1.8474 | 3 | 64.3 MB | 142 MB | subprocess |

Scaling fit `t = a * R^b` (least squares on the logs): **a = 0.0001882**, **b = 0.994**, residual (log RMS) 0.0167, worst |log residual| 0.0277, R² 0.9995 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **0.994** |
| theoretical (a bootstrap is linear in the number of replicates) | **1** |

Agreement: the measured exponent agrees with the algorithm within +/- 0.35.

Production value: **measured** — measured at the production size.

**Kernel `archive_point`**

| n | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.00024502 | 0.00024214 | 0.00025906 | 3 | - | - | - |

Production value: **measured** — measured at the production size.

**Kernel `calibration_coverage`** — `cresst_s.calibration.coverage: mc Monte-Carlo datasets x bootstrap_replicates cluster bootstrap draws each`

> at the reference design (n_runs 2, channels_per_run 2); the production mc was run directly

| mc | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 50 | 0.11703 | 0.1157 | 0.11754 | 3 | - | - | - |
| 100 | 0.23387 | 0.23356 | 0.2343 | 3 | - | - | - |
| 200 | 0.4685 | 0.46753 | 0.47164 | 3 | - | - | - |
| 400 | 0.93044 | 0.92983 | 0.93305 | 3 | 1.54 MB | 177 MB | this process |

Scaling fit `t = a * mc^b` (least squares on the logs): **a = 0.002365**, **b = 0.998**, residual (log RMS) 0.0020, worst |log residual| 0.0032, R² 1.0000 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **0.998** |
| theoretical (a bootstrap is linear in the number of replicates) | **1** |

Agreement: the measured exponent agrees with the algorithm within +/- 0.35.

Production value: **measured** — measured at the production size.

**Kernel `cluster_report`** — `cresst_s.runner._cell_cluster_report -> cluster_report.* (cluster bootstrap per report)`

> the production replicate count was run directly

| R | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1200 | 0.49909 | 0.494 | 0.50418 | 2 | - | - | - |
| 2500 | 1.0081 | 0.9704 | 1.0457 | 2 | - | - | - |
| 5000 | 1.9828 | 1.9289 | 2.0367 | 2 | - | - | - |
| 10000 | 3.7992 | 3.7089 | 3.8895 | 2 | 99.6 MB | 170 MB | this process |

Scaling fit `t = a * R^b` (least squares on the logs): **a = 0.0005563**, **b = 0.959**, residual (log RMS) 0.0055, worst |log residual| 0.0092, R² 0.9999 over 4 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **0.959** |
| theoretical (a bootstrap is linear in the number of replicates) | **1** |

Agreement: the measured exponent agrees with the algorithm within +/- 0.35.

Production value: **measured** — measured at the production size.

**Kernel `empca_robustness`** — `cresst_s.robustness_design.fit_and_score -> src.canonical.empca.classic_pca (1 fit) + empca (EMPCA_RESTARTS = 2 fits, niter = 25)`

> the real CRESST pulse traces are NOT in this container; the synthetic pulse root written by the package's own write_fixture_data_root is used AT THE PRODUCTION DIMENSIONS (n_events 400, n_samples 128, n_noise 400). The EMPCA cost is set by those dimensions and by the fixed niter, not by the values, but the substitution is a substitution and is labelled here.

| rank k | median s | min s | max s | repeats | tracemalloc peak | process peak RSS | RSS measured in |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0.30302 | 0.29753 | 0.305 | 3 | - | - | - |
| 4 | 0.41601 | 0.37514 | 0.41633 | 3 | - | - | - |
| 6 | 0.44288 | 0.44233 | 0.44317 | 3 | - | - | - |
| 8 | 0.52754 | 0.52686 | 0.53325 | 3 | - | - | - |
| 12 | 0.73654 | 0.73515 | 0.7446 | 3 | 3.9 MB | 170 MB | subprocess |

Scaling fit `t = a * rank k^b` (least squares on the logs): **a = 0.2128**, **b = 0.463**, residual (log RMS) 0.0671, worst |log residual| 0.0968, R² 0.9471 over 5 sizes.

| | exponent |
|---|---:|
| fitted (measured) | **0.463** |
| theoretical (a weighted-EM sweep is O(niter . n_train . n_components . k), so linear in the rank k) | **1** |

**Disagreement:** the measured exponent is BELOW the algorithm's by 0.54: at these sizes the kernel is not yet flop-bound (Python/allocation overhead dominates), so a projection made with the fitted exponent UNDERSTATES the cost at larger sizes and one made with the theoretical exponent overstates it at these ones.

Production value: **measured** — measured at the production size.

### Projection arithmetic

| family | cells | s/cell (median) | s/cell (min-max) | basis | fitted b | theory b | extrap. factor in dimension | cells x s/cell = CPU-s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| `archive_interval` | 10 | 1.9969 | 1.397 - 2.661 | measured | 0.994 | 1 | 1 | 10 x 1.9969 = **19.9687** |
| `archive_point` | 1 | 0.00024502 | 0.00024214 - 0.00025906 | measured | - | - | 1 | 1 x 0.00024502 = **0.000245021** |
| `calib_confirm` | 12 | 3.7788 | 3.7561 - 3.8015 | measured | 0.998 | 1 | 1 | 12 x 3.7788 = **45.3456** |
| `calib_control` | 1 | 7.2694 | 7.2411 - 7.2977 | measured | - | - | 1 | 1 x 7.2694 = **7.26937** |
| `calib_pilot` | 12 | 1.4212 | 1.4129 - 1.4295 | measured | 0.998 | 1 | 1 | 12 x 1.4212 = **17.0543** |
| `cluster_report` | 5 | 4.6238 | 1.9421 - 9.3104 | measured | 0.959 | 1 | 1 | 5 x 4.6238 = **23.119** |
| `robustness_confirm` | 315 | 0.31528 | 0.31321 - 0.32034 | measured-on-substitute | 0.463 | 1 | 1 | 315 x 0.31528 = **99.3128** |
| `robustness_pilot` | 35 | 0.47069 | 0.46695 - 0.47404 | measured-on-substitute | 0.463 | 1 | 1 | 35 x 0.47069 = **16.4743** |
| `science_control` | 4 | 0.49578 | 0.0019111 - 1.3431 | measured | - | - | 1 | 4 x 0.49578 = **1.98312** |
| **cell compute** | 395 | | | | | | | **230.528** (range 207.96 - 266.102) |
| **reduction** | | | | measured rate, projected | | | | **0.6456** |
| **total** | | | | | | | | **231.173** (range 208.605 - 266.747) |

Per-cell figures are the measured median over repeats where a family's cells are identical, and the mean over the family's own cells of their measured medians where they are not. The low/high columns are the corresponding measured extremes: they are measurements, not margins.

**Reduction work.** Measured on the fixture run root: 0.04236 s of reduce work over 29 cells = **0.001461 s per cell** (interpreter and import, 0.301 s, subtracted). Measured input throughput 4.28 MB/s against a warm page cache, so the step is **compute-bound**, not I/O-bound: it reads every cell file once but spends its time re-deriving summaries and bootstrapping. Priced as 47 cells (pilot reduce) x 0.001461 s = 0.06865 s + 395 cells (final reduce over the whole run root) x 0.001461 s = 0.577 s = **0.6456 s**.

### Wall time

cells are independent processes that share only the append-only ledger (advisory flock), so the cell compute is taken as perfectly parallel over W workers; the reduction is a single process and is added serially.  wall = cell_compute / W + reduction.

| workers | wall hours (low - high) | status |
|---:|---|---|
| 1 | 0.05795 - 0.0741 | CONDITIONAL on 1 usable cores; the server specification is not known |
| 8 | 0.0074 - 0.009419 | CONDITIONAL on 8 usable cores; the server specification is not known |
| 32 | 0.001985 - 0.002489 | CONDITIONAL on 32 usable cores; the server specification is not known |

### Peak memory and output storage

Peak memory per cell at the production dimension: **170 MB process RSS** (`empca_robustness` at 12, measured in a subprocess), of which 3.9 MB is Python-allocator peak for the call itself. The RSS figure includes the interpreter and numpy, so it is what one worker actually needs. MEASURED.

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

The old estimate (one fixture wall time x `budget.fixture_to_production_scale`) gives **583.973 CPU-s**. The measured cell compute is **230.528 CPU-s**. Ratio measured/flat = **0.3948**. the declared flat scale disagrees with the measurement by a factor 2.53; the flat factor is wrong, not the measurement.

### Findings

- **Every family except the robustness grid was benchmarked at its exact production configuration on the real archived inputs.** The 10 interval reconstructions, the 5 cluster reports, the 4 science controls, the calibration control and all 24 calibration designs were each run at their production replicate budgets. Those numbers are measurements of the production cell, not projections.
- **The robustness cells cannot be executed here at all, and the substitution is on the data, not on the dimensions.** They need raw CRESST pulse traces, which are not in this container and may not be sought here; `run_cell` raises `input not available` for every one of them. The benchmark therefore used the package's own `write_fixture_data_root` at the *production* dimensions (`n_events = 400`, `n_samples = 128`, `n_noise = 400`) and ran the real `fit_and_score`. EMPCA runs a fixed `niter = 25`, so the cost is set by the dimensions and not by the values; the numbers are labelled `measured-on-substitute` and would move linearly in `n_train` if a real channel has a different event count.
- **The EMPCA cost is only weakly a function of the rank: fitted exponent 0.46 against a theoretical 1.** An EM sweep is `O(niter . n_train . n_components . k)`, so the rank should be linear. It is not, because a robustness cell also pays a rank-independent base: the stationary PSD estimate, the real-FFT representation of every pulse, and the classic-PCA fit. A sixfold increase in the rank only takes the cell from 0.303 s to 0.737 s. The protocol's cost proxy (`RD.EMPCA_RESTARTS * SEC_PER_EMPCA_UNIT * n_train * n_comp * rank * niter`) has no such base term, which is why its rank-6 estimate (1.2-1.6 s/cell) is 3-4x the measured 0.32-0.47 s.
- **The declared flat scale overstates the package by a factor 2.5.** 584 CPU-s declared against 231 CPU-s measured. Most of the gap is the robustness grid (350 of the 395 cells).
- **The five cluster-report cells differ by a factor 5 among themselves.** `leave_one_channel_out` is 9.3 s and `variance_decomposition` is 1.9 s at 10 000 replicates, because the reports draw 1x to 14x the replicate count. The family figure is their mean and the low/high bounds are the extremes; the per-report table is in `BENCHMARK.json` under `measured_tables.cluster_report_by_report`.

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

The harness is `experiments/mlst2026/successor_20260913/common/benchmark.py`: `timed` (warm-up discarded, median and min/max over repeats), `peak_memory` and `probe_peak_memory_subprocess` (tracemalloc plus `ru_maxrss`, the latter in a fresh process so the high-water mark belongs to one size), `fit_power_law` (least squares on the logs, returning `a`, `b` and the residual), `exponent_verdict` (fitted against theoretical, with the disagreement stated) and `extrapolation` (which labels anything spanning more than a decade in the dimension as a projection). The whole benchmarking pass takes about 13 minutes on the hardware above.

`cli._estimate` reads each package's `BENCHMARK.json`, multiplies by the enumerated production matrix, adds the measured reduction rate, and writes `RESOURCE_ESTIMATE.json`. If `BENCHMARK.json` is absent it says so (`measured_projection_available: false`) rather than falling back to a guess, and `make_handoff.py` refuses to build a handoff whose estimate is not the measured one.

