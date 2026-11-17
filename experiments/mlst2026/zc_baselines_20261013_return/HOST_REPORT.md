# ZC completed confirmation host return

Host: deepthought. EXP freeze: `a3e625934e29fb3f4bf13976ab10a8c4f7ca54d7`. Kit identity: `1ea066df3e6b76e131bac47215d1a966e22a8dc38974377154eae6b29e156f75`.

ZENODO deferred (D4a); annotated tag `zc-registration-20261013` object `ba0d4bd6711203af2d84dc4495f2c4f694f68d95`; OTS proof in timestamp commit `f0db595796450b874e0484306803c3b6b48534df`. Initial OpenTimestamps stamp succeeded with pending calendar attestations; no mature Bitcoin attestation verification claimed. Registration was pushed before START.

Python/numpy/scipy/h5py: 3.12.2 2.2.6 1.15.3 3.16.0. Workers: 32. START UTC: 2026-10-13T10:10:33.785407+00:00; completion UTC inferred from START plus recorded elapsed: 2026-10-13T11:45:50.168920+00:00; elapsed 5716.383513 s. Attempts: 1; interruptions: none recorded; leftover temporary files: none.

Frozen reported readings, verbatim:

```text
{
 "status": "COMPLETE",
 "records": 2880,
 "readings": {
  "R-C1": {
   "reading": "MET",
   "decisions": {
    "negative": 0,
    "undecided": 4,
    "positive": 0
   },
   "flag": null
  },
  "R-C2": {
   "reading": "NOT MET",
   "decisions": {
    "negative": 42,
    "undecided": 2,
    "positive": 28
   },
   "flag": "reading involves cells with optimization-limited records"
  }
 }
}
```

SHA-256 (original completed bytes):

- `summary.json`: `edef54526ec5eb6be2f86a538bad6b0ba4abaa1ceb4b000e4784601122035215`
- `results.h5`: `0d78ced16dd22038f6f89010921456a5d83fcca0de56c62e85cf5c8c255fa371`
- `START.json`: `f2d7bf0fb1a91e83a17f4c798271609da7a6a088c5c459ea39e6c94fac6d86a6`

Committed census expected and checked before push: 2880 record JSON; 0 array NPZ (save_arrays=false); 11 run files; total 2891.

Anomalies/adjustments: The completed run already wrote summary.json and results.h5. CLI summarize would rewrite those files and remove wall_seconds_this_attempt/tasks_written_this_attempt. To preserve the completed bytes, the operator called check_start, load_records and pure reduce read-only, asserted every reduced key equals the original summary, and emitted the exact _short registered-reading stdout. No consolidate/summarize write or new fit was performed. Independent receiving must check this disclosed operational adjustment.

Optimization flags are retained as reported; no conclusion beyond the frozen reading. The independent receiving reviewer handles acceptance; the author handles any merge.

BLAS/configuration:

```text
/ceph/dwong/zc_venv/lib/python3.12/site-packages/numpy/__config__.py:155: UserWarning: Install `pyyaml` for better output
  warnings.warn("Install `pyyaml` for better output", stacklevel=1)
{
  "Compilers": {
    "c": {
      "name": "gcc",
      "linker": "ld.bfd",
      "version": "10.2.1",
      "commands": "cc"
    },
    "cython": {
      "name": "cython",
      "linker": "cython",
      "version": "3.1.0",
      "commands": "cython"
    },
    "c++": {
      "name": "gcc",
      "linker": "ld.bfd",
      "version": "10.2.1",
      "commands": "c++"
    }
  },
  "Machine Information": {
    "host": {
      "cpu": "x86_64",
      "family": "x86_64",
      "endian": "little",
      "system": "linux"
    },
    "build": {
      "cpu": "x86_64",
      "family": "x86_64",
      "endian": "little",
      "system": "linux"
    }
  },
  "Build Dependencies": {
    "blas": {
      "name": "scipy-openblas",
      "found": true,
      "version": "0.3.29",
      "detection method": "pkgconfig",
      "include directory": "/opt/_internal/cpython-3.12.7/lib/python3.12/site-packages/scipy_openblas64/include",
      "lib directory": "/opt/_internal/cpython-3.12.7/lib/python3.12/site-packages/scipy_openblas64/lib",
      "openblas configuration": "OpenBLAS 0.3.29  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64",
      "pc file directory": "/project/.openblas"
    },
    "lapack": {
      "name": "scipy-openblas",
      "found": true,
      "version": "0.3.29",
      "detection method": "pkgconfig",
      "include directory": "/opt/_internal/cpython-3.12.7/lib/python3.12/site-packages/scipy_openblas64/include",
      "lib directory": "/opt/_internal/cpython-3.12.7/lib/python3.12/site-packages/scipy_openblas64/lib",
      "openblas configuration": "OpenBLAS 0.3.29  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64",
      "pc file directory": "/project/.openblas"
    }
  },
  "Python Information": {
    "path": "/tmp/build-env-sgs8db9u/bin/python",
    "version": "3.12"
  },
  "SIMD Extensions": {
    "baseline": [
      "SSE",
      "SSE2",
      "SSE3"
    ],
    "found": [
      "SSSE3",
      "SSE41",
      "POPCNT",
      "SSE42",
      "AVX",
      "F16C",
      "FMA3",
      "AVX2"
    ],
    "not found": [
      "AVX512F",
      "AVX512CD",
      "AVX512_KNL",
      "AVX512_KNM",
      "AVX512_SKX",
      "AVX512_CLX",
      "AVX512_CNL",
      "AVX512_ICL"
    ]
  }
}
```
