# Y19 completed confirmation host return

Host: deepthought. EXP freeze: `d85fc7fad092fc0d4d497eb92ad3a7f3ca15d85a`. Kit identity: `bbf4e8c64bfb1126bcccb8645a18e78196d9e03c5a2ea1f8117e2464eaa2bfe8`.

MS Y19 freeze: `f2093bc256738258afe7393b67f632c76bac1f26`.

ZENODO deferred (D4a); annotated tag `y19-registration-20261012` object `bfda01c169cbe2a1b75d0a25c88459c4b7f596d2`; OTS proof in timestamp commit `0966aff74b1255613b30058ea9584e362354b770`. Initial OpenTimestamps stamp succeeded with pending calendar attestations; no mature Bitcoin attestation verification claimed. Registration was pushed before START.

Python/numpy/scipy/h5py: 3.12.2 2.2.6 1.15.3 3.16.0. Workers: 4. START UTC: 2026-10-13T10:07:30.828766+00:00; completion UTC inferred from START plus recorded elapsed: 2026-10-13T12:19:02.214180+00:00; elapsed 7891.385414 s. Attempts: 1; interruptions: none recorded; leftover temporary files: none.

Frozen reported readings, verbatim:

```text
COMPLETE 480 20 machine overall: MET
A - - ? + + MET
B - - + + + MET
control ['negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative'] MET
reported overall (protocol §7): MET
optimization-limited records: 62
A optimization-limited deciding cells: [729, 1458, 5834, 11668]
B optimization-limited deciding cells: [302, 604, 2415]
```

SHA-256 (original completed bytes):

- `summary.json`: `6d4c2b16cb32df33f5d38b26ee73b90ee4c96ef5ae10ed46724414ea92518b32`
- `results.h5`: `7f9de85ff7c93240782ccfea5ee68d0f0487436244fe56a2e53d7e81f1c4b2e6`
- `START.json`: `99877a20761fb2a0830675a1958ab29419f97d0dcc5b4b2ecfee6a1ba5b8398f`

Committed census expected and checked before push: 480 record JSON; 480 array NPZ; 9 run files; total 969.

Anomalies/adjustments: No deviations from the frozen completion/archival steps. Original completed scientific outputs preserved.

Optimization flags are retained as reported; no conclusion beyond the frozen reading. The independent receiving reviewer handles acceptance; the author handles any merge.

BLAS/configuration:

```text
/ceph/dwong/y19_venv/lib/python3.12/site-packages/numpy/__config__.py:155: UserWarning: Install `pyyaml` for better output
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
