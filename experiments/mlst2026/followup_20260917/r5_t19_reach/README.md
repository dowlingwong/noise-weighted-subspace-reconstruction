# R5 — optional T19 full-covariance theorem-domain reachability

Status: configured, locally tested on development fixtures; **optional** for
execution (a stronger full-covariance claim).

## Question

Can independent full-covariance estimation enter the theorem domain (eta <= 0.1)
on G1 at a fixed larger calibration allocation, and do in-domain realisations
satisfy the unchanged 44 eta^2 bound?

## Design (new protocol)

- `PROTOCOL_R5_T19_REACH_v1.json`, package `r5_t19_reach`, supersedes T19-S v2 for
  the `S_full` arm only. Fixed design, no selection authority.
- Calibration grid: production [524288 (old anchor), 2,700,000, 3,000,000,
  4,000,000]; fixture [4096, 8192]. Each cell is one `m` with independent seeds.
- Bounded-memory covariance accumulation (`covariance.py`), checked against direct
  construction on small fixtures; memory is O(chunk*d + d^2), independent of m.
- G1 and the eta <= 0.1 domain are unchanged; out-of-domain realisations are
  never discarded or redrawn; `lambda_min(I+E) <= 0` is labelled indefinite.
- `"not reachable"` at the largest m is a valid, reportable result.
- Seed namespaces: development [924000, 924999], pilot [29000, 29299],
  confirmation [30000, 30499].

## Commands

```sh
P=experiments/mlst2026/followup_20260917/r5_t19_reach/PROTOCOL_R5_T19_REACH_v1.json
python -m experiments.mlst2026.followup_20260917.r5_t19_reach run    --protocol $P --out $OUT --fixture
python -m experiments.mlst2026.followup_20260917.r5_t19_reach reduce --protocol $P --out $OUT
# production (optional; real memory/time must be measured on one pilot cell first)
python -m experiments.mlst2026.followup_20260917.r5_t19_reach run    --protocol $P --out $RUN
python -m experiments.mlst2026.followup_20260917.r5_t19_reach reduce --protocol $P --out $RUN
```

## Tests

`tests/test_followup_20260917_r5.py` (6 tests): validate/enumerate, streaming
covariance equals direct construction, nested arms are prefixes, eta declines with
m while out-of-domain records are retained, fixture end-to-end with controls, and
namespace disjointness.

## Limits

No production run has been executed. The 3–4 million-sample cells must be priced
from a measured pilot cell before any allocation; do not allocate millions of
samples under a smoke-test label. Bounded memory is verified on small fixtures,
not at production `m`.
