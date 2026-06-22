# Verification evidence

_Lightweight, reviewable artifacts produced by controlled local validation runs_

---

## 📋 Storage policy

Commit machine-readable summaries, configuration snapshots, environment records, checksums, and text logs here. Keep raw datasets, NumPy arrays, model checkpoints, and other large binary artifacts outside the repository.

GWOSC evidence belongs under:

```text
evidence/gwosc/<UTC timestamp>_<tested commit>/
```

Each run must contain a `manifest.json` and `SHA256SUMS`. A failed run must be preserved using the same structure as a passing run.

The execution agent leaves each run uncommitted. A user reviews the checksums and contents before manually staging, committing, or synchronizing the evidence.

## 🛡️ Prohibited content

Do not commit:

- `.npz` or `.npy` arrays
- HDF5, ROOT, or compressed raw datasets
- Credentials, tokens, or private URLs
- Artifacts unrelated to the recorded run

Follow [the remote GWOSC agent runbook](../docs/REMOTE_GWOSC_AGENT_RUNBOOK.md) when producing a new evidence bundle.

## 📊 Archived runs

| Run | Tested commit | Computational status | Scientific status | Interpretation |
| --- | --- | --- | --- | --- |
| `20260622T162349Z_b7842451781f` | `b7842451781f` | Stage 0 did not run because `uv` was unavailable | Not run | Infrastructure failure retained for provenance |
| `20260622T164907Z_f541c542f778` | `f541c542f778` | Stage 0 passed; 89 tests and all GWOSC commands passed | `failed_acceptance` | PSD matched GWpy exactly, but held-out amplitude spread exceeded prediction |

The scientific interpretation of the second run is recorded in
[`docs/GWOSC_VALIDATION_2026-06-22.md`](../docs/GWOSC_VALIDATION_2026-06-22.md).
