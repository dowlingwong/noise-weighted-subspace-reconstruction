# Verification evidence

_Lightweight, reviewable artifacts produced by controlled remote validation runs_

---

## 📋 Storage policy

Commit machine-readable summaries, configuration snapshots, environment records, checksums, and text logs here. Keep raw datasets, NumPy arrays, model checkpoints, and other large binary artifacts outside the repository.

GWOSC evidence belongs under:

```text
evidence/gwosc/<UTC timestamp>_<tested commit>/
```

Each run must contain a `manifest.json` and `SHA256SUMS`. A failed run must be preserved using the same structure as a passing run.

## 🛡️ Prohibited content

Do not commit:

- `.npz` or `.npy` arrays
- HDF5, ROOT, or compressed raw datasets
- Credentials, tokens, or private URLs
- Artifacts unrelated to the recorded run

Follow [the remote GWOSC agent runbook](../docs/REMOTE_GWOSC_AGENT_RUNBOOK.md) when producing a new evidence bundle.
