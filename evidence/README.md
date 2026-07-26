# Verification evidence

_Lightweight, reviewable artifacts produced by controlled local validation runs_

---

## 📋 Storage policy

Commit machine-readable summaries, configuration snapshots, environment records, checksums, and text logs here. Keep raw datasets, NumPy arrays, model checkpoints, and other large binary artifacts outside the repository.

GWOSC evidence belongs under:

```text
evidence/gwosc/<UTC timestamp>_<tested commit>/
```

CRESST evidence belongs under:

```text
evidence/cresst/<UTC timestamp>_<tested commit>/
```

Each run must contain a `manifest.json` and `SHA256SUMS`. A failed run must be preserved using the same structure as a passing run.

The execution agent verifies each run and creates one evidence-only local commit. A user reviews that commit before manually pushing or otherwise synchronizing it.

## 🛡️ Prohibited content

Do not commit:

- `.npz` or `.npy` arrays
- HDF5, ROOT, or compressed raw datasets
- Credentials, tokens, or private URLs
- Artifacts unrelated to the recorded run

Follow [the remote execution guide](../docs/REMOTE_EXECUTION.md) when producing
a new evidence bundle.

## 📊 Archived runs

| Run | Tested commit | Computational status | Scientific status | Interpretation |
| --- | --- | --- | --- | --- |
| `20260622T162349Z_b7842451781f` | `b7842451781f` | Stage 0 did not run because `uv` was unavailable | Not run | Infrastructure failure retained for provenance |
| `20260622T164907Z_f541c542f778` | `f541c542f778` | Stage 0 passed; 89 tests and all GWOSC commands passed | `failed_acceptance` | PSD matched GWpy exactly, but held-out amplitude spread exceeded prediction |
| `20260622T175125Z_b169c1f595a4` | `b169c1f595a4` | Stage 0 passed; 91 tests and all commands passed | `failed_acceptance` | Official DATA coverage and enhanced diagnostics passed; random and chronological null calibration still failed |

The scientific interpretation of these controlled runs is recorded in
[`docs/GWOSC_RESULT.md`](../docs/GWOSC_RESULT.md).

### CRESST

| Run | Tested commit | Computational status | Scientific status | Interpretation |
| --- | --- | --- | --- | --- |
| `20260704T013758Z_9c9f696ff6bc` | `9c9f696ff6bc` | Stage C5 exit 0; predeclared run on the full public test split, all 7 channels used | `passed` (predeclared gate) | EMPCA beat single-template OF and was competitive with multi-template OF (pooled paired-bootstrap 95% CIs excluding zero), and all 5 negative controls degraded at the pooled level. One channel (`35_58.0`) showed a non-degrading `wrong_psd` control on its own (pooled result unaffected). The trigger-efficiency detector metric was largely inconclusive within the predeclared amplitude grid (0-10x sigma_OF) for most channels -- see `docs/CRESST_RESULT.md`. |

The scientific interpretation of this run is recorded in
[`docs/CRESST_RESULT.md`](../docs/CRESST_RESULT.md).

### Paper bundle

Durable, checksum-backed exports of `scripts/export_paper_bundle.py` belong
under:

```text
evidence/paper_bundle/<UTC timestamp>_<tested commit>/
```

Each run must contain a `manifest.json`, `SHA256SUMS`, and
`BUNDLE_MANIFEST.json` (the exporter's own richer per-file checksum
inventory, produced by `scripts/verify_paper_bundle.py`-checked runs only).

| Run | Tested commit | Computational status | Scientific status | Interpretation |
| --- | --- | --- | --- | --- |
| `20260726T044539Z_eb4592d72ad9` | `eb4592d72ad9` | Exporter and verifier both exit 0; 64 files staged | `passed` (bundle verification) | Superseded by the run below (not deleted, per the evidence rule). First durable paper-evidence bundle after the 2026-07-26 revision: EXP-00 through EXP-03, EXP-06, EXP-07 only. |
| `20260726T050816Z_eb4592d72ad9` | `eb4592d72ad9` | Exporter and verifier both exit 0; 73 files staged | `passed` (bundle verification) | Adds EXP-04 (S5 30-seed confirmatory sweep, `sweeps_v2_20260726T045735Z`) and EXP-05 (`rerun_decisions.json`, `appendix_f_traceability.csv`, `generator_distributional_diagnostics.*`) to the run above. Current durable bundle for the 2026-07-26 revision -- see `results/audits/DEVIATION_REPORT_2026-07-26.md` for the full deviation report. |
