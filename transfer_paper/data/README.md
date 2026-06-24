# Data policy and provenance

This directory contains only lightweight paper-analysis inputs. Raw GWOSC
strain remains outside the repository at the configured data root
(`/ceph/dwong/paper1_dataset` on the execution server). The copied
`raw_metadata.json` records source URLs, detector DATA segments, external file
paths, and SHA-256 digests.

## Authoritative versus derived files

- `gwosc/current/` contains exact copies of the selected archived evidence
  files.
- `gwosc/runs/` contains exact copies of every archived local GWOSC evidence
  run available when the bundle was refreshed.
- `configs/` contains snapshots of the GWOSC and synthetic YAML configs used
  to understand or reproduce the paper-facing experiments.
- `synthetic/` contains exact copies of the S1–S9 multi-seed sweep records.
- `synthetic/figures/` contains exact copies of the existing synthetic result
  figures from `results/figures/`.
- `source_documents/` contains exact copies of the validation record,
  reference protocol, follow-up protocol, roadmap, and progress record.
- `derived/` contains CSV views generated for notebooks and plotting.
- `transfer_manifest.json` records each copied/generated file, its source, and
  its SHA-256 digest.

Within this standalone bundle, the authoritative numerical records are the
copied JSON and sweep records under `gwosc/runs/`, `gwosc/current/`, and
`synthetic/`. Derived CSVs must be regenerated after any new evidence
synchronization.

## Writing-facing derived tables

- `claim_status.csv` constrains manuscript claims.
- `paper_implications.csv` explains what each result supports or rules out.
- `method_traceability.csv` links methods text to configs, protocols, and
  outputs.
- `figure_index.csv` lists available and pending plots.
- `evidence_inventory.csv` records every transferred evidence file and its
  checksum.

## Refresh

```bash
uv run python transfer_paper/scripts/refresh_bundle.py
```

The refresh process never downloads data and never modifies archived evidence.
