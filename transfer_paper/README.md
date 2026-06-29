# Paper transfer bundle

This directory is the controlled handoff from experiment development to paper
analysis and writing. It contains lightweight, versioned evidence; derived
tables; reproducible plotting notebooks; and a claim-status document. It does
not contain raw GWOSC strain arrays or other large datasets.

Start with [WRITING_AGENT_BRIEF.md](WRITING_AGENT_BRIEF.md), then read
[MANUSCRIPT_EVIDENCE_MAP.md](MANUSCRIPT_EVIDENCE_MAP.md),
[PENDING_RESULT_PLACEHOLDERS.md](PENDING_RESULT_PLACEHOLDERS.md), and
[PAPER_WRITING_HANDOFF.md](PAPER_WRITING_HANDOFF.md). Together they state
which claims are supported, which results are negative, which experiments are
implemented but not yet run remotely, where placeholders must be used, and
which statements must not appear in the paper.

This folder is designed to stand alone for paper writing. It includes copied
evidence logs, JSON/YAML records, checksums, config snapshots, derived CSVs,
notebooks, draft captions, and rendered figures. It intentionally does not
include raw GWOSC strain arrays; those are too large for a writing bundle and
are not needed to interpret or cite the archived evidence.

## Directory layout

```text
transfer_paper/
├── PAPER_WRITING_HANDOFF.md
├── README.md
├── data/
│   ├── README.md
│   ├── configs/
│   ├── derived/
│   ├── gwosc/
│   ├── synthetic/
│   └── transfer_manifest.json
├── figures/
├── notebooks/
├── scripts/
└── tables/
```

The JSON, YAML, checksum, and log files under `data/gwosc/` are copied without
scientific modification from the archived evidence. CSV files under
`data/derived/` are convenience views generated from those authoritative
files. If the two ever disagree, the archived evidence JSON is authoritative.

`data/gwosc/current/` is the selected run used for paper plots.
`data/gwosc/runs/` contains the full copied local GWOSC evidence history, so a
writing agent does not need the original `evidence/` directory.

## Refresh after a remote run

After synchronizing a new evidence commit, refresh the bundle:

```bash
uv run python transfer_paper/scripts/refresh_bundle.py
uv run python transfer_paper/scripts/generate_notebooks.py
uv run python transfer_paper/scripts/render_available_figures.py
```

The refresh script selects the newest archived GWOSC run containing an
experiment record. If it finds the new filtering-equivalence or time-local PSD
records, it copies and derives their paper-facing tables automatically.

## Notebook order

1. `00_evidence_inventory.ipynb` — provenance, run inventory, and claim status.
2. `01_synthetic_validation.ipynb` — S1–S9 interval-backed synthetic results.
3. `02_gwosc_baseline_validation.ipynb` — the verified failed global-PSD gate.
4. `03_gwosc_filter_equivalence.ipynb` — shared-FIR experiment; verified
   positive for shared-statistic implementation identity.
5. `04_gwosc_time_local_psd.ipynb` — global/local PSD experiment; verified
   negative on real H1/L1 after the stationary synthetic control passed.

The notebooks save vector PDF and 300-dpi PNG figures under `figures/`.

## Placeholder rule

Any experiment that is pending or absent from `transfer_paper` must be drafted
as an explicit placeholder first. Failed synchronized experiments should be
written as negative results rather than tuned into positive claims. The writing
agent must not turn confirmatory GWOSC, CRESST, event-significance, or
injection-sensitivity claims into prose results until the required evidence
files exist and have been reviewed. Exact placeholder text is in
[PENDING_RESULT_PLACEHOLDERS.md](PENDING_RESULT_PLACEHOLDERS.md).

## Writing-control tables

The most useful machine-readable tables are:

- `data/derived/claim_status.csv` — what is verified, negative, pending, or
  not validated.
- `data/derived/paper_implications.csv` — how each experiment can be used in
  the manuscript.
- `data/derived/method_traceability.csv` — which protocol/config/output backs
  each Methods claim.
- `data/derived/figure_index.csv` — which plots are available and what each
  plot is allowed to communicate.
- `data/derived/evidence_inventory.csv` — file-level checksum inventory for
  the standalone transfer bundle.
