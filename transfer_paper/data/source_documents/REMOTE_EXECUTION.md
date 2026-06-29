# Remote Execution And Evidence Packaging

Last consolidated: 2026-06-29.

## Purpose

Use this document for remote runs and evidence packaging. For scientific status,
read `CURRENT_STATUS.md`. For GWOSC interpretation, read `GWOSC_RESULT.md`.

## Stage 0 Remote Gate

Run Stage 0 from a clean remote Linux checkout:

```bash
python3 scripts/stage0_remote_gate.py
```

The gate runs:

```bash
uv sync --extra dev --extra gwosc
uv run pytest -q
uv run python scripts/run_all_core.py
uv run python scripts/make_tables.py
uv run python scripts/make_all_figures.py
```

Stage 0 passes only when all commands exit zero and the checkout remains clean
apart from archived evidence outputs.

## GWOSC Preflight

The GWOSC follow-ups require the documented waveform:

```text
/ceph/dwong/paper1_dataset/gwosc/raw/GW150914/waveforms/fig2-unfiltered-waveform-H.txt
```

If missing, run:

```bash
uv run python scripts/download/download_gwosc.py \
  --config configs/gwosc/gw150914_smoke.yaml \
  --download \
  --timeout 900
```

If the downloader cannot reach the waveform URL, fetch it directly:

```bash
mkdir -p /ceph/dwong/paper1_dataset/gwosc/raw/GW150914/waveforms
curl -L \
  https://www.gw-openscience.org/s/events/GW150914/P150914/fig2-unfiltered-waveform-H.txt \
  -o /ceph/dwong/paper1_dataset/gwosc/raw/GW150914/waveforms/fig2-unfiltered-waveform-H.txt
```

A missing waveform is a setup failure, not a scientific failure.

## Follow-Up Run Commands

```bash
uv run python scripts/run_experiment.py \
  --config configs/gwosc/filter_statistic_equivalence.yaml

uv run python scripts/run_experiment.py \
  --config configs/gwosc/time_local_noise.yaml
```

Preserve each generated `.json`, `.config.yaml`, and `.log` file.

## Evidence Bundle Refresh

The paper-revision evidence bundle lives under `transfer_paper/`.

Common refresh commands:

```bash
uv run python transfer_paper/scripts/refresh_bundle.py
uv run python transfer_paper/scripts/generate_notebooks.py
uv run python transfer_paper/scripts/render_available_figures.py
```

If `refresh_bundle.py` cannot find source synthetic sweeps under
`results/sweeps`, it should preserve existing synchronized synthetic evidence
rather than deleting valid transfer evidence. Treat refresh failures as
packaging/tooling failures until the scientific JSON evidence is inspected.

## Expected Follow-Up Evidence

GWOSC follow-up records:

```text
transfer_paper/data/gwosc/followup/filter_equivalence.json
transfer_paper/data/gwosc/followup/filter_equivalence.config.yaml
transfer_paper/data/gwosc/followup/time_local_noise.json
transfer_paper/data/gwosc/followup/time_local_noise.config.yaml
```

Derived summaries:

```text
transfer_paper/data/derived/gwosc_filter_equivalence_summary.csv
transfer_paper/data/derived/gwosc_time_local_psd_summary.csv
transfer_paper/data/derived/gwosc_time_local_psd_blocks.csv
transfer_paper/data/derived/gwosc_time_local_psd_spectral_summary.csv
transfer_paper/data/derived/claim_status.csv
transfer_paper/data/derived/paper_implications.csv
```

Figures:

```text
transfer_paper/figures/gwosc_filter_equivalence.png
transfer_paper/figures/gwosc_time_local_psd.png
```

## Failure Handling

- Preserve failed scientific results.
- Do not rerun with changed thresholds or tuned settings to obtain a pass.
- Distinguish setup failures from scientific failures:
  - missing `uv`, missing `gwpy`, missing waveform: setup;
  - failed predeclared calibration gate after successful execution: scientific.
- Report commands, exit status, generated evidence paths, pass/fail verdicts,
  and manuscript-safe interpretation.

## Final Report Template

```text
Commands run:
- ...

Evidence:
- ...

Verdicts:
- F1 shared-FIR: passed/failed
- L1 time-local PSD: passed/failed

Interpretation:
- ...

Open issues:
- setup/tooling:
- scientific:
- packaging:
```
