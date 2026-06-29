# CRESST Download and Setup

Public dataset: **CRESST-II/III pulse-shape data**, described in
[arXiv:2508.03078](https://arxiv.org/abs/2508.03078) and hosted by the ORIGINS
Dark Matter Data Center (DMDC):

https://www.origins-cluster.de/odsl/dark-matter-data-center/available-datasets/cresst

## What the release contains

| File | Shape / rows | Use in Paper 1 |
| --- | --- | --- |
| `X_train.npy` | `(979446, 512)` float32 | raw voltage traces, train detectors |
| `X_test.npy` | `(78084, 512)` float32 | raw voltage traces, test detectors |
| `y_train.npy` / `y_test.npy` | `(n,)` ±1 | quality-cut survival labels |
| `features_train.csv` / `features_test.csv` | `n` rows | per-record features incl. `noise`, `clean`, `run`, `channel`, `pulse_height`, `rise_time`, `decay_time` |
| `detectors.csv` | per sample | run/channel/`noise` overview |
| `resolutions_test.csv` | per channel | optimum-filter resolutions |

Traces are 25 kHz records downsampled to 512 samples. The `noise` flag marks
random-triggered baselines (used to estimate the noise covariance/PSD); `clean`
marks accepted pulses (used for reconstruction).

## Obtaining the download URL

The DMDC serves files through an interactive repository browser, so there is no
single static public direct-download link. Open the CRESST DMDC page, locate the
directory that holds the released files, and copy that base URL. The downloader
then composes per-file URLs from it (or pass explicit per-file URLs via the
config `cresst_file_urls`).

## Download (remote server)

Data is written outside the repository, under the data root
(`/ceph/dwong/paper1_dataset/cresst/raw` by default; override with
`PAPER1_DATA_ROOT` or `--data-root`).

Quick start — fetch only the ~160 MB test split first:

```bash
uv run python scripts/download/download_cresst.py \
  --base-url <DMDC_FILE_DIRECTORY_URL> --group test
```

Full release (~2.5 GB):

```bash
uv run python scripts/download/download_cresst.py \
  --base-url <DMDC_FILE_DIRECTORY_URL> --group all
```

Useful flags: `--dry-run` (print the plan), `--check` (verify already-present
files without downloading), `--files X_test.npy y_test.npy` (explicit subset),
`--force` (re-download). The script verifies every `.npy` shape/dtype and CSV
row count against the published manifest, writes `raw/cresst_manifest.json` with
checksums and provenance, and refuses to write inside the repo.

If you download manually instead, place the files under `.../cresst/raw/` and
run with `--check` to verify them.

## Build a development cache and run

```bash
uv run python scripts/preprocess/preprocess_cresst.py \
  --max-pulses 5000 --max-noise 5000

uv run python scripts/run_experiment.py \
  --config configs/cresst/pulse_shape_smoke.yaml
```

The predeclared validation run uses
`configs/cresst/cresst_validation.yaml`; the protocol is in
`docs/EXPERIMENT_PROTOCOLS.md` (CRESST Validation Protocol).

## Citation

Cite both: (1) CRESST Collaboration, *Description of CRESST-II and CRESST-III
pulse shape data*, arXiv:2508.03078 (2025); (2) G. Angloher et al., *Towards an
automated data cleaning with deep learning in CRESST*, Eur. Phys. J. Plus 138,
100 (2023), arXiv:2211.00564. Do not commit downloaded `.npy`/`.csv` arrays.
