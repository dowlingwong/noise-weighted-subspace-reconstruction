# Datasets to download — all four papers

Data lives **outside** the repositories. Paper 1 root: `PAPER1_DATA_ROOT`,
default `/ceph/dwong/paper1_dataset`.

## Paper 1 — Noise Covariance Defines Reconstruction Geometry

| Name | Size | Phase | URL |
|---|---|---|---|
| Synthetic S0–S9 | generated, 0 MB | all synthetic claims | — (`src/noise_geometry/experiments/synthetic.py`) |
| CRESST-II/III pulse shapes — **test split** | ~160 MB | CRESST-A positive validation | ORIGINS DMDC, see note · [arXiv:2508.03078](https://arxiv.org/abs/2508.03078) |
| CRESST-II/III — **full release** | ~2.5 GB | full validation (`X_train.npy` alone ≈ 2 GB) | same |
| GWOSC GW150914 strain | ~MB | completed **negative** real-noise stress test | https://gwosc.org/data/ |
| GWOSC `fig2-unfiltered-waveform-H.txt` | ~KB | PSD / reference check | https://www.gw-openscience.org/s/events/GW150914/P150914/fig2-unfiltered-waveform-H.txt |
| TIDMAD *(optional)* | ~2.2 GB per file; default subset 2 validation + 2 science ≈ 9 GB | optional real SQUID noise | `download_data.py` in https://github.com/jessicafry/TIDMAD |

**Notes**

- CRESST has **no static direct link.** DMDC serves through an interactive browser: open the CRESST page, copy the directory URL, pass as `--base-url` (or set `cresst_base_url` / `cresst_file_urls`). https://www.origins-cluster.de/odsl/dark-matter-data-center/available-datasets/cresst
- Release contents: `X_train.npy` `(979446, 512)` f32, `X_test.npy` `(78084, 512)` f32, `y_*.npy` ±1 quality labels, `features_*.csv`, `detectors.csv`, `resolutions_test.csv`. 25 kHz records downsampled to 512 samples. `noise` flag = random-triggered baselines (for Σ/PSD); `clean` = accepted pulses.
- **Checksum TIDMAD files before use.** A prior download produced `abra_validation_0013.h5` at 17.76 GB against an expected ~2.2 GB — a duplicated stream, not a truncation — failing with `OSError: wrong B-tree signature`.
- Nothing above enters the repo: no `.npy`, `.h5`, ROOT, or compressed raw data.

## Paper 2 — Architecture-Constrained Likelihood Geometry (NFPA)

| Name | Size | Phase | URL |
|---|---|---|---|
| K-alpha traces `data/k_alpha/k_alpha_traces.h5` (dataset `traces`, `(N, T)`) | TBD | reconstruction dataset | **not public — internal** |
| K-alpha RQs `data/k_alpha/k_alpha_rqs.h5` (dataset `rqs`) | TBD | per-record quantities | **not public — internal** |

**Gap.** `paper2/data/datasets.py` is a placeholder API, and no provenance, licence, or acquisition route is recorded for the K-alpha files. As it stands Paper 2 is not reproducible by a reader. Either document the source and terms, or substitute a public dataset before submission.

## Paper 3 — Consequence-Aware Failure Diagnostics

| Name | Size | Phase | URL |
|---|---|---|---|
| NuBench **Hexagon Ice LE** (SQLite) | large — see catalogue | `nubench_hexagon_ice_le_dynedge` pilot | https://github.com/graphnet-team/NuBench |
| NuBench released predictions (Parquet) | — | Table 6 metric reproduction | same |
| NuBench DynEdge checkpoint | — | module-dropout pilot | same |

**Notes**

- [arXiv:2511.13111](https://arxiv.org/abs/2511.13111), JINST 21 (2026) T05001. Catalogue spans ~130M ν_μ interactions, 10 GeV–100 TeV, six geometries (Flower S/L/XL, Triangle, Cluster, Hexagon); only Hexagon Ice LE is used.
- Requires `graphnet` (`SQLiteDataset`). Ships as SQLite **and** Parquet.
- **Open blocker:** CPU checkpoint re-inference differs event by event from the released predictions, and released-prediction rescoring misses the strict 0.01 tolerance. V0 remains open — `paper3/results/nubench_hexagon_ice_le_dynedge/RESULT.md`.

## Agent paper — Availability-Aware Causal Branching

Separate repository (`agent_sys_paper`). Tier 0–1 total ≈ 35 MB.

| Name | Size | Phase | URL |
|---|---|---|---|
| Fashion-MNIST | ~30 MB | Tier 1 · `imageClassificationFMnist` · faults F1b, F2b | https://huggingface.co/datasets/zalandoresearch/fashion_mnist |
| SATLIB `uf250-1065` (100 SAT instances) | ~2 MB | Tier 1 · `3SATTime` hard region, m/n ≈ 4.26 | https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html |
| SATLIB `uuf250-1065` (100 UNSAT instances) | ~2 MB | Tier 1 · F2a UNSAT path | https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html |
| Titanic *(optional)* | ~100 KB | Tier 1 · optional replacement task | https://www.kaggle.com/c/titanic |
| MLGym (repo + task configs) | repo only | Tier 0–1 · `battleOfSexes`, `3SATTime`, `imageClassificationFMnist` | https://github.com/facebookresearch/MLGym |
| MLAgentBench (repo + 2–3 task datasets) | TBD | Tier 2 · task selection not yet made | https://github.com/snap-stanford/MLAgentBench |
| MLE-bench Lite (22 Kaggle competitions) | ~33 GB | Tier 3 · **do not download yet** — sealed until confirmatory freeze | https://github.com/openai/mle-bench |
| Agents' Last Exam | TBD | Tier 4 · negative control, minimal fetch only | https://arxiv.org/abs/2606.05405 |

**Notes**

- MLGym is CC BY-NC 4.0 — approval (W0-2) required before use or redistribution.
- MLE-bench Lite needs a Kaggle API token and rule acceptance per competition (×22).
- SATLIB ships DIMACS CNF; confirm the task's input format.
- DeepSeek V4 Flash 0731 is API-only — no download, needs a key.
