# Dataset Links and Access Notes

## Policy

Do not commit large downloaded datasets. Keep raw public data under `data/raw/`
or `data/external/`, processed arrays under `data/processed/`, and generated
figures/tables under `results/`.

## GWOSC

Role: primary public real-noise benchmark.

Canonical links:

- [GWOSC data portal](https://gwosc.org/data/)
- [GWOSC homepage](https://gwosc.org/)
- [GWOSC event catalog](https://gwosc.org/eventapi/html/GWTC/)
- [gwosc Python docs](https://gwosc.readthedocs.io/en/stable/)
- [GWpy docs](https://gwpy.readthedocs.io/en/stable/)
- [GWpy open-data tutorial](https://gwpy.github.io/docs/2.1.1/timeseries/opendata/)
- [GWOSC tutorial GitHub](https://github.com/gwosc-tutorial/introduction_gwosc_data)

Suggested first implementation:

1. Use a known event such as GW150914.
2. Fetch short H1 and L1 strain windows only.
3. Estimate PSD from an off-source window.
4. Whiten strain and template.
5. Run matched-filter / OF-style projection.
6. Compare raw MSE, PSD-weighted residual, whitening consistency, and
   matched-filter score.

The default scripts must not download full observing-run data.

## CRESST-II / CRESST-III Pulse-Shape Data

Role: strongest complementary public dataset for cryogenic detector pulse-shape
processing.

Canonical links:

- [ORIGINS / DMDC CRESST page](https://www.origins-cluster.de/odsl/dark-matter-data-center/available-datasets/cresst)
- [CRESST pulse-shape paper](https://arxiv.org/abs/2508.03078)
- [CRESST papers page](https://cresst-experiment.org/papers)

Suggested first implementation:

1. Download released trace arrays manually from the ORIGINS / DMDC page.
2. Put raw downloads under `data/external/cresst/`.
3. Inspect trace shapes, labels, detector metadata, and examples.
4. Estimate PSD or covariance from baseline/noise-like traces or trace
   pre-regions.
5. Build average templates or learned pulse subspaces.
6. Compare PCA, EMPCA, MSE tied AE, weighted tied AE, and optional OF.

Manual download is acceptable until a stable direct-download API is confirmed.

## TIDMAD Optional

Role: optional AI-for-science dark-matter denoising benchmark.

Canonical links:

- [TIDMAD GitHub](https://github.com/jessicafry/TIDMAD)
- [TIDMAD arXiv](https://arxiv.org/abs/2406.04378)
- [TIDMAD OpenReview](https://openreview.net/forum?id=GgKHKZgg0Y)
- [TIDMAD Zenodo](https://zenodo.org/records/15418539)

Use only after GWOSC and CRESST are stable. Start with a manageable validation
subset and compare a baseline model, simple filter, and noise-aware loss.

## Not Core for Paper 1

IceCube, NuBench, OpenFWI, and MicroBooNE are not core Paper 1 datasets. NuBench
is better reserved for Paper 2-style detector geometry, coverage, and
architecture studies.
