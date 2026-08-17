# Anonymized code link — release recipe (v2, after external audit 2026-08-18)

**Decision: conditional GO.** A link materially strengthens an experiment-heavy,
"measured-not-asserted" paper, and the CFP permits one. **The v1 recipe of this note
was NO-GO**: an audit found 53 files in the proposed export carrying identity,
institution, host, or absolute-path strings. The correct release is a **fresh,
whitelist-built repository — never a cleaned copy of this one.**

**Fallback, stated up front:** if the snapshot does not pass every gate below by the
paper freeze, submit without the link. The paper is self-contained; a leaking or
broken link is worse than no link.

## Verified leak inventory (do not trust intuition; these were found by grep)

- **Four source files**, not two: `scripts/train_tidmad_local.sh`,
  `src/tidmad/paths.py`, `src/tidmad/backbone/checkpoints.py`, `src/tidmad/model.py`
  — `/ceph/dwong` candidates, host names, or identity strings. Each must be edited
  in the export (`/path/to/...` placeholders) and re-checked.
- `pyproject.toml` names the author; `LICENSE` contains "Dowling_Wong". The export
  needs neutral packaging metadata and an anonymized license notice (e.g. "The
  Authors"); confirm the chosen license permits an anonymized copyright line during
  review.
- Git history, `docs/plans/`, `paper_rps2026/{plan,notes,sources}/`, run logs,
  `artifacts/`, `wandb/`, `__pycache__/`: excluded by construction (whitelist).
- `results/rps2026/` **as a directory** contains machine paths and failed-run logs.
  Export only the exact final CSV/JSON files the figures and tables read.

## Whitelist (build the repo from exactly this, nothing else)

1. Source: `src/tidmad/` (train, psd, loss, config, score, score_unrounded,
   whitening_range, paths, model, backbone), `src/NFPA/`,
   `experiments/synthetic/nfpa_confirmatory/` (no `__pycache__`, no `tests` unless
   they pass in the export), **plus the `src/noise_geometry/` subset the
   confirmatory code imports** (`covariance.py`, `metrics.py`, `methods.py` import
   from it; the v1 list was dependency-incomplete).
2. Scripts: `make_rps_figure.py`, `rung3_whitening_range.py`,
   `train_tidmad_local.sh` (path-neutralized).
3. Data records, individually named: the Block A/B CSVs, `E5_*_eval_series.csv`,
   `E5_benchmark_contrast_interval.json`, `E6b_rung3_whitening_range.json`,
   **`E6_whitening_range.json` (must first be synced from the server — it is not in
   the local tree)**, and validated E7 artifacts when they exist.
4. A README: TIDMAD is public (CC-BY-4.0, Fry et al. 2024) with download source;
   one command per figure and table; neutral tone, no names.

## Gate status (2026-08-18, builder `scripts/build_anonymized_export.sh`)

| Gate | Status |
|---|---|
| 1 whitelist build | **PASS** — 78 files, nothing outside the whitelist |
| 2 anonymity grep | **PASS** — zero hits; runs inside the build and fails it on any hit (first run caught 3 survivors in the launcher, since fixed) |
| 3 manual read of every file | open — operator |
| 4a figure reproduction from the export | **PASS** — identical statistics to the archived figure |
| 4b tests from the export | **PARTIAL** — tidmad suite 10 passed / 5 torch-skipped; all 18 NFPA tests collect; the full NFPA suite (~12 min of ALS fits) exceeds the local sandbox's ~3-minute call cap — run once on the server: `PYTHONPATH=src:. pytest experiments/synthetic/nfpa_confirmatory/tests` |
| 5 upload immutable commit | open — author |
| 6 signed-out URL check | open — author |

Still to add before upload: `E6_whitening_range.json` and validated `E7_*`
artifacts (the builder picks both up automatically once synced).

## Gates (all mandatory, in order)

1. Fresh export builds; no file outside the whitelist.
2. `grep -riE 'dwong|wongdowling|dowling|ceph|deepthought|delight|kit\.edu|karlsruhe|imperial'`
   over the export returns **zero** hits.
3. Manual inspection of every file (the export is small; read it all).
4. From a clean environment: tests pass and `make_rps_figure.py` reproduces
   Figure 1 byte-comparably from the exported records.
5. Pin one immutable commit; upload to [Anonymous GitHub](https://anonymous.4open.science/).
6. Open the anonymous URL in a signed-out browser and re-check.

## Wiring into the paper

One line in §4 or a page-1 footnote:
`\url{https://anonymous.4open.science/r/<id>}`. Page-budget donors listed in
`round8_contingency.md`.
