# CRESST Remote Agent Runbook

Created: 2026-07-03. Governs the CRESST-A study kickoff on a remote Linux
server. Scientific context: `docs/CURRENT_STATUS.md`; protocol and acceptance
principles: `docs/EXPERIMENT_PROTOCOLS.md` (CRESST Validation Protocol
section); dataset details: `scripts/download/download_cresst.md`.

## Mission

Stand up the public CRESST-II/III pulse-shape dataset (DMDC release,
arXiv:2508.03078), verify the existing code path end to end, implement the
missing protocol components, and only then execute the predeclared validation
run. CRESST is the candidate for positive public real-data validation after
the GWOSC negative stress test. Do not touch GWOSC.

## Hard Rules

1. A command exiting zero is computational success, not scientific acceptance.
2. Freeze all thresholds, grids, ranks, FPR points, and splits **before**
   inspecting any real-data validation score. After a failed gate, do not
   tune anything to obtain a pass; preserve the failure as evidence.
3. The smoke config (`pulse_shape_smoke.yaml`) is a computational check only.
   Never quote its metrics as scientific results.
4. Large data stays outside the repo (downloader enforces this). Never pass
   `--allow-repo-data`.
5. Report raw MSE only as a diagnostic; primary metrics follow the declared
   noise likelihood.
6. Preserve every generated `.json`, `.config.yaml`, and `.log`.

## Stage C0 — Environment And Baseline Health

```bash
uv sync --extra dev
uv run pytest -q
```

Expected: full pass except `gwpy`-dependent GWOSC tests, which skip or fail
only if the `gwosc` extra is not installed (setup issue, not scientific).
If the checkout is new, optionally close Stage 0 first:
`python3 scripts/stage0_remote_gate.py`.

## Stage C1 — Data Acquisition

Data root resolution: `--data-root` > `PAPER1_DATA_ROOT` > config >
`/ceph/dwong/paper1_dataset`. Target layout: `<root>/cresst/raw/`.

The release is hosted in the DMDC GitLab repository at MPCDF
(verified 2026-07-03; linked from the CRESST DMDC page):

```text
https://gitlab.mpcdf.mpg.de/dmdc/cresst/cresst-pulse-shape-deep-learning
```

All eight release files sit in `data/` on branch `master`. Base URL for the
downloader (it composes `<base-url>/<filename>`):

```text
https://gitlab.mpcdf.mpg.de/dmdc/cresst/cresst-pulse-shape-deep-learning/-/raw/master/data
```

```bash
BASE=https://gitlab.mpcdf.mpg.de/dmdc/cresst/cresst-pulse-shape-deep-learning/-/raw/master/data
# Plan first
uv run python scripts/download/download_cresst.py --base-url "$BASE" --group test --dry-run
# Test split (~160 MB), then full release (~2.5 GB)
uv run python scripts/download/download_cresst.py --base-url "$BASE" --group test
uv run python scripts/download/download_cresst.py --base-url "$BASE" --group all
```

Fallbacks if raw URLs misbehave (e.g. Git LFS redirects): download the
data-only archive
`https://gitlab.mpcdf.mpg.de/dmdc/cresst/cresst-pulse-shape-deep-learning/-/archive/master/cresst-pulse-shape-deep-learning-master.tar.gz?path=data`,
or `git clone` the repository (with `git lfs` installed), then move the files
into `<root>/cresst/raw/` and verify with `--check`.

The script verifies every `.npy` shape/dtype and CSV row count against the
published manifest and writes `raw/cresst_manifest.json`. If files were
fetched manually, place them in `<root>/cresst/raw/` and run with `--check`.
Acquisition is complete only when the manifest verification passes for
`X_train (979446,512)`, `X_test (78084,512)`, labels, feature CSVs,
`detectors.csv`, and `resolutions_test.csv`.

Record the dataset citations (printed by the downloader) in the run report.

## Stage C2 — Computational Smoke (No Scientific Claims)

```bash
uv run python scripts/preprocess/preprocess_cresst.py --split test
uv run python scripts/run_experiment.py --config configs/cresst/pulse_shape_smoke.yaml
```

Acceptance (computational only): exit zero; run record JSON written under
`results/metrics/`; `n_noise_traces >= 2`; finite metrics; EMPCA/AE principal
angle small. Any failure here is a setup/code bug — fix it before proceeding,
this is the one stage where iteration is allowed.

## Stage C3 — Implementation Gap (Must Close Before Validation)

`configs/cresst/cresst_validation.yaml` predeclares components that
`src/noise_geometry/cresst/analysis.py` does not yet implement. Implement,
with unit tests on synthetic data, before any validation run:

1. **Multi-template OF bank** (`multi_template_of`): GLS template bank over
   the `rise_time`/`decay_time` quantile grid (and `t0` shifts), scored by
   minimum weighted residual. This is the strong classical baseline and is
   currently absent from the code surface.
2. **Detector-facing metrics** (see `docs/EXPERIMENT_PROTOCOLS.md` Metrics):
   residual whitening ratio vs. frequency; detection score
   `chi2(noise-only) - chi2(signal-subspace)`; trigger efficiency vs.
   injected amplitude at fixed background FPR (0.1%, 1%, 5%) with thresholds
   set on background only; 50%/90% efficiency thresholds; amplitude RMSE and
   bias; bootstrap 95% intervals (`bootstrap_reps: 300`).
3. **Injection machinery**: prefer direct injection into `noise`-flagged
   baseline traces (an event-level noise bank exists in this release); if a
   measured-PSD Gaussian condition is added, label it "measured-PSD
   injection", never "measured-noise-trace injection", and report the
   degradation between conditions.
4. **Negative controls**, each of which must degrade the result: wrong
   PSD/covariance; permuted time bins; random weighted-orthonormal basis at
   matched rank; train on noise-only, test on signal.
5. **Leakage-safe splits**: split by detector/run/channel groups from
   `features_*.csv`, not by random trace, so no
   calibration/evaluation or duplicate-trace leakage. The current random
   70/30 permutation split in `_run_release` is not sufficient for the
   validation claim.

Every new component gets a test in `tests/` runnable without real data.

## Stage C4 — Predeclaration Freeze

Before running on real data, write `docs/CRESST_PREDECLARED_GATES.md`
containing numeric values for: template-bank grid points, rank `k`
(config currently declares 6), FPR points, efficiency thresholds, split
definition and seeds, covariance shrinkage, and the quantitative acceptance
gate for the headline claim:

```text
EMPCA at rank k must beat single-template OF and remain competitive with
multi-template OF on the likelihood-aligned metrics, with all negative
controls degrading as expected.
```

Commit this file (and the final `cresst_validation.yaml`) before executing
Stage C5. The commit hash is part of the evidence.

## Stage C5 — Predeclared Validation Run

```bash
uv run python scripts/run_experiment.py --config configs/cresst/cresst_validation.yaml
```

Run once per predeclared seed list. Whatever the outcome, archive evidence
and report. If the gate fails, the result is a valid negative finding —
handle exactly as GWOSC was handled (preserve, interpret, do not tune).

## Evidence Packaging

Mirror the GWOSC pattern:

- raw run records under `results/metrics/` (JSON + config + log);
- an evidence snapshot under `evidence/cresst/<UTCSTAMP>_<hash>/`;
- synchronized copies and derived CSV summaries under
  `transfer_paper/data/cresst/` via `transfer_paper/scripts/refresh_bundle.py`;
- update `transfer_paper/data/derived/claim_status.csv` with the CRESST row
  (allowed statement + prohibited overclaim);
- update `docs/CURRENT_STATUS.md` validation table and `CHANGELOG.md`.

## Final Report Template

```text
Commands run:
- ...

Data acquisition:
- manifest verification: passed/failed, files, checksums path

Stage C2 smoke: passed/failed (computational only)

Implementation delivered (C3):
- multi-template OF / detector metrics / injections / negative controls / splits
- tests added and passing: ...

Predeclaration (C4): commit hash, gate values

Stage C5 verdicts:
- EMPCA vs single-template OF: passed/failed
- EMPCA vs multi-template OF: competitive/not
- negative controls: all degraded as required? yes/no
- trigger efficiency / resolution summary: ...

Interpretation (manuscript-safe): ...

Open issues:
- setup/tooling:
- scientific:
- packaging:
```
