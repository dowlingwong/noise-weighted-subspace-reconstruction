# meg_z kit: WS-D MEG multichannel study (development kit, NOT FROZEN)

Semi-synthetic test of the paper's two linear classes on **measured** MEG noise: OpenNeuro ds000117
(Wakeman & Henson 2015; snapshot 2.0.0, DOI 10.18112/openneuro.ds000117.v2.0.0, licence CC0), magnetometers,
known Gaussian signal injected into real noise windows so the clean
risk is computable. Design: MS `plan/onthefly/DESIGN_ZC_ZD_IMPLEMENTATION_2026-10-12.md` Part D (authoritative)
and `plan/onthefly/RUNBOOK_MLST_MULTICHANNEL_UPGRADE_2026-10-12.md` §6. Implementation report:
MS `reviews/megz_kit_20261013/REPORT.md`. Deepthought prompt (D0-D4, stops before any freeze):
MS `plan/onthefly/AGENT_PROMPT_DEEPTHOUGHT_MEGZ_DEV_2026-10-13.md`.

**Status.** Revised after the independent review MS `reviews/megz_kit_review_20261013_01/INDEPENDENT_REVIEW.md`
(CHANGES REQUIRED; all recommendations adopted): Ledoit-Wolf Sigma_ref with the metric-corrected certificate
B_dagger, decimation 12 (91.67 Hz), C = 24, T = 20 (218.2 ms), per-subject theta_s rule, point-factor PZ2,
isotropic-vs-reference PZ5, effect-size G3, automatic G1 retest, budget exclusion, confirmation guards in every
CLI. Every value in `protocol_dev.json` is a development default, finalised at D4 from the 4 dev subjects only. No real MEG data has been touched by this kit; all tests use synthetic fixtures. A confirmation
run needs a frozen `protocol.json` (written at D4, new kit hashes, freeze commit, tag + OpenTimestamps per D4a).

## Files

| File | Role |
|---|---|
| `common.py` | record layout (C x T, vec index c + C t as T25/Y19), seeds (`t25.rng` construction), hashing, atomic JSON |
| `data_manifest.py` | D0: subject split (fixed: dev sub-01..04, confirmation sub-05..16), openneuro-py include list (snapshot 2.0.0), git-annex second-source check, `DATA_MANIFEST.json`, verify |
| `preprocess.py` | D1: raw (refuses SSS and MaxShield), mags, bads from channels.tsv, 1-40 Hz, decimate x12, +34.5 ms with an enforced STI101 cross-check, channel selection (runs 1-2), gap-rule noise windows, run partition, reference moments with the effective sample count, HDF5 + SHA-256, window budget; refuses confirmation subjects before the freeze |
| `estimators.py` | Ledoit-Wolf full, Kronecker flip-flop MLE, nearest Kronecker product, reference metric, arm estimators |
| `gates.py` | D2: G1 (raw, spatial condition, full condition; automatic drop-one retest), G2 (band + Marchenko-Pastur spread), G3 (effect size of positional means; gap-mean significance reported), lag-one dependence, G4, G5, E1 with the budget rule |
| `signal.py` | D3: evoked-peak spatial templates, bump + derivatives temporal basis, theta rotation (exactly separable at theta = 0), Gaussian latents with one scale per subject (tr K = target at theta = 0) |
| `population.py` | D3: general path (a_best with 64 starts, c_U, c_F, N*), certificate B_dagger (proof sketch in `b_dagger`), theta_s root search, m grid rule, eligibility; pinned Y19/TR09 path for the Kronecker sensitivity |
| `fit.py` | T25 recipe (moments, starts, structured, unrestricted, map_for, risks) under metric W; scores (clean risk in the Sigma_ref metric, Euclidean, semi-analytic, amplitude readout) |
| `runner.py` | verify-kit, write-dependencies, write-kit-hashes, preflight, smoke, run, summarize; PZ1-PZ5 reading engine; subject-level Bonferroni-t (df = n - 1); E1; refuses confirmation subjects without `--confirmation` and a frozen-protocol hash |
| `secondary_d8.py` | D8: held-out weighted residual and face-vs-scrambled decoding on real trials (descriptive) |
| `fixture.py` | synthetic fixtures (derived-level subjects with Kronecker / non-Kronecker noise; MNE RawArray BIDS-like raw data) |
| `benchmark.py` | timing at C = 24, T = 20 on synthetic data only |
| `protocol.py`, `protocol_dev.json` | protocol loader; development defaults (NOT FROZEN) incl. pinned dependency hashes |
| `tests/` | estimators, preprocessing (gap rule, partition leakage, raw fixture end to end, trigger check, MaxShield, guards), gates (incl. automatic G1 retest), signal/population (B_dagger, theta_s rule, pinned vs general path), fit/scores, reading rules, runner (smoke, resume, preflight refusals, kit identity), data manifest |

Reused code (imported unchanged, bytes pinned in `protocol_dev.json` -> `dependencies`, T25 closure checked by
the X3 guard before import): `t25_production_20260930/science.py`, `y19_20261012/predictions.py`
(`constants`, `certificate`), `p1nl_dev_20261006/tr09_first_order/core.py`, `successor_20260913/t20s/generator.py`
(`partial_trace_cap`), `t24_cumulant/core.py` (Mardia statistics).

## Environment

EXP root, clean checkout, no `.pyc` under `experiments/`:
```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
```
Python >= 3.10; numpy 2.2.6, scipy 1.15.3, mne 1.8.0, h5py, openneuro-py 2026.9.1, pytest (scikit-learn only
for one optional cross-check test). Tested here with Python 3.10.12, numpy 2.2.6, scipy 1.15.3, mne 1.8.0,
h5py 3.16.0, pytest 9.1.1.

## Commands

```bash
K=experiments.mlst2026.meg_z_20261013
python -m $K.runner verify-kit
python -m pytest experiments/mlst2026/meg_z_20261013/tests -q -p no:cacheprovider        # ~70 s
python -m $K.runner smoke --out /tmp/megz_smoke --workers 4                               # synthetic, ~30 s
# D0-D4 on the host: see the Deepthought prompt
python -m $K.data_manifest plan --target $RAW
python -m $K.preprocess --bids-root $RAW --subject sub-01 --out $DERIVED
python -m $K.gates --derived $DERIVED --subjects sub-01,sub-02,sub-03,sub-04 --bids-root $RAW --out gates.json
python -m $K.population --derived $DERIVED --subjects sub-01,sub-02,sub-03,sub-04 --gates gates.json --jobs 4 --out predictions_dev.json
python -m $K.runner run --out $OUT --derived $DERIVED --predictions predictions_dev.json --gates gates.json --subjects dev --workers 32
python -m $K.runner summarize --out $OUT
```
Preparation only (changes the kit): `runner write-dependencies`, then `runner write-kit-hashes`.
