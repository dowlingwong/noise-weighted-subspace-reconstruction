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
# D0-D4 on the host: see the independently reviewed version-3 resume instructions
# DATA_MANIFEST must name the original verified D0 DATA_MANIFEST.json.
python -m $K.data_manifest plan --target $RAW
python -m $K.preprocess --bids-root $RAW --subject sub-01 --out $DERIVED --data-manifest $DATA_MANIFEST
python -m $K.gates --derived $DERIVED --subjects sub-01,sub-02,sub-03,sub-04 --bids-root $RAW --out gates.json --data-manifest $DATA_MANIFEST
python -m $K.population --derived $DERIVED --subjects sub-01,sub-02,sub-03,sub-04 --gates gates.json --jobs 4 --out predictions_dev.json
python -m $K.runner run --out $OUT --derived $DERIVED --predictions predictions_dev.json --gates gates.json --subjects dev --workers 32
python -m $K.runner summarize --out $OUT
```
Preparation only (changes the kit): `runner write-dependencies`, then `runner write-kit-hashes`.


## Development version 3: explicit stimulus timing contract

This development revision supersedes the version-2 assumption that snapshot 2.0.0 TSV stimulus onsets are uncorrected raw trigger times. The original kit commit `23ed9bf01cdad3bc2d1367dea5a4878db36cfb08`, identity `b277339a88b706e48ab6efd09aa4a9f95f2b479d46f7ce926c78c11e0eafa880`, and failed D1 attempt remain historical records. This revision authorizes no confirmation processing.

The explicit `timing_convention` binds ds000117, snapshot 2.0.0, its DOI, 1100 Hz and the verified D0 manifest content SHA-256. Every run source is checked against that manifest's byte size and SHA-256 before magnetometer processing. For all six runs, header/STI101/TSV preflight must pass before the first magnetometer load/filter. MNE first_samp is excluded from the relative stimulus sample origin. TSV onset must agree with TSV sample/sfreq within 50 microseconds plus 1e-10 seconds for arithmetic, with an exact ordered code/count bijection and **TSV stimulus sample = stable raw relative stimulus sample + 37**. There is no auto-detection, nearest-event rematching, tolerance widening, or offset fallback.

Physical onset is **stable raw relative STI101 sample / sfreq + 0.0345 seconds**. The positive delay is applied exactly once and is not rounded to 37 samples. TSV originals are retained in event metadata. The 37-sample TSV adjustment is empirically verified on development subjects; upstream converter code is available only on request, so its origin is not source-code verified. The primary Wakeman and Henson (2015) acquisition description reports a positive 34 ms projector delay; it does not independently establish 34.5 ms precision. Historical design wording “trigger −34.5 ms” is superseded by this explicit equation. Response exclusion times remain unchanged TSV times. Each derived manifest subject entry carries its explicit timing convention, manifest binding and all six run trigger checks; automatic G1 retest replaces these fields from the new subject summary rather than retaining stale diagnostics.

Stable low-byte stimulus plateaus require at least two raw samples. Each removed stimulus-code plateau must be exactly one sample, adjoining a retained stable stimulus on one side and zero on the other, with intermediate bits a subset of the stable code. Every admitted edge intermediate is recorded with sample, code, length, neighboring codes/lengths and classification. Isolated pulses, chains, non-subset codes, transitions between two stable stimuli and unclassified removals STOP. Genuine two-sample stimuli remain. Overlapping response bits do not alter the low-byte stimulus plateau.

Synthetic raw fixtures explicitly declare `synthetic-uncorrected-v1`, offset zero, without a dataset manifest. Their onsets lie on the raw sample grid. This is a distinct test contract, never a fallback for real snapshot data. Legacy trigger field aliases remain available. All scientific delay, preprocessing settings, run partitions, gap rules, thresholds, predictions, seeds and confirmation refusal remain unchanged.

An accepted plan is not implementation approval. Candidate tests/hash and a host preflight require independent review before D1–D4 starts. Use fresh development attempt output/log paths, preserving all failed artifacts. Do not create protocol.json, tags or timestamps; do not preprocess, gate, predict or fit sub-05 through sub-16 before a later author freeze and independent review.
