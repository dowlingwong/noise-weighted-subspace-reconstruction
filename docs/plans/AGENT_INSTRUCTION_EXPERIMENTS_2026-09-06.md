# Instruction for the implementation agent — Paper 1 experiments (2026-09-06)

_You are building the experimental evidence for the manuscript "Noise Covariance Defines
Reconstruction Geometry" (repo `arXiv-v1-polished-extended-preprint`, called **MS** below) in the
experiment repository `noise-weighted-subspace-reconstruction` (called **EXP** below). This
document is self-contained: it tells you what to run, in which order, what each run must
produce, and the numeric standard that counts as passed. The paper-level rationale is
`MS/plan/EXPERIMENT_PLAN_2026-09-06.md`; the run-level spec for the already-planned P1-E1…E10 is
`EXP/docs/EXPERIMENT_PLAN_PAPER1.md`. Where the two disagree, this document wins for the tasks it
covers._

## 0. Ground rules (apply to every task)

1. **Environment.** Work in `EXP`. Use the project venv: `uv sync --extra dev` then `uv run …`
   (fallback: `.venv/bin/python`). Before anything else run `uv run pytest -q` and record the
   pass count in your report; do not proceed with a red suite unless the failure is in a module
   you will not touch, and say so.
2. **Freeze before you look.** Every task card below has a "Predeclared" block. Copy it verbatim
   into `results/paper1/<ID>/PREDECLARED.md` *before* the first evaluation run, together with the
   SHA-256 of the config you will use. If you change anything after seeing a result, the run is
   *post-freeze* and must be labelled so in every output and in the report. Never tune a
   threshold, seed list, file choice, window or rank into a pass.
3. **Seeds.** Synthetic: ≥ 10 seeds unless the card says otherwise, seeds `0…9`. Real data: all
   channels / files the card names, no selection by outcome. Bootstrap intervals: 10 000
   resamples over the unit named in the card, seed `20260906`.
4. **Outputs.** Everything goes to `results/paper1/<ID>/` with `manifest.json` written by
   `experiments/paper1/common.write_manifest` (config hash, git commit, seeds, SHA-256 of every
   output). CSV column names are fixed by the card; add columns, never rename.
5. **Report negatives.** A failed prediction is a result. Print it in the summary with the same
   prominence as a pass. Do not delete a run; archive it under `results/paper1/<ID>/attempts/`.
6. **Do not** print or reuse anything from `QP_simulator/`, `paper2/`, the internal K-α data, or
   the `noise_module` component labels (SQUID/Johnson/TD/Er). **Do not** cite
   `results/rps2026/*` in the manuscript before 29 Sep 2026 (double-blind R/PS review); you may
   use it as a regression reference.
7. **Registration.** A number is citable only after: the result file is copied to
   `MS/transfer_paper/data/paper1/<ID>/`, added to `MS/transfer_paper/BUNDLE_MANIFEST.json`
   (use `MS/transfer_paper/scripts/refresh_bundle.py`, or append entries with `bytes`, `path`,
   `sha256`, `source`), and given a row in `MS/transfer_paper/MANUSCRIPT_EVIDENCE_MAP.md` with
   columns *claim area · evidence · support level · safe implication · boundary*. Both repos
   carry a copy of `transfer_paper/`; write to **both** (this duplication is a known risk,
   `MS/OPEN_ISSUES.md` Resolved list).
8. **Data.** Public data live under `PAPER1_DATA_ROOT` (default `/ceph/dwong/paper1_dataset`)
   with `cresst/`, `tidmad/`, `gwosc/` subtrees. Never commit `.npy`, `.h5`, strain files.
   Checksum every downloaded file (`EXP/DATASETS.md` records a corrupted TIDMAD download).
9. **Cost.** Report wall-clock and machine for each run in `manifest.json` (`extra={"walltime_s":…,
   "host":…}`) — the manuscript reports parameter counts and cost wherever two classes compare.
10. **Final deliverable.** `EXP/results/paper1/RESULTS_REPORT_2026-09.md` (template in §4) plus
    the registrations of rule 7. The report is the only thing the author reads first.

## 1. Definition of done

The task is complete when all of the following hold:

- [ ] `uv run pytest -q` passes (count recorded), including the new tests listed per card.
- [ ] Every **A-priority** card (E-0, E2*, E6*, E9*, E11, E12, E13, E4b) has a `manifest.json`,
      a `PREDECLARED.md`, a `summary.json`, and a pass/fail line in the report.
- [ ] Every B-priority card attempted has the same, or a one-paragraph reason it was not run.
- [ ] Every result the report calls citable is registered per rule 7 in **both** repos.
- [ ] `RESULTS_REPORT_2026-09.md` follows the §4 template and contains no number that is not in
      a manifest-checksummed file.
- [ ] No file under `results/` or `transfer_paper/` was deleted or overwritten; superseded runs
      live under `attempts/`.

## 2. Task cards

### E-0 — Environment and regression baseline (do first; 30 min)

**Steps.** `uv sync --extra dev`; `uv run pytest -q`; `uv run python
experiments/paper1/fig1_factors.py --seeds 0 --smoke`; confirm `results/paper1/P1-E1/manifest.json`
`max_regression_abs_diff` ≤ 1e-6 against `results/rps2026/E8_semantics/separable_alpha_0`.
**Pass.** Test suite green; smoke run reproduces the archived E1 projector errors to 1e-6.
**Fail.** Stop and report; nothing downstream is trustworthy.

---

### E2* — Nonseparability α-sweep at the frozen protocol (A; CPU ≈ 1 h)

**Purpose.** §6.4 "approximation error versus metric error" and the NFPA optimisation term.
**Code.** `experiments/paper1/alpha_sweep.py` (exists). Add reporting of the restart spread.
**Predeclared.** α ∈ {0, 0.05, 0.1, 0.2, 0.4, 0.8}; seeds 0–9; methods NFPA, CW-PCA, Iso-MPCA,
IsoPCA; rank 6 = (2,3); `PrimaryConfig` unchanged (`n_restarts = 10`); metric
`test_normalized_chi2`; intervals: seed bootstrap.
**Steps.** (1) Modify `alpha_sweep.py` so that for each (α, seed) the NFPA fit records, over its
10 ALS restarts, `chi2_restart_min`, `chi2_restart_max`, `projector_err_c_restart_sd`,
`projector_err_t_restart_sd` (the optimisation term). (2) Run
`uv run python experiments/paper1/alpha_sweep.py --alphas 0 0.05 0.1 0.2 0.4 0.8 --seeds 0 1 2 3 4 5 6 7 8 9`.
(3) `scripts/make_alpha_figure.py` (write; matplotlib) → `fig_alpha_cost.pdf`: x = α; panels:
approximation term NFPA−CW-PCA with interval; metric term NFPA−Iso-MPCA; channel/temporal
projector errors; restart spread as error bars.
**Outputs.** `results/paper1/P1-E2/alpha_sweep.csv` (columns: `alpha, seed, method,
test_normalized_chi2, raw_mse, angle_to_cwpca_deg, projector_err_c, projector_err_t,
chi2_restart_min, chi2_restart_max, projector_err_c_restart_sd, projector_err_t_restart_sd`),
`summary.json`, `fig_alpha_cost.pdf`, `manifest.json`.
**Pass.** (a) At α = 0 the NFPA−CW-PCA held-out gap interval includes ≤ 0 (restriction inactive
when truth is separable). (b) The approximation term is non-decreasing in α with each successive
interval overlapping or above the previous. (c) The metric term NFPA−Iso-MPCA is negative with
interval excluding zero at every α. (d) Restart spread of χ̄² ≤ 10 % of the |metric term| at
every α — if larger, the optimisation term is *material* and must be printed in §6.4 as such
(this is a reportable outcome, not a failure).
**Wiring.** One sentence + a 6-row table in §6.4 (`MS/05b_separable_subspaces.tex`, after the
"four sources" paragraph); Table 1 caption gains the parameter counts already there.

---

### E6* — Rank-one corollary at C = 8, population regime (A; CPU ≈ 20 min)

**Code.** `experiments/paper1/rank_one_check.py` (exists).
**Predeclared.** n = 20 000 events, SNR 10, seeds 0–9, (k_c, k_t) = (1, 1), separable template.
**Steps.** `uv run python experiments/paper1/rank_one_check.py --seeds 0 1 2 3 4 5 6 7 8 9 --n 20000`;
if memory fails, chunk the second-moment accumulation (do not lower n). Additionally run
n ∈ {1000, 3000, 10000, 20000} at 3 seeds and record the temporal |cos θ| vs n so the finite-n
approach to 1 is a curve.
**Outputs.** `results/paper1/P1-E6/rank_one_check.csv` (`seed, n, abs_cos_channel,
abs_cos_temporal, amp_corr_code_vs_of, amp_corr_code_vs_truth`), `summary.json`.
**Pass.** At n = 20 000: min over seeds of |cos θ_c| ≥ 0.999, |cos θ_t| ≥ 0.999,
r(code, OF) ≥ 0.999. If |cos θ_t| plateaus below 0.999, report the plateau and the fitted
1 − |cos θ_t| ∝ n^{−β} exponent — that is the finite-sample statement Cor. 6.3 needs.
**Wiring.** One sentence after Cor. 6.3; appendix line in App. E.3.

---

### E9* — Physical-variable probes at 10 seeds (A; CPU ≈ 30 min)

**Code.** `experiments/paper1/latent_probes.py` (exists; 4-seed result archived).
**Predeclared.** seeds 0–9, α = 0, rank 6; methods NFPA, CW-PCA, Iso-MPCA, IsoPCA; ridge probes
fit on train, scored on test; variables amplitude, rise_time, decay_time, secondary.
**Steps.** `uv run python experiments/paper1/latent_probes.py --seeds 0 1 2 3 4 5 6 7 8 9`.
**Outputs.** `results/paper1/P1-E9/probes.csv` (`seed, method, variable, probe_r2,
max_abs_corr, participation`), `corr_matrices.npz`, `summary.json`.
**Pass.** For amplitude and rise_time, the across-seed interval of (NFPA − Iso-MPCA) and
(CW-PCA − IsoPCA) probe R² excludes zero. Decay-time R² is reported whatever it is (the 4-seed
run gave 0.15–0.28: a boundary, print it).
**Wiring.** `MS/07b_separable_evidence.tex` "Which axes carry the physics" table; Fig. 1 panel (a).

---

### P1-E11 — Excited support: does the data or the initialisation fix T_S^⊥? (A; CPU ≈ 1 h)

**Purpose.** Turn Prop. 7.3 (`MS/06_loss_geometry.tex`, `prop:excited-support`) into a
measured statement inside Paper 1.
**Code to write.** `experiments/paper1/excited_support.py`. Reuse
`experiments/oracle/oracle_cov/generator.make_training_set` (planted forward model with named
latents) and `experiments/oracle/oracle_cov/tangent.tangent_geometry(ts, excited=…)` (whitened
T_S basis and T_S^⊥ projector by central-difference Jacobians), and
`experiments/synthetic/nfpa_confirmatory/methods.fit_cw_pca / fit_nfpa`; `methods.als_fit`
already takes `seed` and `n_restarts` (restart r uses `rng(seed*1000 + r)`), so per-restart
fits are obtained by calling `als_fit` with `n_restarts=1` and `seed` = 1000·seed + r; for
CW-PCA (closed form, no initialisation) the complement is fixed by the eigen-solver's ordering
of the noise eigenvalues — record it but expect P-b to be decided by NFPA.
**Predeclared.** Two training supports: S_full = {amplitude, rise_time, decay_time, secondary}
and S_restricted = {amplitude, rise_time} (decay_time and secondary held at their means in the
training draw — implement by overriding `sample_latents` output before `build_coefficients`).
Rank k = 6 for CW-PCA; (2,3) for NFPA. Seeds 0–9 (data); R = 8 initialisations per seed.
Quantities per (support, seed, method): (i) `angle_to_TS_max`: largest principal angle between
the learned whitened subspace and T_S (computed with `tangent_geometry` at 64 training latents);
(ii) `complement_spread`: mean pairwise largest principal angle, across the R restarts, of the
learned subspace's component orthogonal to T_S; (iii) `held_out_chi2` on a test set drawn from
S_full (so the unexcited coordinates are present at test time).
**Predictions.** P-a: `angle_to_TS_max` < 5° for both supports and both methods (the excited
span is always recovered). P-b: under S_restricted, `complement_spread` > 30° (the complement
is initialisation-determined); under S_full, `complement_spread` < 5°. P-c: under
S_restricted the held-out χ̄² on S_full test data exceeds the S_full-trained value by an amount
whose interval excludes zero, and is *not* reduced by more training data (rerun at 4× N_train:
the gap must not shrink by more than 10 %).
**Outputs.** `results/paper1/P1-E11/excited_support.csv` (`support, seed, method, restart,
angle_to_TS_max, held_out_chi2, n_train`), `complement_spread.csv` (`support, seed, method,
complement_spread_deg`), `summary.json`, `fig_excited_support.pdf` (two panels: angle-to-T_S
and complement spread by support).
**Pass.** P-a, P-b, P-c each pass or are reported failed with the numbers.
**Wiring.** One paragraph + a 4-row table after Prop. 7.3 in `MS/06_loss_geometry.tex`; the
proposition is restated as exact invariance under rotations within T_S^⊥ (review Major 3);
App. E.5 gets the table.

---

### P1-E12 — Tied versus untied linear autoencoders (A; CPU ≈ 2 h)

**Purpose.** The only ablation the tied-adjoint claim (§5.4, Rem. E.3) needs.
**Code.** Exists: `src/noise_geometry/autoencoders/trained.train_weighted_linear_ae` (tied, Σ⁻¹
loss, L-BFGS/Adam), `train_whitened_linear_ae`, `linear.tied_linear_ae_closed_form`,
`experiments/synthetic/nfpa_confirmatory/methods.fit_iso_ae_gradient`. **Write**
`src/noise_geometry/autoencoders/untied.py` with `train_untied_linear_ae(samples, weights, rank,
*, loss="weighted"|"mse", optimizer, seed, max_iter, l2=0.0)` optimising encoder E (k×d) and
decoder D (d×k) independently under L = mean (x − DEx)ᵀ M (x − DEx), M = Σ⁻¹ or I; return D, E,
the projector Π = DE, obliqueness `‖Π − Π*_Σ‖_F / ‖Π‖_F` where Π*_Σ = D(DᵀΣ⁻¹D)⁻¹DᵀΣ⁻¹ is the
Σ-orthogonal projector onto range(D), and principal angles of range(D) to the EMPCA subspace.
Add `tests/test_untied_linear_ae.py`: at convergence with Σ = I, untied MSE recovers the PCA
subspace to 1e-6.
**Predeclared.** Data: S3 configuration (`configs/synthetic/s3_ae_bridge.yaml`, colored noise)
and the planted generator at α = 0 (vectorised, rank 6). Arms: (a) tied + Σ⁻¹ loss; (b) untied +
Σ⁻¹ loss; (c) tied + MSE; (d) untied + MSE. Seeds 0–9; optimiser L-BFGS to tolerance 1e-12
(`converged` arm) **and** a second pass stopped at 200 iterations (`early` arm); l2 = 0.
Metrics per arm: `angle_to_empca_max_deg`, `obliqueness`, `held_out_weighted_residual`,
`held_out_raw_mse`, `iterations`.
**Predictions.** (a) converged: angle < 0.01°, obliqueness < 1e-6. (b) converged: angle <
0.01° (same subspace at the optimum — the untied optimum of the weighted loss is still the
Σ-orthogonal projector) but early-stopped: obliqueness > 1e-2 and angle > 1° in ≥ 8/10 seeds.
(c), (d) converged: angle to EMPCA > 30° under the S3 colored covariance (they find the PCA
subspace); angle < 0.01° in the S4 white control (run the white control too, seeds 0–4).
**Outputs.** `results/paper1/P1-E12/ae_ablation.csv` (`dataset, seed, arm, stopping,
angle_to_empca_max_deg, obliqueness, held_out_weighted_residual, held_out_raw_mse, iterations,
walltime_s`), `summary.json`, `fig_ae_ablation.pdf`.
**Pass.** Each prediction passes or is reported failed. Note in the report whether (b)
converged equals (a): if yes, the paper's sentence "untied encoders generally define oblique
projections" must be qualified to "away from the optimum / under regularisation"; if (b) does
*not* converge to (a), report the landscape difference — both are results.
**Optional real-data arm (cluster).** On CRESST, add `linear_ae_mse` (tied, MSE loss) to
`baselines` in a copy of `configs/cresst/cresst_validation.yaml` (rank 6, otherwise identical,
new `experiment_id: P1_E12_CRESST_AE`); report per channel the angle between the MSE-trained
and Σ⁻¹-trained AE subspaces and the weighted residual of each. Pass: Σ⁻¹-AE weighted residual
< MSE-AE in ≥ 6/7 channels with the equal-channel hierarchical interval excluding zero;
otherwise report.
**Wiring.** One sentence in §5.4 + table in App. C; real-data arm → one sentence in §9.3.

---

### P1-E13 — TIDMAD operating-point sweep, zero-predictor baseline, norm-matched control (A; CPU, I/O-bound ≈ 3–6 h)

**Purpose.** Review Major 2: the headline compares two estimators both below the zero predictor
(0.0747 < 0.0880 < 0.1209 on file 0000). The paper needs either an operating point above the
noise floor or an honest floor statement, plus an informative random-basis control.
**Code.** `scripts/tidmad_phases.py` (phase2, phase6), `src/noise_geometry/tidmad/analysis.py`
(`run_metric_reversal`, `build_window_set`), `src/noise_geometry/tidmad/negative_controls.py`.
Write (1) `experiments/paper1/tidmad_window_sweep.py` looping `window_samples` and files and
calling `run_metric_reversal` with everything else from `configs/tidmad/phase2_metric_reversal_real_noise.yaml`;
(2) in `negative_controls.py`, add control `random_weighted_basis_norm_matched`: draw a random
Σ-orthonormal basis of the same rank and rescale the projected reconstruction so its whitened
energy equals the fitted basis's mean whitened reconstruction energy on the fit windows; keep
the existing control unchanged; (3) add `zero_predictor_amplitude_rmse` and
`zero_predictor_paired_difference` (zero estimate vs EMPCA, with the same bootstrap) to the
Phase 2 output.
**Predeclared.** Files `abra_validation_0000.h5`, `0005`, `0010` (the three already used, no new
selection); `window_samples` ∈ {4096, 16384, 65536}; `n_windows` adjusted so the same sample
range is covered (512 → 128 → 32; if 32 windows give < 10 evaluation windows, report the row
as underpowered rather than lengthening the range); rank 2; fit fraction 0.7; guard 4 windows;
bootstrap 300 window-level and block-level (block 16 → 4 → 1); seed 20260727. Metric: paired
absolute amplitude error PCA − EMPCA; plus amplitude RMSE of PCA, EMPCA, zero predictor.
**Predictions.** T-a: at every (file, window) the three-way ordering raw-MSE → PCA, weighted →
EMPCA, truth → EMPCA holds (direction check). T-b: there exists a window length at which
EMPCA amplitude RMSE < zero-predictor RMSE on file 0000 (the SNR floor is crossed) — if not,
report the floor as "not reached at 65 536 samples". T-c: the norm-matched random basis
degrades the primary metric with interval excluding zero on file 0000 at 4096 (the control is
informative where the saturated one was not).
**Outputs.** `results/paper1/P1-E13/window_sweep.csv` (`file, window_samples, n_fit, n_eval,
rmse_pca, rmse_empca, rmse_zero, paired_diff_pca_minus_empca, ci95_low_window, ci95_high_window,
ci95_low_block, ci95_high_block, direction_ok`), `controls_0000_4096.json`, `summary.json`,
`fig_tidmad_floor.pdf` (RMSE of the three estimators vs window length, per file).
**Pass.** T-a, T-b, T-c each pass or fail explicitly. Whatever T-b gives, the report states the
smallest window at which EMPCA beats zero, or that none does.
**Wiring.** Table 4 in `MS/07_experiments.tex` gains a zero-predictor row and a window column;
§9.4 and Limitations get the floor sentence; if T-b fails on every file, the abstract's
"amplitude-RMSE reduction 27.2 %" sentence is rewritten to state ordering only (the author
does this; you flag it).

---

### P1-E4b — Hyperparameter table (A; desk, 1 h)

**Steps.** Write `scripts/make_hyperparameter_table.py` that reads, and prints the source path
for, every value: synthetic S1–S9 (`configs/synthetic/*.yaml`: d, N, SNR, noise kind, rank,
seeds), planted generator (`nfpa_confirmatory/config.py`: C, d, (k_c,k_t), α, SNR,
N_train/val/test, restarts, ALS tol, N_calibration), CRESST (`configs/cresst/cresst_validation.yaml`:
rank, shrinkage, covariance model, split fraction, template-bank grid, baseline fraction),
GWOSC (`configs/gwosc/*.yaml`: window, PSD estimator, FIR length, radii), TIDMAD
(`configs/tidmad/phase*_real_noise.yaml`: window, rank, n_windows, split, guard, bootstrap,
block). Emit `results/paper1/P1-E4b/hyperparameters.tex` (booktabs, one row per value, a
"source" column) and `.csv`.
**Pass.** Every row has a source path that exists; the author spot-checks five rows.
**Wiring.** §9.1 of the manuscript (or an appendix table with a one-line pointer).

---

### P1-E10 — Anisotropy predictor across CRESST channels (A; cluster, half a day)

Follow `EXP/docs/EXPERIMENT_PLAN_PAPER1.md` P1-E10 exactly (predictor
a = 1 − (tr Σ_S/k)/λ_max(Σ_S) frozen before looking; outcome = archived same-rank fractional
reduction per channel from `MS/transfer_paper/data/cresst/sensitivity/…/paper_facing_table.csv`).
Write `scripts/gain_predictor.py`. **Pass:** Spearman ρ over the seven channels reported with a
permutation interval (10 000 permutations); the sign and the interval are the result, whatever
they are. **Wiring:** predictor column in the §9 instrument table.

---

### CRESST rank sweep and subset reporting (B; cluster, hours)

**Predeclared.** Copies of `configs/cresst/cresst_validation.yaml` with `rank ∈ {2, 4, 8}`
(`experiment_id: P1_CRESST_RANK_k`), nothing else changed; subset = lowest SNR tercile per
channel defined by the single-template OF amplitude / published resolution on the *calibration*
segment (so the cut never sees held-out data).
**Outputs.** `results/paper1/P1-CRESST-RANK/per_channel.csv` (`channel, rank, subset,
r_of, r_multiof, r_pca, r_empca, r_ae, frac_reduction_pca_empca, n_events`), hierarchical
intervals as in the archived sensitivity analysis.
**Pass.** Reported, not gated: the same-rank fractional reduction at each rank and in the
low-SNR tercile. If the k = 6 headline (21.6 %) is an outlier among ranks, say so.
**Wiring.** §9.3 one sentence; appendix table.

---

### P1-E14 — Truth-referenced gravitational-wave arm (B; 1–2 days)

**Purpose.** Give the nonstationary regime a recorded truth (review Major 5, plan §3.B).
**Data.** Route 1: GWOSC O1/O2 transient hardware injections
(`https://gwosc.org/o1_inj/`, `https://gwosc.org/o2_inj/`): fetch the injection table (GPS
time, detector, waveform family and parameters), download 64 s of 4 kHz strain around ≥ 10
successful CBC injections per detector via `gwosc`/`gwpy`, regenerate each injected waveform
with `lalsimulation` or `pycbc.waveform` from the published parameters (record the approximant
and the parameters in the manifest; state that the injected waveform is regenerated, not
retrieved). Route 2 (fallback if parameters are incomplete): the MLGWSC-1 generator
(`github.com/gwastro/ml-mock-data-challenge-1`, dataset 4 = real O3a noise + injections with
known parameters) — generate 100 injections with the challenge's own script and keep the
parameter file as truth.
**Design.** Per injection: window 8 s centred on the injection; PSD from off-source calibration
windows (chronological, as in `src/noise_geometry/gwosc/analysis.py`); signal class = span of
the injected template and its time-derivatives up to order 3 (rank 4; this mimics the "known
template + jitter" regime and avoids fitting a bank); estimators PCA (raw MSE), EMPCA (Σ⁻¹) at
matched rank fitted across injections; truth error = whitened distance between reconstruction
and the regenerated injection; also report the amplitude (SNR) estimate error.
**Predeclared.** Injections chosen by the published "successful" flag only; both detectors;
paired bootstrap over injections, 10 000 resamples.
**Predictions.** G-a: EMPCA truth error < PCA truth error with interval excluding zero when
the off-source calibration test passes for that window (score dispersion in [0.8, 1.2]);
G-b: where the calibration test fails, the difference is not significant or reverses — report
the two strata separately (this is the paper's own prediction: the metric helps when the
covariance model is adequate).
**Outputs.** `results/paper1/P1-E14/injections.csv` (`detector, gps, approximant, snr_published,
calibration_ratio, truth_err_pca, truth_err_empca, amp_err_pca, amp_err_empca`), `summary.json`,
`fig_gw_truth.pdf`.
**Pass.** G-a and G-b reported per stratum with intervals; any "insufficient injections"
outcome is reported with the count.
**Wiring.** New paragraph in §9.3 (GWOSC) turning the boundary into a truth-referenced
stratified result; instrument table gains a "truth" cell for GW.

---

### P1-E15 — HPGe regime: Majorana Demonstrator AI/ML release (B; 2–3 days)

**Data.** arXiv:2308.10856; DataPlanet persistent id `perma:83.ucsddata/UQWQAV`, partial copy
on Zenodo 8257027. HDF5 chunks of 65 000 events (≈ 1.8 GB); fields: raw waveform (3 800
samples; baseline and tail pre-summed ×4, rising edge at 10 ns), energy label (keV), detector
id, run, `tp0`, four PSD labels. Terms: papers may be published without collaboration
co-authorship; cite the release and its reference [3]. Use **one** chunk; record its SHA-256.
**Design.** Per detector with ≥ 2 000 events: noise PSD from the pre-trigger baseline
(first ~1 000 samples after de-presumming by linear interpolation, as the release describes)
of the chronologically earlier 70 % of events; template = mean aligned pulse at `tp0` from the
same segment; estimators single-template OF, PCA, EMPCA at k ∈ {1, 2, 4, 6}, tied Σ⁻¹-AE at
k = 6; evaluation on the later 30 %. Metrics: held-out weighted residual per estimator;
amplitude per event from the gauge-invariant template projection (Eq. gauge-amplitude);
**consequence metric**: FWHM of the ²⁰⁸Tl 2 614.5 keV line in the amplitude spectrum per
estimator after a linear calibration fitted on the calibration segment (line energies from
nature: 583.2, 2614.5 keV). Reuse `src/noise_geometry/cresst/analysis.py` by adding a
`release_format: majorana` loader in `src/noise_geometry/cresst/loader.py` (or a sibling
`src/noise_geometry/hpge/`).
**Predeclared.** Detectors: all with ≥ 2 000 events in the chunk, none excluded by outcome;
rank list fixed; FWHM by a Gaussian + linear background fit in ±15 keV; paired bootstrap
over events for residuals, over detectors for the FWHM contrast.
**Predictions.** H-a: same-rank PCA vs EMPCA weighted residual favours EMPCA in ≥ 80 % of
detectors. H-b: FWHM(2614.5) for EMPCA at k = 2–4 ≤ FWHM for single-template OF (the learned
subspace absorbs rise-time variation); H-c: FWHM for PCA at matched rank ≥ FWHM for EMPCA.
Report all three with intervals whatever they are; a failure of H-b is physically informative
(charge-collection variation is not a linear subspace).
**Outputs.** `results/paper1/P1-E15/per_detector.csv` (`detector, n_cal, n_eval, rank,
estimator, weighted_residual, fwhm_2614_keV, fwhm_ci_low, fwhm_ci_high`), `summary.json`,
`fig_hpge.pdf`.
**Pass.** H-a…H-c reported; ≥ 5 detectors analysed.
**Wiring.** New §9 subsection "HPGe calibration lines"; instrument table row; the FWHM contrast
is the first *physical-units* consequence Paper 1 can claim on public data.

---

### P1-E8a / E8b — PE uniqueness numerics (B; CPU ≈ 1 h; only after the corollary is written)

Follow `EXP/docs/EXPERIMENT_PLAN_PAPER1.md` P1-E8a and E8b verbatim (kernel invariance defects
δ_G, δ_T for sinusoidal / chain-Laplacian / noise-Laplacian encodings on the TIDMAD file-0000
PSD and a nonstationary Σ from `noise_module`; γ-sweep Var/CRB). **Pass:** δ_G(noise-Laplacian)
< 1e-10 and δ_G(others) > 1e-2; argmin γ = 1 with Var/CRB within 1 ± 0.03. Do **not** run E8c.

## 3. Order

E-0 → E2* → E6* → E9* → E11 → E12 (synthetic) → E4b → E13 → [cluster: E10, CRESST rank sweep,
E12 CRESST arm] → E14 or E15 (ask the author which first; default E14) → E8a/b. Run E13 in the
background while doing E11/E12 if the data are present.

## 4. Report template — `results/paper1/RESULTS_REPORT_2026-09.md`

```
# Paper 1 experiment report — <date>
Environment: <host>, git <commit>, pytest <passed>/<total>.

## Summary table
| ID | status (pass / fail / partial / not run) | headline number(s) with interval | file | registered in transfer_paper (Y/N) |

## Per experiment
### <ID> — <title>
Predeclared: <copied verbatim, with config hash>
Observed: <one paragraph, numbers with intervals>
Pass/fail per prediction: <list>
Post-freeze changes: <none | list with reason>
Cost: <walltime, host>
Manuscript sentence(s) the author can paste (numbers only from the files above): <…>
What this does NOT show: <one or two lines>

## Open problems found
<anything that blocks the author: data missing, a gate that cannot be met, a defect in existing code>
```

Every number in the report must be copy-pasted from a file whose SHA-256 is in a manifest. If a
number cannot be traced, write "[untraced — do not cite]" next to it.

## 5. What counts as passed, in one paragraph

An experiment is **passed** when its predeclared predictions are each individually true at the
stated tolerance on the stated seeds/files, its outputs carry a manifest with config hash and
checksums, and its numbers are registered in both `transfer_paper/` copies. It is **failed** when
a prediction is false — and a failed experiment is still complete if it is reported with the same
files and the same care. It is **incomplete** when a step was skipped, a threshold was moved
after a result was seen without being labelled post-freeze, a number cannot be traced, or a
negative outcome was left out of the report. Only complete experiments — passed or failed —
are handed to the author.
