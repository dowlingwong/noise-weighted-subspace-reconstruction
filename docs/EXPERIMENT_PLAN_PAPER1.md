# Experiment plan — Paper 1 (post-merge, 2026-09-02)

_Implementable plan for the evidence the merged Paper 1 still needs.  Derived
from the manuscript (`arXiv-v1-polished-extended-preprint`: §6 separable class,
§7 loss geometry, `07b_separable_evidence.tex`, `OPEN_ISSUES.md`), the
storyline plan (`plan/THREE_PAPER_STORYLINE_2026-09-02.md`) and
`paper_rps2026/lesson_from_sam.md` §4, §13, §21.  Companion checklist:
`docs/IMPLEMENTATION_CHECKLIST.md`.  Everything runs from this repository._

## 0. What is already evidenced (do not re-run)

| Manuscript claim | Archived evidence | Status |
|---|---|---|
| S1–S9 synthetic gates; S5 30-seed metric reversal | `results/sweeps/S*_sweep_10seeds.*`, `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.*` | done; provenance audit open (P1-E5) |
| CRESST within-channel hierarchy, same-rank contrast, hierarchical bootstrap | `evidence/cresst/20260704T013758Z_9c9f696ff6bc/`, `transfer_paper/data/cresst/` | done |
| GWOSC implementation identity vs calibration failure | `transfer_paper/tables/gwosc_*` | done |
| TIDMAD estimator identity, PSD calibration, file-0000 truth agreement, rank-2 negative, denoising score | `transfer_paper/data/tidmad/current/` | done |
| NFPA reversal (10 seeds), replication (20), white control, estimated-Σ (N_cal=2048), corrected angles | `results/rps2026/E1_nfpa_blockA`, `E2_nfpa_blockB`, `E9_blockA_replication`, `E10b_*`, `E11_corrected_angles` | done (enters Paper 1 after 29 Sep) |
| Planted channel/temporal projector recovery; parameter decoding | `results/rps2026/E8_semantics/separable_alpha_0`, `primary_alpha_0p1` | done |
| Rung-4 factored-attention pilot (2 seeds) | `results/rps2026/E5_*`, `E7_rung4_seed_replication.json` | done; boundary only |

## 1. Experiments to run

Each experiment names the manuscript hook it feeds, the code that exists, what
must be written, the output contract and the acceptance rule.  IDs are stable;
results go to `results/paper1/<ID>/` with a `manifest.json` carrying config
hash, git commit, seed list and SHA-256 of every output.

### P1-E1 — Figure 1 data product: learned versus planted factors
- **Feeds:** Paper 1 Figure 1 (lesson §10/§21.2); `07b_separable_evidence.tex` figure placeholder.
- **Question removed:** "does the metric change *what physical structure* is represented?" shown as a picture, not a number.
- **Exists:** E8 fits per seed (`E8_semantics/separable_alpha_0/`), `semantic.planted_subspace_errors`, generator with planted `V`, `Tb`.
- **Write:** `experiments/paper1/fig1_factors.py` — for seeds 0–9 at α=0, refit NFPA and Iso-MPCA from the frozen config, save `U_cU_c^T`, `V V^T`, `U_tU_t^T`, `T_bT_b^T` per seed and method; one noisy event, its two reconstructions, and the decoded amplitude/rise/decay per method. `scripts/make_paper1_fig1.py` renders: event → two factorizations → projector overlays → parameter recovery.
- **Output:** `results/paper1/P1-E1/{projectors.npz, event_example.npz, decoding.csv, manifest.json}`, `results/paper1/P1-E1/fig1.pdf`.
- **Accept:** channel projector error per seed reproduces the archived E8 values to 1e-6 (regression against `separable_alpha_0/summary.json`).

### P1-E2 — Cost of the separability prior: nonseparability sweep
- **Feeds:** §6.4 "approximation error versus metric error" — the sentence "measured by fitting the restricted and unrestricted classes at matched rank" currently has no number behind it.
- **Design:** α ∈ {0, 0.05, 0.1, 0.2, 0.4, 0.8}, colored regime, 10 seeds, matched rank 6; methods NFPA, CW-PCA, Iso-MPCA, PCA. Report held-out χ̄² gap NFPA−CW-PCA (the approximation term), NFPA−IsoMPCA (the metric term, must stay negative), mean principal angle NFPA↔CW-PCA, planted channel/temporal projector errors.
- **Exists:** `nfpa_confirmatory` generator (`nonseparability_alpha` field), `methods`, `metrics`, `semantic`.
- **Write:** `experiments/paper1/alpha_sweep.py` (config override via `dataclasses.replace`, new hash per α), seed-bootstrap intervals via `r9_common`.
- **Output:** `results/paper1/P1-E2/alpha_sweep.csv`, `summary.json`, `fig_alpha_cost.pdf`.
- **Accept:** at α=0 the NFPA−CW-PCA held-out gap is ≤ 0 within the interval (restriction inactive when the truth is separable); the gap is monotone non-decreasing in α; the metric term keeps its sign at every α.

### P1-E3 — Calibration-size sweep (E12 rerun, CPU)
- **Feeds:** §7 estimated-covariance paragraph — "one fixed condition, not a sweep" becomes a sweep (lesson P4).
- **Design:** N_cal ∈ {64, 128, 256, 512, 1024, 2048, 4096} noise-only events; separable Σ̂ estimated as in E10b; 10 frozen seeds; report reversal sign per seed, χ̄² and MSE degradation vs oracle, condition number κ(Σ̂⁻¹Σ).
- **Exists:** `r12_sweep.py`, `run_calibration_sweep.py`, `covariance_estimation_v2.py`, tests `test_round12_sweep.py`. Run 1 failed on the server (`docs/plans/rps2026/E12_RUN_1_FAILURE.md`, label defect since fixed in `ccd329f`).
- **Write:** nothing new; a local dispatch wrapper `experiments/paper1/run_e12_local.sh` that runs the existing runner with `--n-jobs 4`.
- **Output:** `results/paper1/P1-E3/` (mirrors the E12 contract).
- **Accept:** all tests in `test_round12_sweep.py` pass; reversal holds in ≥ 9/10 seeds at N_cal ≥ 512; the smallest N_cal at which any seed flips is reported, not hidden.

### P1-E4 — Three instruments, three questions (table compilation)
- **Feeds:** the Panda-style instrument table (lesson §21.1) replacing three subsections.
- **Design:** no new runs. Assemble from archived JSON: TIDMAD (χ²/dof 1.003, dispersion 1.014; file-0000 truth paired difference +0.0190 [0.0045, 0.0321]; secondary files' intervals), CRESST (seven channels, same-rank fractional reductions 1.6–62.0 %, hierarchical interval [1.22, 6.69]), GWOSC (global and 64-s PSD calibration failures; filter-equivalence pass).
- **Write:** `scripts/make_instrument_table.py` reading `transfer_paper/data/**` and `transfer_paper/tables/**`, emitting `results/paper1/P1-E4/instrument_table.csv` and `instrument_table.tex` (booktabs).
- **Accept:** every cell traces to a named file and key (the script prints the trace); numbers equal the manuscript's to the printed precision.

### P1-E5 — S5 provenance audit (OPEN_ISSUES blocking #0)
- **Feeds:** `OPEN_ISSUES.md` #0 — `S5_PROVENANCE_TASK.md` is referenced but does not exist anywhere.
- **Design:** (i) re-run the S5 confirmatory protocol at 30 seeds from the archived config and compare with `S5_confirmatory_30seeds.json` (digest and values); (ii) sweep every `S*.json` under `results/` and `evidence/` for the manuscript numbers 48.418/47.878/0.540/2367.99/5388.75/86.07 and report which file, if any, contains each; (iii) repeat the digest check for S1–S4, S6–S9 against `synthetic_summary.csv`.
- **Write:** `experiments/paper1/s5_provenance.py`; report `results/paper1/P1-E5/PROVENANCE_REPORT.md`.
- **Accept:** either the source run is located or the manuscript's substitution is confirmed reproducible; the report is the closure document for OPEN_ISSUES #0.

### P1-E6 — Rank-one corollary check at C = 8
- **Feeds:** Corollary 6.2 (rank-one NFPA ↔ joint-channel OF); the appendix currently has the C = k_c = 2 check only.
- **Design:** separable template s̃ = v tᵀ, isotropic whitened noise, population limit approximated at N = 20 000; fit (k_c,k_t)=(1,1) NFPA; report |cos θ| of each factor to (v, t) and the amplitude correlation of the code to the OF estimate; 10 seeds.
- **Write:** `experiments/paper1/rank_one_check.py`.
- **Accept:** |cos θ| ≥ 0.999 and amplitude r ≥ 0.999 in every seed.

### P1-E7 — Rung-4 strengthening (deferred; cluster)
- Option B (narrow the claim) was taken; the pilot stays a boundary paragraph. If reopened: ≥ 5 seeds per arm, matched compute, the E7 config, and the whitening-range check (E6) on every arm. Runbook: `docs/plans/rps2026/ROUND9_OVERNIGHT_REMOTE_RUNBOOK.md` Task B; do not run before the R/PS decision.

### P1-E8 — Noise-Laplacian PE uniqueness (author decision 2026-09-03: pursue; CPU)
- **Feeds:** new App. E.5 `app:pe-uniqueness` (corollary of Thm E.3 on the PE Gram kernel K = ΦΦ†) and one sentence after `prop:stationary-pe` in §3.4. Theory target, in the manuscript `TODO.md`: (a) token-axis symmetry T = G_noise ∩ permutations fixes the *basis* (Fourier, for stationary Σ) but leaves the spectral weights w_k free — sinusoidal and chain-Laplacian PEs are members of this family; (b) invariance under all of G_noise forces K = λΣ⁻¹, i.e. Φ = Σ^{-1/2}U — the noise-Laplacian PE, unique up to gauge; sinusoidal PE is (b) for white noise; (c) among Fourier encodings with weights J_k^{-γ} a linear readout attains the CRB iff γ = 1; (d) nonstationary Σ: the invariant encoding is Σ^{-1/2}, not a function of the index.
- **Question removed:** "is the precision-eigenbasis PE a construction or a consequence?" — it is the *only* G_noise-invariant encoding; shift symmetry alone cannot pick the weights.
- **Exists:** stationary PSD estimation (`src/tidmad/psd.py`), planted generator with coloured Σ (`experiments/synthetic/nfpa_confirmatory/generator.py`, `covariance.py`), whitening utilities (`r9_common`), OF/CRB reference (`src/of.py`). No attention model in the repo.
- **Write:** `experiments/paper1/pe_uniqueness.py` with four sub-runs, `experiments/paper1/pe_models.py` (numpy/torch-CPU two-layer single-head attention encoder with pluggable PE), `scripts/make_paper1_fig_pe.py`.
  - **E8a kernel identification (no learning).** Σ ∈ {TIDMAD file-0000 PSD (stationary, d ∈ {256, 512, 1024, 2048}), nonstationary Σ = D^{1/2} Σ_stat D^{1/2} with a slowly varying envelope D, and the planted-generator coloured Σ}. Compute (i) principal angles between the top-m eigenvectors of Σ⁻¹ and the DFT modes vs m and d (boundary term must vanish with d); (ii) invariance defect δ_G(K) = ‖Q†KQ − K‖_F/‖K‖_F averaged over 50 Haar-random Q = Σ^{1/2}UΣ^{-1/2} ∈ G_noise, for K ∈ {sinusoidal (Vaswani schedule, m = 64), chain-graph Laplacian eigenvectors (m = 64), noise-Laplacian Σ^{-1/2} truncated to m}; (iii) shift-invariance defect δ_T(K) under cyclic shifts. Prediction: δ_T ≈ 0 for all three under stationary Σ; δ_G ≈ 0 only for noise-Laplacian; under nonstationary Σ, δ_T ≠ 0 for noise-Laplacian while δ_G stays 0.
  - **E8b weight sweep (no learning).** Fourier features weighted J_k^{-γ}, γ ∈ {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5}; fixed linear readout (least squares on the encoded tokens) of the amplitude of the planted template at SNR 10; 2000 Monte-Carlo events × 5 seeds; report Var/CRB vs γ. Prediction: minimum = 1 at γ = 1 to MC precision; γ = 0 (sinusoidal) sits above by the mismatched-filter factor computable in closed form (report both). Repeat with the precision eigenbasis under nonstationary Σ: Fourier family has no γ that reaches 1; precision basis does.
  - **E8c learning ladder.** Two-layer single-head attention encoder, ≈50k parameters *matched across arms*, linear head predicting the amplitude and the k latent coefficients. Arms: PE ∈ {none, sinusoidal, learned, chain-Laplacian, noise-Laplacian}; inputs ∈ {raw, prewhitened}; Σ ∈ {stationary coloured, nonstationary}; train at SNR 10, test at SNR ∈ {3, 10, 30} (extrapolation arms); 5 seeds; n_train = 4000, n_test = 2000; CPU budget ≤ 10 min per arm. Metric: truth-referenced amplitude error / CRB and subspace principal angle (as P1-E1). Report parameter count and CPU inference time per event per arm. Pre-registered predictions: (1) prewhitened + stationary: sinusoidal ≈ noise-Laplacian — whitening already carries the weights; report as a boundary, not a win; (2) raw + stationary: noise-Laplacian closes most of the gap to the prewhitened arms; (3) nonstationary: noise-Laplacian beats every shift-based PE with or without whitening; (4) extrapolation to SNR 3/30 degrades the learned PE most. A prediction that fails is reported as such.
  - **E8d (optional, one seed).** E8c on TIDMAD frequency tokens (token-axis principle, file-0000 PSD), real data only as an illustration.
- **Output:** `results/paper1/P1-E8/{kernel_defects.csv, angles_vs_d.csv, gamma_sweep.csv, ladder.csv, resources.csv, manifest.json}`, `results/paper1/P1-E8/fig_pe.pdf` (three panels: δ vs PE type; Var/CRB vs γ; ladder).
- **Accept:** E8a δ_G(noise-Laplacian) < 1e-10 and δ_G(others) > 1e-2; E8b argmin γ = 1 with Var/CRB within 1 ± 0.03; E8c: predictions (1)–(3) each pass or are reported as failed; matched parameter counts within 2 %.
- **Do not claim:** anything about transformers at scale; the ladder is a mechanism check on a 50k-parameter model.

## 1b. Status after the 2026-09-03 container run

P1-E1 done (4 seeds, regression 3e-14); P1-E2 done at 5 seeds × 4 α (approximation term −0.0028 → −0.0014, monotone; metric term keeps sign; NFPA channel error 0.033 → 0.173 with α); P1-E4 done; P1-E5 closed; P1-E6 partial (n = 3000); P1-E3 not run. Summaries in `results/paper1/<ID>/`.

## 2. Order

P1-E5 (closes a blocking issue) → P1-E1 → P1-E2 → P1-E6 → P1-E3 (longest CPU
job; run last, in background) → P1-E4 (pure assembly) → P1-E8 (a, b first — no
learning, one hour; c after the theory statement is fixed, so the predictions
are pre-registered against the written proposition).  All are CPU-only and
run in `$HOME/venv` (numpy, scipy, pytest); none needs the cluster.

## 3. Manuscript wiring after the runs

| Experiment | TeX target |
|---|---|
| P1-E1 | Figure 1; `07b_separable_evidence.tex` placeholder |
| P1-E2 | one sentence + table in §6.4; row in Table 1 caption |
| P1-E3 | replaces "one fixed condition, not a sweep" in `07b` |
| P1-E4 | new Table in §7 replacing the three instrument subsections' summary paragraphs |
| P1-E5 | closes `OPEN_ISSUES.md` #0; updates `MANUSCRIPT_EVIDENCE_MAP.md` |
| P1-E6 | one sentence after Corollary 6.2; appendix table |
| P1-E8 | App. E.5 `app:pe-uniqueness` figure + one sentence after `prop:stationary-pe` |
