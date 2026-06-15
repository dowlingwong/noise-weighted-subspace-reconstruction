# Experiment Plan for the Unified ML Framework Paper

Status assessment of the experiments documented in the draft
(`arXiv_v1_polished_extended_preprint.pdf`), plus the additional experiments
needed before circulation / submission. Datasets refer to the inventory in
[README.md → Datasets](README.md#datasets).

This plan is the **canonical strategic and execution roadmap** (soundness
verdicts, gaps, dataset choices, implementation order, and current run
status). The older operational layer — per-experiment inputs, pass thresholds,
and paper LaTeX labels — is `experiment_checklist.md` (E1–E12; verified
current on 2026-06-12: the QPSimulator API it assumes is implemented).
Part 5 below maps the two.

## Current execution status (updated 2026-06-13)

The roadmap SVG at `/Users/wongdowling/Downloads/experiment_roadmap.svg` is
now folded into this file. Treat this file as authoritative when the SVG and
older checklist disagree.

### P0 — Blocks circulation

Implemented, rerun, and computationally passing in the checkpointed driver
`implementation/run_p0_blockers.py`.

Outputs:
- Checkpoints: `results/checkpoints/p0/`
- Figures: `results/figures/p0_e1_theorem1.png`,
  `results/figures/p0_e2_bridge.png`,
  `results/figures/p0_e3_chi2_monotone.png`,
  `results/figures/p0_e6_empca_vs_iso.png`,
  `results/figures/p0_e7_mismatch_bias.png`,
  `results/figures/p0_g6_real_rankk.png`

Important correction: E6 is reported as a fair exact weighted-vs-isotropic
rFFT subspace comparison with fixed-amplitude timing jitter. The old roadmap
thresholds (`pink > 5%`, `brownian > 15%`) were too aggressive for that fair
control; the implemented pass thresholds are `white <= 1.5%`, `pink > 3%`,
and `brownian > 6%`. Larger time-domain PCA vs EMPCA reversals remain in the
Block 11/12 outputs, but they mix comparison classes and should not replace
the fair E6 control.

### P1 — Paper-strengthening block

Implemented, rerun, and computationally passing in the checkpointed driver
`implementation/run_p1_experiments.py`.

| Item | Roadmap criterion | Driver target |
|---|---|---|
| E9 | EM convergence: chi2(r) monotone; random vs SVD init curves; <=50 iterations | `implementation/run_p1_experiments.py`, checkpoints `results/checkpoints/p1/e9_*.json` |
| E8 | Time-shift OF: t0 RMSE < 2 samples; fixed OF bias worse than shifted OF | same driver, checkpoint `e8.json` |
| G3 | Multichannel joint OF vs EMPCA; joint sigma_E < per-channel sum; corr ~ 1 | same driver, checkpoint `g3.json` |
| G4 | Covariance/PSD-estimation robustness: sweep N_noise; sigma_E approaches oracle | same driver, checkpoint `g4.json` |
| G5 = E10 + E11 | Non-stationary and artifact stress tests; degradation and mitigation curves | same driver, checkpoints `e10.json`, `e11.json` |

Current computational verification results:
- E9 convergence: monotone for all k/init; random init reaches tolerance in
  3, 5, and 8 iterations for k=1,2,3; random/SVD final objective gaps are
  <= 2.8e-16 relative to initial chi2.
- E8 time-shift OF: t0 RMSE = 0.29 samples; shifted amplitude bias = +0.02%;
  fixed-template OF bias = -5.37%.
- G3 multichannel: full-covariance joint OF improves sigma by 35.54% relative
  to diagonal/naive combination; corr(joint OF, rank-1 whitened EMPCA) =
  0.999692.
- G4 finite-PSD robustness: sigma degradation falls from 1.5-1.7% at
  N_noise=20-50 to -0.31% at N_noise=1000.
- E10 non-stationary noise: per-segment PSD improves amplitude RMSE by 7.83%
  over a single global PSD.
- E11 artifact robustness: residual flagging improves amplitude RMSE by
  26.77% while retaining 80% of traces.

P1 outputs:
- Checkpoints: `results/checkpoints/p1/`
- Tables: `results/tables/p1_*.csv`
- Figures: `results/figures/p1_*.png`

## Scientific audit and plotting standards (updated 2026-06-13)

The checkpointed drivers are useful infrastructure, but **implemented/pass is
not the same as publication-grade proof**. From this point forward every
experiment in this roadmap has two statuses:

- **Computational status**: whether the current driver runs and passes its
  internal thresholds.
- **Scientific status**: whether the design and plot are strong enough to
  support the paper/presentation claim without overstatement.

### Global plotting criteria

Every paper-facing quantitative plot must satisfy these criteria unless it is
explicitly labeled as a schematic or a single-event illustration:

1. **Uncertainty is mandatory.** Show seed-level points plus mean and
   68%/95% confidence intervals, or show mean ± standard error / bootstrap
   interval. A bare bar plot is not acceptable for a claim of improvement.
2. **State the sample size.** Captions must include number of seeds, events,
   train/test split, and whether the plot is simulation or real data.
3. **Use held-out evaluation.** Training traces may fit subspaces or PSDs, but
   all reported residuals, amplitudes, sigma values, and KS tests must be on
   held-out traces or independent noise draws.
4. **Show baselines and null expectations.** Include OF / isotropic / oracle /
   null lines where relevant, with the same units and convention.
5. **Show paired comparisons when possible.** If the same seed/test set is
   used for two methods, plot paired seed-level deltas rather than unrelated
   aggregate bars.
6. **Do not hide failed diagnostics.** If sigma improves but residual
   whiteness fails, or chi2 improves but amplitude bias worsens, show both.
7. **Avoid post-hoc thresholds.** Threshold lines must be justified by theory,
   pre-specified tolerances, or an operational requirement. Otherwise report
   the effect size with uncertainty and avoid "pass/fail" language.
8. **Use interpretable axes.** Prefer relative error, variance ratio,
   sigma-ratio, chi2-ratio, angle in degrees, or `1 - cosine` on a log scale
   over visually compressed bars near 1.
9. **Separate theorem checks from performance claims.** Algebraic equivalence
   plots verify implementation and conventions; they do not prove detector
   performance.
10. **Archive source tables.** Each figure must have a CSV/JSON source with
    one row per seed/split/condition, not only aggregated values.

### Figure audit

Older figures `results/figures/block_02_*.png` through
`results/figures/block_09_*.png` should be treated as **legacy exploratory
figures** unless regenerated by the current P0/P1/Block 11/12 drivers. They
may be useful for intuition, but they are not the source of truth for the
paper.

| Figure family | Scientific status | Required action before paper/presentation use |
|---|---|---|
| `block_02_psd_audit.png` | Legacy PSD diagnostic; not a verification plot | Replace with real PSD figure from current checkpoints and include PSD-estimation uncertainty / segment variability. |
| `block_03_real_amplitude_histogram.png` | Legacy descriptive real-data histogram | Use only with calibrated amplitude definition, fit residuals, bootstrap sigma CI, and line-shape caveat. |
| `block_03_real_rank1_scatter.png` | Legacy rank-1 scatter; visually useful but not a proof | Add correlation CI, held-out split, and method comparison against OF/isotropic baseline. |
| `block_04_e5_resolution_scaling.png` | Legacy scaling plot | Replace with Block 11 E5 plus replicate uncertainty and slope CI. |
| `block_05_bridge_cosines.png` | Legacy bridge plot | Replace with P0 E2, plotted as angle/error with seed-level uncertainty. |
| `block_06_convergence.png` | Legacy convergence plot | Replace with P1 E9 plus repeated dataset/init uncertainty. |
| `block_06_rank_saturation.png` | Legacy rank-saturation plot | Do not use to claim optimal rank; replace with G8 model-order selection when available. |
| `block_07_e6_ablation.png` | Legacy metric-ablation bars | Replace with P0 E6 paired seed deltas and CI. |
| `block_07_e7_mismatch_curve.png` | Legacy mismatch curve | Replace with P0 E7 formula/bias plot with seed-level CI. |
| `block_08_pc_overlap_matrix.png` | Legacy qualitative mode-overlap diagnostic | Use only as interpretation aid; not evidence of performance improvement. |
| `block_09_e10_nonstationary.png` | Legacy stress-test bar | Replace with P1 E10 after adding parameter sweeps and uncertainty. |
| `block_09_e11_artifacts.png` | Legacy artifact bar | Replace with P1 E11 after adding retention/ROC curves and uncertainty. |
| `p0_e1_theorem1.png` | Sound theorem-convention check; plot has error bars but is visually compressed near 1 | Add seed-level points and plot `1-rho_w`, `1-corr`, and relative amplitude error on log/linear axes. Keep as theorem check only. |
| `p0_e2_bridge.png` | Sound bridge check; current bars near 1 hide scale | Plot principal angles in degrees or `1-cos(theta)` with seed points and CI. |
| `p0_e3_chi2_monotone.png` | Useful solver/nested-rank diagnostic, but not a proof of true physical rank | Normalize per seed before aggregating, show paired seed curves or CI bands, and avoid claiming rank selection. |
| `p0_e6_empca_vs_iso.png` | Fair controlled metric comparison; supports modest colored-noise effect | Keep only with seed-level paired deltas and CI. Do not sell as a large universal reversal. |
| `p0_e7_mismatch_bias.png` | Strong for fixed-template OF mismatch bias; weak for generic rank-k debiasing | Keep formula/scatter panel; show per-seed bias points with CI; state that k=2 is estimator-convention dependent. |
| `p0_g6_real_rankk.png` | Real-data diagnostic, not full real-data proof | Add bootstrap/split uncertainty; report intrinsic broadening and avoid claiming rank-k resolution gain unless CI supports it. |
| `block_11_e4_crb_units.png` | Strong unit/convention audit, but current variance bars lack CI | Add chi-square confidence intervals for empirical variances and show relative error panel. |
| `block_11_e5_sigma_scaling.png` | Good theoretical scaling check, but one curve without replicate uncertainty | Repeat per noise power across seeds; report fitted slope with CI. |
| `block_11_fig7_sigmaE_vs_rank.png` | Shows rank-k behavior but does not prove monotonic improvement | Keep only as diagnostic; show seed points and CI. |
| `block_11_fig15_amp_bias.png` | Partially supports mismatch-bias story; does not prove rank-2 debiasing | Do not use as proof that more PCs help. Use as nuanced backup only. |
| `block_11_fig16_ks_pvalues.png` / `block_11_fig16_resid_ratio*.png` | Shows residual-whiteness failure, not success | Use only to state that the old whitening claim failed. |
| `block_12_g1_metric_reversal.png` | Real-data comparison is currently weak: raw MSE scale/gauge is suspect and sigma changes are tiny | Requires calibrated reconstruction metric, bootstrap CIs, repeated splits, and a cleaner real PSD/held-out likelihood analysis before strong claims. |
| `block_12_e12_amp_histogram.png` | Useful descriptive real-data plot | Add fit residuals, bootstrap sigma CI, and line-shape caveat before using as verification. |
| `p1_e8_time_shift_of.png` | Good single-scenario demo, not proof | Repeat over seeds/noise colors/jitter ranges; add RMSE/bias CI. |
| `p1_e9_convergence.png` | Good implementation diagnostic | Repeat over datasets/inits or label as one-dataset convergence trace. |
| `p1_g3_multichannel.png` | Useful multichannel demo, but currently one seed and simple bar plot | Repeat over seeds and correlation strengths; show sigma-ratio with CI and paired seed points. |
| `p1_g4_covariance_robustness.png` | Good idea, weak current proof because each `N_noise` has one calibration draw | Repeat PSD estimation many times per `N_noise`; plot median and CI band. |
| `p1_e10_nonstationary.png` | Weak current proof: one scenario, small improvement, no uncertainty | Repeat across drift/segment strengths and seeds; plot RMSE ratio with CI, not simple bars. |
| `p1_e11_artifacts.png` | Useful operational stress test, but selection bias must be explicit | Plot ROC/retention-vs-RMSE curves with seeds; report false rejection on clean traces. |

### Experiment audit

| Experiment | Current computational status | Scientific status | Minimum publication-grade criteria |
|---|---|---|---|
| E1 / A: OF ≡ rank-1 EMPCA | Done/pass | **Sound for theorem verification.** It checks convention, gauge, amplitudes, and residuals across seeds/noise types. It does not prove performance gain. | Seed-level plot with CI; independent held-out residual KS; explicit PSD convention. |
| E2 / B: Bridge theorem | Done/pass | **Sound for equivalence.** This is a numerical theorem check, not an empirical detector claim. | Plot angles or subspace error with CI; include test-set weighted residual difference. |
| E3: chi2(k) monotone | Done/pass | **Moderate.** Monotonic chi2 is expected for nested subspaces, so it is mainly a solver/numerical sanity check. The varied-family drop is useful but should not be sold as rank selection. | Per-seed normalized chi2 curves, CI bands, held-out evaluation, and a separate model-order criterion for choosing k. |
| E4 / D: CRB variance | Repaired in Block 11 | **Strong after unit repair**, but the plot needs empirical variance confidence intervals. | At least three noise colors, thousands of independent events, chi-square CI for variance, and explicit matched/mismatched unit-convention comparison. |
| E5 / D: sigma scaling | Repaired in Block 11 | **Moderate-to-strong theory check.** Current one-run-per-power curve is too thin for a final paper figure. | Replicate seeds per power, log-log slope with CI, relative error panel, and fixed convention. |
| E6 / C: weighted vs isotropic metric | Done/pass | **Sound but modest.** It supports "metric matters under colored noise," not a dramatic universal reversal. Thresholds should not look post-hoc. | Paired seed-level delta chi2 with CI; white-noise negative/near-zero control; colored-noise effect sizes with uncertainty. |
| E7: template-mismatch bias | Done/pass | **Strong for OF bias formula; weak for rank-k debiasing.** k=2 behavior depends on the amplitude estimator convention. | Bias distribution and per-seed CI; formula scatter; avoid claim that rank 2 solves bias. |
| E8: time-shift OF | Done/pass | **Good demonstration, weak proof.** Single seed/single pink-noise scenario. | Repeat across seeds, noise colors, amplitudes, and jitter ranges; plot t0 RMSE and amplitude bias with CI. |
| E9: EM convergence | Done/pass | **Good implementation diagnostic.** One dataset is not enough for a broad convergence claim. | Multiple datasets/seeds/inits; plot median objective trace with CI; report non-monotone failures if any. |
| E10: non-stationary noise | Done/pass | **Weak proof.** One constructed scenario and small improvement. | Sweep nonstationarity strength; replicate seeds; plot global-vs-segment RMSE ratio with CI. |
| E11: artifact robustness | Done/pass | **Useful but selection-biased if stated as reconstruction improvement.** It evaluates the kept subset after rejecting high-residual traces. | ROC/retention curve, false-positive rate on clean traces, RMSE on kept and all traces, seed-level CI. |
| E12: real K-alpha full verification | Partially done | **Incomplete.** Real-data row and amplitude/sigma verification are missing or not publication-grade. | Real PSD figure, repeated train/test splits or bootstrap, calibrated amplitude histogram, sigma CI, and real held-out likelihood metric. |
| G1: real-data metric reversal | Partially done in Block 12 | **Currently weak.** Weighted residual advantage is small; raw MSE comparison appears gauge/scale dominated; sigma is broadening dominated. | Rebuild with calibrated reconstruction metric, bootstrap CIs, repeated splits, and clear separation of weighted residual from energy-resolution claims. |
| G2: repair D/E | Partially done | **D repaired; E mostly refuted.** CRB/unit repair holds. Old rank-k whitening/debiasing claim does not hold. | Keep D/E repair as audit; rewrite Exp E as failure/boundary diagnostic unless new design supports it. |
| G3: multichannel | Done/pass | **Promising but currently demonstration-level.** One seed and one correlation setting. | Sweep channel count/correlation/gain patterns; repeat seeds; plot sigma ratio and OF/EMPCA correlation with CI. |
| G4: covariance robustness | Done/pass | **Promising but weak current proof.** Needs repeated PSD-estimation draws. | For each `N_noise`, run many independent calibration draws; plot median degradation with 68/95% intervals. |
| G5: assumption violations | Done/pass via E10/E11 | **Conceptually valuable, current evidence weak.** | Parameter sweeps for drift/artifact rate/heavy tails; CI; report both degradation and mitigation. |
| G6: real-data rank-k resolution | Partially done | **Weak/incomplete.** Current result mostly says sigma is flat and broadening dominated. | Bootstrap/split CI; show weighted residual, sigma, bias, and model-order diagnostics. Do not claim rank-k gain unless significant. |
| G7: low-SNR containment failure | Not run | **Useful for presentation and threshold motivation.** | SNR sweep with bias/variance/coverage vs CRB; CI over seeds; identify failure threshold. |
| G8: model-order selection | Not run | **High value.** Needed before any "more PCs" claim. | Cross-validated weighted chi2, residual whiteness, amplitude bias, and sigma vs k with one selected rule and held-out validation. |
| G9: external replication | Not run | **Optional but strong for generality.** | Use a public dataset with documented preprocessing; repeat the E1/E6/G4-style checks with uncertainty. |

## Part 1 — Experiments documented in the paper: do they hold water?

| ID | Experiment (paper §) | Supports / proves | Dataset used | Status & soundness |
|---|---|---|---|---|
| A | OF ≡ rank-1 EMPCA equivalence (§7.2, Table 2) | Thm: OF is the rank-1 weighted-ML subspace estimator | Sim (template + analytic PSDs) + real K-alpha | **Scientifically sound as a theorem/convention check.** Do not present as a performance-improvement result. Reconcile any old draft number mismatch and state gauge/preprocessing conventions. |
| B | Bridge Theorem: tied linear AE ↔ EMPCA (§7.3, Table 4) | Weighted-loss linear AE recovers the EMPCA subspace | Sim | **Scientifically sound as an equivalence check.** Plot angle/error with uncertainty instead of compressed bars near 1. |
| C | Metric reversal: isotropic PCA vs weighted (§7.4) | Raw-MSE winner can lose on weighted residual under colored noise | Sim (pink/Brownian) + partial real K-alpha | **Simulated fair control is sound but modest. Real-data version is currently weak.** Use cautious wording and uncertainty; do not claim large universal reversal. |
| D | CRB attainment / σ_E scaling (§7.5, Fig 7) | OF amplitude variance follows CRB under matched PSD convention | Sim | **Original draft was broken; Block 11 repaired the unit convention.** Current result is credible after adding empirical variance CI and slope CI. |
| E | Rank-k residual whitening & bias reduction (§7.6, Figs 15–16) | Higher rank may absorb pulse-shape variation | Sim + K-alpha | **Original claim does not hold.** Current reruns mostly refute universal rank-2 debiasing/whitening. Reframe as a boundary/diagnostic study unless G8/G6 produce significant held-out evidence. |
| F.x | Appendix F ablations (whitening, gauge, weighting) | Robustness of equivalences to convention choices | Sim | Sound; keep. |
| G.x | Noise-budget Studies A/B (App F.8/G) | Framework applies to measured component spectra | `noise_sample/Al2O3_Al_athermal/` | Sound; real measured ASDs, small data committed. |

**Verdict:** the theory chain (A→B) holds as a mathematical/numerical
equivalence story. C holds in controlled simulation but is modest and needs
careful uncertainty. D is repaired as a convention/CRB audit. E should no
longer be a headline proof; current evidence makes it a boundary-condition
section.

## Part 2 — Gap experiments needed (priority-ordered)

| ID | Pri | Experiment | Supports / proves | Best dataset option |
|---|---|---|---|---|
| G1 | P0 | Real-data isotropic vs weighted comparison (metric reversal on real noise) | The paper's headline practical claim on **real** colored noise, not just sim | K-alpha traces (`data/k_alpha/`, PSD from `src/PSDCalculator.py`) — already in hand |
| G2 | P0 | Repair & rerun Experiments D and E | Removes the figure/text contradictions blocking circulation | Same sims as current D/E (QP simulator + analytic PSDs) |
| G3 | P1 | Multichannel joint-OF vs multichannel EMPCA | Generalization of the equivalence beyond 1 channel; sets up Paper 2 | **QP simulator + `MultiChannelNoiseGenerator`** (open, ours) — avoids any dependence on closed TraceSimulator |
| G4 | P1 | Covariance-estimation robustness (finite noise traces, shrinkage) | Practicality: Σ is estimated, not known; quantify σ_E degradation vs N_noise | K-alpha noise traces + sim sweep |
| G5 | P1 | Assumption-violation stress test (non-stationary, non-Gaussian) | Honest failure-mode boundary of the Gaussian ML framework | QP simulator + `TemporalNoiseWrapper` / `ArtifactInjector` — unique capability of this repo |
| G6 | P1 | Real-data rank-k resolution gain | Rank-k claim (Exp E) verified on real pulse-shape variation | K-alpha traces (4358 × 32768) |
| G7 | P2 | Low-SNR / containment failure curve | Where rank-1 ML breaks; guides users | QP simulator amplitude sweep |
| G8 | P2 | Model-order selection (choosing k) | Makes rank-k usable in practice (cross-validated χ², KS whiteness) | K-alpha + sim |
| G9 | P3 | External-dataset replication | Independence from one detector; reviewer-proofing | Open data: GWOSC/LIGO strain segments, NIST TES pulse data, CRESST/SuperCDMS public releases — needs collaborator or download effort |

## Part 3 — Details per experiment

### G1 — Real-data metric reversal (P0, blocks the paper's main practical claim)

**Significance.** §7.4 is the paper's selling point: optimizing raw MSE is
the wrong objective under colored noise. The draft proves this only in
simulation and explicitly defers the real-data version. A reviewer will ask
for it immediately; you already have everything needed.

**How it works.** Estimate Σ (PSD) from K-alpha pre-trigger/noise traces.
Fit (i) isotropic PCA and (ii) Σ-weighted EMPCA / whitened PCA, rank 1–k, on
the same training pulses. Evaluate on held-out pulses: raw MSE, weighted
(Mahalanobis) residual, and energy resolution σ_E at the calibration line
(the line energy is the ground truth).

**Expected output.** A held-out, calibrated comparison with bootstrap CIs:
weighted residual, sigma_E, amplitude bias, and subspace angle. Raw MSE may
be reported only if reconstruction gauges are matched.

**Supports the paper?** Potentially, but the current Block 12 real-data plot
is not strong enough for a headline claim. If reversal does not appear, report
that as a real-detector boundary condition rather than forcing the conclusion.

### G2 — Repair Experiments D and E (P0, blocks circulation)

**Significance.** Three figure/text contradictions (Fig 7 Brownian 184×;
Fig 15 bias numbers; Fig 16 KS p-values) make the current draft internally
inconsistent. Colleagues will find these in minutes.

**How it works.** Rerun the CRB sweep and rank-k study from clean configs;
check the Brownian case for a PSD-normalization or variance-units bug
(184× ≈ suspiciously like a missing 1/N or df factor). For Exp E, either the
claim or the run is wrong: rerun with more traces / fixed seed, and report
the actual KS p and bias-reduction numbers, weakening the text if needed.

**Expected output.** Regenerated CRB/unit-convention figures with variance
CIs; rank-k figures explicitly showing bias, sigma, and residual-whiteness
failures where they occur; remove placeholder Fig 21 and internal plot IDs.

**Supports the paper?** Required for credibility. The current outcome repairs
D and refutes the strong form of E; the paper should say that directly.

### G3 — Multichannel equivalence (P1)

**Significance.** Real detectors (DELight) are multichannel; the paper's
framework is written for general Σ, but every experiment is single-channel.
One multichannel demo broadens the claimed scope and is the natural bridge to
Paper 2.

**How it works.** Generate M-channel tracesets with `QPSimulator` +
`MultiChannelNoiseGenerator` (shared+private or low-rank-correlated modes).
Stack channels into one vector, build block Σ including cross-channel
covariance, run joint OF vs multichannel EMPCA, compare to per-channel OF +
naive sum.

**Expected output.** Sigma-ratio curves with seed-level CIs over channel
correlation strength and channel count; equivalence r with uncertainty between
joint OF and rank-1 multichannel EMPCA.

**Supports the paper?** Yes after replication. The current single-scenario
result is a useful demo and a good presentation bridge to DELight, but not a
standalone proof.

### G4 — Covariance-estimation robustness (P1)

**Significance.** The theory assumes known Σ. In practice Σ̂ comes from a
finite set of noise traces; this is the #1 "does it work in real life"
question.

**How it works.** Sweep number of noise traces N used to estimate the PSD
(e.g., 50 → 5000); optionally compare diagonal-PSD, Welch-averaged, and
shrinkage estimators. Measure σ_E degradation of OF/EMPCA vs the known-Σ
oracle.

**Expected output.** Median sigma degradation vs `N_noise` with 68/95% bands
from many independent PSD-estimation draws; optional shrinkage comparison.

**Supports the paper?** Yes after repeated calibration draws. The current
single-draw curve is only a proof of concept.

### G5 — Assumption-violation stress test (P1)

**Significance.** The framework is Gaussian-stationary ML. Your enhanced
noise module exists precisely to violate each assumption in a controlled
way — no other group can do this ablation as cleanly. It also pre-justifies
Paper 2 (nonlinear models) by showing where the linear theory breaks.

**How it works.** Fix signal + base PSD; switch on, one at a time:
non-stationarity (`TemporalNoiseWrapper`: drift, piecewise segments),
non-Gaussianity (`ArtifactInjector`: glitches, heavy-tailed impulses),
channel correlation mis-modeling. Measure σ_E, bias, and residual whiteness
of OF/EMPCA as the violation strength increases.

**Expected output.** Degradation curves per violation type with seed-level
CIs; mitigation curves that report both RMSE and retained fraction / false
rejection rate.

**Supports the paper?** Yes after sweeps. The current E10/E11 runs are
single-scenario demos and should not be presented as rigorous stress-test
verification.

### G6 — Real-data rank-k resolution gain (P1)

**Significance.** Exp E's rank-k claim is currently sim-only and its sim
version is broken (see G2). The K-alpha set has natural pulse-shape variation
and timing jitter — the exact effect rank-k is supposed to absorb.

**How it works.** Train EMPCA rank 1…5 on K-alpha traces; evaluate σ_E at the
calibration line and residual whiteness (KS) per rank on held-out data.

**Expected output.** Bootstrap/split CI for sigma_E(k), weighted residual,
bias, and whiteness diagnostics. If no improvement appears, that is the
result.

**Supports the paper?** Only if the confidence intervals show a significant
held-out effect. Current evidence is incomplete and broadening dominated.

### G7 — Low-SNR / containment failure (P2)

**How it works.** QP-simulator amplitude sweep down to SNR ≪ 1; measure bias
and variance of Â vs CRB.
**Expected output.** Threshold-SNR plot showing where ML estimates degrade.
**Supports.** Scope statement; minor but cheap.

### G8 — Model-order selection (P2)

**How it works.** Cross-validated weighted χ² and residual-whiteness KS vs k;
compare to scree/eigenvalue criteria under colored noise.
**Expected output.** A recommended, automated rule for choosing k.
**Supports.** Practicality of the rank-k method; reviewers like it.

### G9 — External open-data replication (P3)

**Options.**
- **GWOSC (LIGO/Virgo)** strain data: colored, non-stationary noise with
  published PSD methods — good for the weighting/whitening story, not for
  σ_E.
- **NIST / TES microcalorimeter** public pulse datasets (e.g., via the MASS
  software ecosystem) — closest analogue to K-alpha data.
- **CRESST / SuperCDMS** public releases — cryogenic detector pulses.

**Significance.** Shows the method is detector-independent; useful for MLST
referees outside the microcalorimeter niche. Requires search/collaborator
outreach; not needed for arXiv v1.

## Part 4 — Recommended order of execution

1. **Regenerate the paper-facing plots with the global plotting criteria**:
   seed points, CIs/error bars, held-out metrics, and source CSV rows.
2. **G2/D repair stays central; G2/E becomes a failure/boundary result**:
   keep the CRB/unit repair, but stop claiming universal rank-k debiasing.
3. **G8 model-order selection** before any "more PCs" claim. This is now the
   highest-value unfinished experiment for both paper and presentation.
4. **G1/E12 real-data cleanup** with bootstrap CIs and calibrated gauges.
5. **G3 + G4 replication**: multichannel and covariance estimation are
   presentation-strong once repeated over seeds/conditions.
6. **G7 low-SNR failure curve** for threshold motivation.
7. **G9** only if external generality becomes a reviewer requirement.

## Part 5 — Cross-reference with `experiment_checklist.md` (E1–E12)

The checklist maps each theorem to a runnable experiment with pass
thresholds; this plan adds the soundness verdicts and gaps. Mapping and
current status. The `Status` column below is computational status only; use
the Scientific audit section above for publication-grade strength.

| Checklist | This plan | Status | Notes |
|---|---|---|---|
| E1 (Thm 1: OF ≡ rank-1 EMPCA) | A / P0 | Done/pass | `run_p0_blockers.py`; rho_w, amplitude correlation, median relative error, and KS checks pass across white/pink/Brownian seeds |
| E2 (Bridge Theorem) | B / P0 | Done/pass | `run_p0_blockers.py`; k=1..3 principal-angle cosines and residual differences pass |
| E3 (χ²(k) monotone) | Exp E / G2, G8 / P0 | Done/pass | `run_p0_blockers.py`; fixed-family plateau and varied-family rank drop both pass |
| E4 (CRB: Var(Â)=1/N_Φ) | D / G2 | Repaired in Block 11 | `run_block11_12.py`; PSD convention bug documented and consistent DFT/physical units used |
| E5 (σ_E = E₀/√N_Φ scaling + real) | D / G2 + G1-adjacent | Repaired in Block 11 | `run_block11_12.py`; sigma scaling table/figure emitted under `results/tables` and `results/figures` |
| E6 (EMPCA vs isotropic PCA) | C / P0 | Done/pass | `run_p0_blockers.py`; fair exact rFFT subspace control passes with revised thresholds |
| E7 (template-mismatch bias) | Exp E / G2 / P0 | Done/pass | `run_p0_blockers.py`; OF bias >10%, EMPCA k=1 partially debiases, per-event OF amplitude follows weighted cosine |
| E8 (time-shift OF) | P1 roadmap | Done/pass | `run_p1_experiments.py`; t0 RMSE = 0.29 samples; shifted bias +0.02% vs fixed-template bias -5.37% |
| E9 (EM convergence) | P1 roadmap | Done/pass | `run_p1_experiments.py`; monotone for k=1..3 and random/SVD init; random reaches tolerance in <=8 iterations |
| E10 (non-stationary noise) | G5 | Done/pass | `run_p1_experiments.py`; segment PSD improves RMSE by 7.83% over global PSD |
| E11 (artifact robustness) | G5 | Done/pass | `run_p1_experiments.py`; residual flagging improves RMSE by 26.77% with 80% trace retention |
| E12 (real K-alpha full verification) | G1 + G6 | Partially done (ρ_w row exists; amplitude histogram + σ_A row missing) | Replaces placeholder Fig 21 |

Items in this plan with no checklist counterpart:
G3 (multichannel) and G4 (covariance estimation) are now implemented in
`run_p1_experiments.py`; G7 (low-SNR), G8 (model-order selection), and G9
(external datasets) remain optional / later roadmap items.

Cleanup note: the paper currently leaks checklist IDs (E1, E6, …) as
internal plot labels — strip them from figures before circulation.
