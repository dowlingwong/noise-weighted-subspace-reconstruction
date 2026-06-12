# Experiment Plan for the Unified ML Framework Paper

Status assessment of the experiments documented in the draft
(`arXiv_v1_polished_extended_preprint.pdf`), plus the additional experiments
needed before circulation / submission. Datasets refer to the inventory in
[README.md → Datasets](README.md#datasets).

This plan is the **strategic layer** (soundness verdicts, gaps, dataset
choices). The **operational layer** — per-experiment inputs, pass thresholds,
and paper LaTeX labels — is `experiment_checklist.md` (E1–E12; verified
current on 2026-06-12: the QPSimulator API it assumes is implemented).
Part 5 below maps the two.

## Part 1 — Experiments documented in the paper: do they hold water?

| ID | Experiment (paper §) | Supports / proves | Dataset used | Status & soundness |
|---|---|---|---|---|
| A | OF ≡ rank-1 EMPCA equivalence (§7.2, Table 2) | Thm: OF is the rank-1 weighted-ML subspace estimator | Sim (template + analytic PSDs) + real K-alpha | **Mostly sound.** r = 0.9942 in Table 2 vs "mean r = 1.000000" in Fig 2 text — reconcile numbers and state matching conventions (gauge, preprocessing) explicitly. |
| B | Bridge Theorem: tied linear AE ↔ EMPCA (§7.3, Table 4) | Weighted-loss linear AE recovers the EMPCA subspace | Sim | **Sound**, but Table 4 (Δχ² = −0.23%) and Table 7 (−0.03%) disagree; unify. |
| C | Metric reversal: isotropic PCA vs weighted (§7.4) | Raw-MSE winner loses on weighted residual / σ_E under colored noise | Sim (pink/Brownian) | **Sound and central.** But the real-data version is explicitly deferred — see G1. |
| D | CRB attainment / σ_E scaling (§7.5, Fig 7) | OF and rank-1 EMPCA attain the CRB | Sim | **Contradiction.** Fig 7 Brownian shows σ_E ≈ 184× OF baseline while text claims "matches within 2%". Rerun or re-plot; likely a normalization/units bug. |
| E | Rank-k residual whitening & bias reduction (§7.6, Figs 15–16) | Higher rank absorbs pulse-shape variation; residuals → white | Sim + K-alpha | **Contradictions.** (i) Fig 16: KS p = 0.033 (pink, k=2), ≈ 0 (white/Brownian) vs claim "KS p > 0.05 at k=2". (ii) Fig 15: rank-2 bias (−0.352/−0.818) is not the claimed ">30% reduction" vs rank-1 (−0.347/−0.797). Must rerun before circulation. |
| F.x | Appendix F ablations (whitening, gauge, weighting) | Robustness of equivalences to convention choices | Sim | Sound; keep. |
| G.x | Noise-budget Studies A/B (App F.8/G) | Framework applies to measured component spectra | `noise_sample/Al2O3_Al_athermal/` | Sound; real measured ASDs, small data committed. |

**Verdict:** the theory chain (A→B→C) holds; experiments D and E as plotted
contradict their own text and must be rerun/repaired. Fig 21 is still a
placeholder ("Simulation PSD (real K-alpha data absent)").

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

**Expected output.** A two-panel figure: isotropic wins or ties raw MSE,
weighted wins σ_E and weighted residual; a table of σ_E (eV) per method.
Subspace angle θw between isotropic and weighted bases on real noise.

**Supports the paper?** Yes — directly converts the central simulated claim
into a real-detector result. If reversal does *not* appear (K-alpha noise too
white), that is still publishable as a boundary condition and motivates the
colored-noise sims.

### G2 — Repair Experiments D and E (P0, blocks circulation)

**Significance.** Three figure/text contradictions (Fig 7 Brownian 184×;
Fig 15 bias numbers; Fig 16 KS p-values) make the current draft internally
inconsistent. Colleagues will find these in minutes.

**How it works.** Rerun the CRB sweep and rank-k study from clean configs;
check the Brownian case for a PSD-normalization or variance-units bug
(184× ≈ suspiciously like a missing 1/N or df factor). For Exp E, either the
claim or the run is wrong: rerun with more traces / fixed seed, and report
the actual KS p and bias-reduction numbers, weakening the text if needed.

**Expected output.** Regenerated Figs 7, 15, 16 consistent with text; updated
Tables; remove placeholder Fig 21 (replace with real K-alpha PSD — data is in
hand) and internal plot IDs (E1, E6, P2–P6).

**Supports the paper?** Required for credibility; outcome determines whether
the rank-k bias-reduction claim survives as stated or gets softened.

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

**Expected output.** σ_E (joint, full Σ) < σ_E (per-channel + sum) when
channels are correlated; equivalence r ≈ 1 between joint OF and rank-1
multichannel EMPCA.

**Supports the paper?** Yes — and crucially it uses only the **open** QP
simulator, so the paper does not depend on the closed TraceSimulator at all.
(TraceSimulator remains a nice-to-have for position-dependent realism in
Paper 2.)

### G4 — Covariance-estimation robustness (P1)

**Significance.** The theory assumes known Σ. In practice Σ̂ comes from a
finite set of noise traces; this is the #1 "does it work in real life"
question.

**How it works.** Sweep number of noise traces N used to estimate the PSD
(e.g., 50 → 5000); optionally compare diagonal-PSD, Welch-averaged, and
shrinkage estimators. Measure σ_E degradation of OF/EMPCA vs the known-Σ
oracle.

**Expected output.** σ_E vs N curve approaching the oracle; a practical rule
of thumb (e.g., N ≳ 10× trace length / segments for <5% degradation).

**Supports the paper?** Strengthens it as a methods paper; cheap to run.

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

**Expected output.** Degradation curves per violation type; identification of
which violation hurts most (likely impulsive artifacts) and how much robust
preprocessing (trace cuts) recovers.

**Supports the paper?** Yes — turns implicit assumptions into measured
boundaries, and gives Paper 2 its motivation section for free.

### G6 — Real-data rank-k resolution gain (P1)

**Significance.** Exp E's rank-k claim is currently sim-only and its sim
version is broken (see G2). The K-alpha set has natural pulse-shape variation
and timing jitter — the exact effect rank-k is supposed to absorb.

**How it works.** Train EMPCA rank 1…5 on K-alpha traces; evaluate σ_E at the
calibration line and residual whiteness (KS) per rank on held-out data.

**Expected output.** σ_E(k) curve; expected modest improvement from k=2
(timing-derivative component), plateau after. If no improvement, report that
honestly — it bounds the claim.

**Supports the paper?** Yes, replaces a broken sim result with a real one.

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

1. **G2** (repair D/E) and **G1** (real-data reversal) — both P0, block
   circulation of a self-consistent draft.
2. **G6** (real rank-k) — pairs with G2's rerun.
3. **G3 + G5** (multichannel + stress test, open QP stack) — biggest added
   value per effort; both Paper-1-strengthening and Paper-2-motivating.
4. **G4, G7, G8** as appendix material.
5. **G9** only if MLST referees ask, or for Paper 2.

## Part 5 — Cross-reference with `experiment_checklist.md` (E1–E12)

The checklist maps each theorem to a runnable experiment with pass
thresholds; this plan adds the soundness verdicts and gaps. Mapping and
current status:

| Checklist | This plan | Status | Notes |
|---|---|---|---|
| E1 (Thm 1: OF ≡ rank-1 EMPCA) | A | Partially done (real-data ρ_w = 0.9999999655 ✓; sim script with all four metrics missing) | `tests/test_rank1_of_empca_equivalence.py` now covers the sim version minimally |
| E2 (Bridge Theorem) | B | Partially done | Reconcile Table 4 vs Table 7 numbers while rerunning |
| E3 (χ²(k) monotone) | Exp E / G2, G8 | Not run systematically | Run with `mode='full'` (fixed 2026-06-12); fast-mode decoupling is a suspect in the Fig 15/16 contradictions |
| E4 (CRB: Var(Â)=1/N_Φ) | D / G2 | **Broken in draft** (Fig 7 Brownian 184×) | Units suspect: `noise_module.build_psd()` returns DFT power, not A²/Hz — see notebook check 6b; compute N_Φ from the *physical* PSD |
| E5 (σ_E = E₀/√N_Φ scaling + real) | D / G2 + G1-adjacent | Not run | Same units caveat as E4 |
| E6 (EMPCA vs isotropic PCA) | C | Sim version sound; **real-data version = G1 (P0)** | The paper's headline ablation |
| E7 (template-mismatch bias cos²θ_w) | Exp E / G2 | Contradicted in draft (Fig 15 bias numbers) | Rerun with full-mode rank-2 |
| E8 (time-shift OF) | — (not in plan; keep) | Not run | Cheap; supports §3 extension; needed anyway for K-alpha jitter handling in G1/G6 |
| E9 (EM convergence) | — (not in plan; keep) | Not run | Add fast-vs-full comparison to the init study — directly tests whether the M-step approximation matters |
| E10 (non-stationary noise) | G5 | Not run | `TemporalNoiseWrapper` |
| E11 (artifact robustness) | G5 | Not run | `ArtifactInjector` |
| E12 (real K-alpha full verification) | G1 + G6 | Partially done (ρ_w row exists; amplitude histogram + σ_A row missing) | Replaces placeholder Fig 21 |

Items in this plan with no checklist counterpart (add to checklist when
scheduled): G3 (multichannel), G4 (covariance estimation), G7 (low-SNR),
G8 (model-order selection), G9 (external datasets).

Cleanup note: the paper currently leaks checklist IDs (E1, E6, …) as
internal plot labels — strip them from figures before circulation.
