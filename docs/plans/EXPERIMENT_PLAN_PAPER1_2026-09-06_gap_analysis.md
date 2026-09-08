# Paper 1 — experiment plan and gap analysis (2026-09-06)

_Paper-level map: which claim in the manuscript is carried by which experiment, whether the
experiment actually carries it, what is missing, and how to build it. The run-level
specification lives in the experiment repository
(`noise-weighted-subspace-reconstruction/docs/EXPERIMENT_PLAN_PAPER1.md`, IDs P1-E1…E10) and is
not duplicated here; this document extends it with P1-E11…E16 and states the standard each
experiment must meet to enter the paper. The agent-facing implementation instruction is
`plan/AGENT_INSTRUCTION_EXPERIMENTS_2026-09-06.md`. Written after the four-pass review
`reviews/2026-09-06-paper-review.md`; section numbers are those of the current build
(§9 = experiments, §7 = loss geometry, §6 = separable class)._

## 1. What the experiment repository contains, and whether it can be used

Everything under `noise-weighted-subspace-reconstruction/` can be used, with three
qualifications.

| Area | Location | Usable for Paper 1? | Note |
|---|---|---|---|
| Core library: OF, weighted PCA/EMPCA, closed-form and gradient-trained tied AE, PSD/whitening, principal angles, CRESST/GWOSC/TIDMAD loaders and analyses | `src/noise_geometry/`, `src/canonical/` | **yes** — complete, tested (`tests/`, 30 files) | this is the code every archived number came from |
| Synthetic S0–S9 runner, sweeps, 30-seed S5 | `scripts/run_experiment.py`, `sweep.py`, `run_s5_confirmatory_sweep.py`; `results/sweeps`, `results/metrics`, `evidence/paper_bundle/` | **yes** — archived, provenance closed (P1-E5) | bundle copy now in the manuscript repo |
| NFPA confirmatory harness: planted generator (channel × time factors, α-nonseparability, Brownian/AR channel noise, SNR normalisation), ALS fits for NFPA / CW-PCA / Iso-MPCA / IsoPCA / OF, metrics, semantic probes, estimated-Σ, corrected angles, calibration sweep | `experiments/synthetic/nfpa_confirmatory/` | **yes** — the workhorse for §6 and §7.4 evidence | R/PS numbers archived under `results/rps2026/`; gated until 29 Sep |
| Paper-1 runners written after the merge (Fig. 1 factors, α-sweep, rank-one check, probes, instrument table, provenance) | `experiments/paper1/`, `results/paper1/P1-E1…E9` | **yes** — E1, E4, E5, E9 complete; E2 and E6 at dev protocol | E6 fails its own temporal-cosine gate at n = 3000 (0.993 < 0.999); rerun at n ≥ 20 000 |
| CRESST public-release pipeline (chronological within-channel split, PSD covariance with shrinkage, five baselines, five negative controls, template bank, trigger-efficiency grid, hierarchical bootstrap) | `src/noise_geometry/cresst/`, `configs/cresst/`, `scripts/run_cresst_postfreeze_sensitivity.py` | **yes** — but needs the release on `/ceph` (test split ≈ 160 MB) | rank is a config key; a rank sweep is a config change |
| TIDMAD stage: G0 contract, Phases 1–6, Wiener estimator, factored-attention rung-4 pilot | `scripts/tidmad_phases.py`, `scripts/run_tidmad_stage.py`, `configs/tidmad/`, `src/tidmad/` | **yes** — needs the `.h5` files (≈ 2.2 GB each) | Phase 6 has a `random_weighted_basis` control that saturates; norm-matched variant is a small code change |
| GWOSC stage (PSD/GWpy reference, shared-FIR identity, global and time-local calibration) | `src/noise_geometry/gwosc/`, `configs/gwosc/` | **yes** — completed negative; keep as boundary | can be extended to hardware-injection windows (P1-E14) |
| Noise simulator: PSD models, temporal nonstationarity wrapper, multichannel generator, artefacts, non-Gaussian tails, calibration | `noise_module/` | **yes** — the source of controlled nonstationary Σ for P1-E8a/E11 | internal-detector component *labels* (SQUID/Johnson/TD/Er) must not enter the paper (CHANGELOG 07-27) |
| QP trace simulator | `QP_simulator/` | **no** for the paper — internal-detector signal-formation model | usable for private sanity checks only |
| Paper-2 scaffold (AE/transformer trainers, K-α data) | `paper2/` | **no** — K-α data are internal and undocumented; trainers have known defects (`docs/EVALUATION.md`) | do not cite |
| ORACLE / Paper-3 packages (tangent space, families, monitors) | `experiments/oracle/` | **partly** — `oracle_cov/tangent.py` already computes T_S, T_S^⊥ and I(z) on the planted generator; reuse for P1-E11 | results belong to the companion paper |

Qualifications: (i) nothing derived from the internal detector (QP_simulator, `noise_module`
component labels, K-α) may be printed; (ii) `results/rps2026/*` may be cited only after the R/PS
decision (29 Sep, double-blind); (iii) any number entering the paper must sit in
`transfer_paper/` with a manifest checksum and an evidence-map row — `results/paper1/*` is not
citable until copied and registered.

## 2. Experiments in the build today, by claim — and whether they carry the claim

| Paper claim (section) | Experiment | Status | Carries the claim? | Gap |
|---|---|---|---|---|
| OF attains the CRB; σ_E formula (§3.2, App. A) | S1 (10 seeds); TIDMAD Phase 1 gates 1–2 on real SQUID noise | archived | yes | none |
| OF bias under template mismatch, cos²θ_w law (§3.5) | TIDMAD Phase 1 gate 5 | archived | yes (real noise) | no synthetic curve vs angle; optional |
| Stationary basis = precision eigenbasis; token-axis rule (§3.4) | none | — | **no** (stated as a design rule) | P1-E8a/b planned, not run |
| Rank-1 EMPCA → OF (Thm 4.1/4.2) | S2; App. D | archived | yes | S2 reports 3.9° at finite N; fine |
| k > 1 helps; containment; timing → rank (§4.3, B.1) | S6 | archived | yes for the synthetic model | no real-data rank sweep |
| Davis–Kahan finite-sample stability; Σ̂ error (§4.4, Rem. 5.3) | S7 (calibration size, synthetic) | archived | partly | no real-data calibration-size curve (P1-E3 is planted-synthetic) |
| Bridge Theorem (Thm 5.1) | S3 (gradient-trained tied AE, gap 2e-16); CRESST tied-AE angle 1.9e-6° | archived | yes as an identity check | **no ablation**: nothing shows an *untied* AE breaks the equivalence, or that an MSE-trained tied AE lands elsewhere on real data |
| Multi-channel / joint OF (§2.5, App. B) | S9 | archived | yes | none |
| NFPA restricted bridge, rank-one corollary, parameter counts, error decomposition (§6) | `results/rps2026/E1,E2,E8,E9,E10b,E11`; P1-E1, E2 (dev), E6 (partial), E9 (4 seeds) | run, **not in build** | — | gated; E2/E6/E9 not at frozen protocol; no optimisation-term measurement; no real multichannel data |
| Architecture-neutral metric (Prop. 7.1) | none directly; S4/S5 + CRESST + TIDMAD are the linear-class instances | — | indirectly | the nonlinear pilot (rung 4, 2 seeds) shows *no* reversal — a boundary, honestly stated in `07b` |
| Pullback metric (Prop. 7.2) | none in Paper 1 | — | — | companion paper; a one-hour synthetic check is possible (P1-E11b) |
| Excited support (Prop. 7.3) | none | — | **no** | P1-E11 (cheap, decisive) |
| Metric reversal, synthetic (P1, P2) | S4 white control; S5 30-seed | archived | yes | none |
| Metric reversal on real data, same rank (§9.3) | CRESST PCA vs EMPCA at k = 6, 7 channels, hierarchical bootstrap | archived | yes, heterogeneous | rank 6 only; no subset reporting (low-SNR tercile); no anisotropy predictor (P1-E10) |
| Signal-model capacity (§9.3) | CRESST OF vs EMPCA vs multi-template OF | archived | yes | none |
| Truth agreement (P3) (§9.4) | TIDMAD Phase 2, file 0000 (+0010, 0005 not significant) | archived | **weakly**: one file; both estimators below the zero predictor (0.0747 < 0.0880 < 0.1209) | needs a zero-predictor row, a norm-matched control, an operating point above the noise floor, a second significant file |
| Covariance model calibrates on real noise (§9.4) | TIDMAD Phase 4; GWOSC global/local failures | archived | yes (contrast) | none |
| Detector-facing benchmark (§9.4) | TIDMAD Phase 5 (negative) | archived | yes (negative) | none — keep |
| Rank-2 prediction on real data | TIDMAD Phase 3 (3 files, not confirmed) | archived | negative, stated | none — keep |
| Negative controls | CRESST 5/5 pooled; TIDMAD 3/5 | archived | yes | TIDMAD `random_weighted_basis` saturated |

**Do the current experiments make sense?** Yes: the four-stage arc (identities in synthetic
data → real-data hierarchy without truth → a calibration boundary → real data with truth) is the
right design, the predeclaration discipline is unusual and good, and the negative results are
reported. Three structural weaknesses remain. First, the evidence is almost entirely
*gates* (pass/fail at one operating point) rather than *curves*; a referee wants to see the
effect as a function of SNR, calibration size, rank and anisotropy, because that is where the
mechanism shows. Second, the truth-referenced leg is one file at an operating point where the
estimator is worse than predicting zero. Third, the claims added on 2–3 September — the
separable class, the architecture-neutral metric, the three determinants — have no evidence in
the build, and the tied-AE claim has no ablation. Everything below is aimed at those three.

## 3. Missing experiments — what, why, how, cost

IDs continue the experiment repository's numbering. "Hook" is the manuscript location the
result feeds. Priority: **A** = needed for MLST submission; **B** = strengthens materially;
**C** = optional.

### A. Close the gaps a referee will name first

**P1-E2 / E6 / E9 at the frozen protocol** (§6, `07b`). E2: 10 seeds × α ∈ {0, 0.05, 0.1, 0.2,
0.4, 0.8}; E6: n ≥ 20 000 so the temporal cosine gate (≥ 0.999) is either met or the gate's
finite-n dependence is reported; E9: 10 seeds. Cost: CPU, ≈ 2 h total. Existing code
(`experiments/paper1/alpha_sweep.py`, `rank_one_check.py`, `latent_probes.py`). Hook: §6.4
table, sentence after Cor. 6.3, `07b` "Which axes carry the physics".

**P1-E11 — Excited support (Prop. 7.3) on the planted generator.** Train NFPA / CW-PCA with the
training distribution varying only S ⊂ {amplitude, rise, decay, secondary} (e.g. hold decay
time fixed), from R different random initialisations. Measure (i) the principal angles between
the learned subspace and T_S (must → 0), (ii) the angles between the learned complement
directions across restarts (must be initialisation-dependent, i.e. large and seed-scattered),
(iii) the same quantities when S = all (angles across restarts → 0). Reuse
`experiments/oracle/oracle_cov/tangent.py` for T_S and `nfpa_confirmatory` for the fits.
Cost: 1 h CPU. This turns the third determinant from a statement into a measured fact inside
Paper 1, which answers the referee question "why is Prop. 7.3 here?". Hook: one paragraph +
one small table after Prop. 7.3; App. E.5.

**P1-E12 — Tied versus untied linear AE (the missing ablation for §5.4 / Rem. E.3).** On the S3
configuration and on the planted generator: train (a) tied AE with Σ⁻¹ loss (exists:
`train_weighted_linear_ae`), (b) untied linear AE (independent encoder E, decoder D) with the
same Σ⁻¹ loss, (c) tied AE with MSE loss, (d) untied with MSE. Report, per seed, the principal
angles to the EMPCA subspace, the obliqueness ‖DE − (DE)ᵀ_Σ‖ of the fitted projection, and the
held-out χ̄². Prediction (pre-registered): (a) → 0°, (b) reaches the same *subspace* at the
optimum but an oblique projector when stopped early or regularised, (c)/(d) → the PCA subspace
(large angle under colored noise). Cost: 2 h CPU (new ≈ 150 lines: untied trainer). Hook:
§5.4 one sentence + App. C/E table. Also run (a) vs (c) on CRESST at k = 6 if the release is on
`/ceph` (adds "reversal at the AE class on real data" to §9.3; cost: one CRESST run).

**P1-E13 — TIDMAD operating-point sweep and zero-predictor baseline (§9.4).** On files 0000,
0005, 0010: window length ∈ {4096, 16384, 65536}, rank 2, everything else frozen; report for
each (file, window) the amplitude RMSE of PCA, EMPCA, and the constant-zero predictor, plus the
paired PCA−EMPCA interval. Add the norm-matched random-basis control (random subspace scaled to
the fitted basis's whitened energy) to Phase 6. Cost: CPU, ≈ 1 h per file per window (I/O
bound). Acceptance is not "the estimator beats zero" — it is that the table is printed with the
SNR floor (smallest window at which EMPCA < zero-predictor) stated. If no window beats the
zero predictor on any file, the abstract's consequence sentence is reworded to ordering only.
Hook: §9.4 Table 4 gains a zero-predictor row and a window column; Limitations gets a number.

**P1-E10 — Anisotropy predictor across CRESST channels and instruments** (already specified in
the experiment repo). Cost: cluster (needs per-channel PSDs on `/ceph`), half a day. Hook: §9
instrument table gets a predictor column; this is the "three regimes, one principle" claim.

**Hyperparameter table** (P1-E4b). Assemble one table with every value actually used per stage
from the archived configs and manifests. Desk work, no runs. Hook: §9.1.

### B. Strengthen

**P1-E14 — Truth-referenced gravitational-wave arm.** GWOSC is currently a pure boundary
because the interval has no recorded truth. Two public routes provide *known injected signal
in real interferometer noise*: (i) O1/O2 transient hardware injections, whose GPS times and
CBC parameters are published by GWOSC (O3 has no CBC hardware injections), so the injected
waveform can be regenerated with LALSimulation/PyCBC and compared to the reconstruction; (ii)
the MLGWSC-1 challenge generator (dataset 4 = real O3a noise + injections with known
parameters, code public on GitHub). Run the same PCA-vs-EMPCA-vs-truth comparison as TIDMAD
Phase 2 on a matched-filter-bank subspace around the injection. Cost: 1–2 days (new loader,
waveform regeneration; the estimator code exists). Hook: §9 gains a second truth-referenced
instrument, which is the single most valuable addition for "three noise regimes" — GW would then
be the *nonstationary* regime with truth. Report the calibration failure alongside; if the
weighted estimator does not win under a misspecified Σ, that is the predicted outcome and a
strong result.

**P1-E15 — HPGe regime: Majorana Demonstrator AI/ML release.** 3.19 M raw germanium waveforms
(3800 samples, pre-trigger baseline, energy label, detector id) from ²²⁸Th calibration, HDF5,
permissive terms (DataPlanet / Zenodo 8257027). This adds a fourth physical system and a
truth-adjacent consequence metric that neither CRESST nor TIDMAD has: the width of the
2614.5 keV line under each estimator is an energy-resolution measurement with no circularity
(the line energy is nature's, not the pipeline's). Design: per detector, PSD from pre-trigger
baselines (chronological split), OF vs PCA vs EMPCA at k ∈ {1, 2, 4, 6}; report weighted
residual, and FWHM at 2614.5 keV per estimator. Cost: 2–3 days (loader + the CRESST pipeline
with a new `release_format`); data ≈ 1.8 GB per chunk, one chunk suffices. Hook: §9 new
subsection; instrument table row.

**CRESST rank sweep and subset reporting.** Rerun `cresst_validation.yaml` at k ∈ {2, 4, 8}
(three config changes) and re-cut the archived event residuals by SNR tercile. Cost: cluster,
hours. Hook: §9.3 one sentence + appendix table; the Kieseler inclusive+subset standard.

**P1-E8a/b — PE uniqueness numerics** (specified in the experiment repo; no learning). Cost:
1 h CPU. Hook: App. E.5 figure. Only after the corollary is written.

**NFPA optimisation term.** For P1-E2 at each α, run ALS from 10 restarts and report the spread
of the held-out χ̄² and of the projector error across restarts (this is the "optimisation term"
of §6.4 that is currently named but never measured). Cost: folded into P1-E2 (restarts already
= 10 in `PrimaryConfig`; only the reporting is missing).

**Real multichannel NFPA.** The separable class has no real-data instance. Check whether the
CRESST release pairs phonon and light channels per event (channels 57/58 of run 35 may be one
module); if so, C = 2 NFPA vs joint EMPCA at matched rank is a one-day experiment. If not,
state in Limitations that the class is evidenced on planted data only.

### C. Optional

TIDMAD Phase 2 on a fourth/fifth validation file chosen by the frozen selection rule (power).
P1-E8c attention ladder (demoted). SIXTE/xifusim TES simulator (public, GitHub `thdauser/sixte`)
as an *independent* microcalorimeter simulator with known photon energies — useful only if a
referee objects that the synthetic generator is the authors' own; not needed now.

## 4. Datasets and simulators

| Resource | What it gives | Truth? | Used in build | Candidate for |
|---|---|---|---|---|
| CRESST-II/III pulse-shape release (arXiv:2508.03078; ORIGINS DMDC) | 7 channels, pulses + random-trigger noise, 512 samples | no injected truth; energies from CRESST pipeline | yes (§9.3) | rank sweep, subset reporting, AE reversal, anisotropy predictor, possibly C = 2 NFPA |
| TIDMAD / ABRACADABRA (github.com/jessicafry/TIDMAD, CC-BY-4.0) | SQUID readout + injected reference channel, 10 MHz | **yes** (injected) | yes (§9.4) | window sweep, zero-predictor, norm-matched control, more files |
| GWOSC GW150914 + O1/O2 hardware-injection lists (gwosc.org/o1_inj, o2_inj) | real strain; injection times and CBC parameters | **yes** via regenerated waveform | interval only (§9.3) | P1-E14 |
| MLGWSC-1 (github.com/gwastro/ml-mock-data-challenge-1) | generator: real O3a noise + injections with known parameters | **yes** | no | P1-E14 alternative |
| Majorana Demonstrator AI/ML release (arXiv:2308.10856; DataPlanet, Zenodo 8257027) | 3.19 M raw HPGe waveforms, baseline, energy labels, ²²⁸Th lines | line energies (physical truth) | no | P1-E15 |
| Own planted generator (`nfpa_confirmatory`) | channel × time factors, α-nonseparable, colored/AR noise, per-event latents | **yes** (planted) | R/PS (gated) | E2/E6/E9/E11/E12 |
| `noise_module` | PSD models, nonstationary envelopes, multichannel Σ | synthetic | S1–S9 | E8a, E11 nonstationary Σ |
| SIXTE / xifusim (github.com/thdauser/sixte) | TES pulse simulator with noise, known photon energies | **yes** | no | optional independent simulator |
| SuperCDMS data releases | reduced spectra/limits as far as the index shows; raw traces not confirmed | — | no | not recommended without checking the ZIP contents |

## 5. Order and budget

1. Desk (this week): hyperparameter table; zero-predictor sentence; Prop. 7.3 restatement.
2. CPU, no new data (≈ 1 day): E2/E6/E9 frozen protocol → E11 → E12 (synthetic part) → E8a/b.
3. Data-bound CPU (≈ 2 days): E13 TIDMAD window sweep + norm-matched control.
4. Cluster (≈ 2 days): CRESST rank sweep + subset cut + E12 on CRESST + E10 predictor.
5. New instrument (choose one, ≈ 3 days): E14 (GW with truth) **or** E15 (HPGe). E14 is the
   better fit for the narrative (turns the boundary into a truth-referenced nonstationary
   regime); E15 gives a physical-units consequence (line width). If time allows, both.
6. After 29 Sep: input `07b`, register all runs in the evidence map, compress for MLST.

Total: roughly two working weeks of experiment time plus the writing.

## 6. Pass standard for anything entering the paper

An experiment enters §9 only if: the config was frozen before its evaluation data were scored
and the freeze is recorded (hash in the manifest); seeds ≥ 10 for synthetic, all available
channels/files for real data; every printed number traces to a file in `transfer_paper/data`
with a manifest checksum and an evidence-map row; the acceptance rule was stated in advance and
the outcome — pass *or* fail — is printed; a negative or ambiguous outcome is written as such
and never re-tuned into a pass; and the parameter count / compute cost is reported wherever two
classes are compared. These are the rules the paper already follows for S1–S9, CRESST and
TIDMAD; the new experiments are held to the same standard, and the agent instruction makes
each rule a checkable item.
