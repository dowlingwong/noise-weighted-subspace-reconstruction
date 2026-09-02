# Experiment plan — Paper 3 / ORACLE (2026-09-02)

_Implementable plan for the experiments of the ORACLE proposal
(`ORACLE/latex/paper3_proposal.tex`, §3 Mechanism, §4 Claims C1–C4, §5
Testbeds; `ORACLE/docs/EXPERIMENT_DESIGN.md`; `ORACLE/docs/IMPLEMENTATION_PLAN.md`
WP0–WP11).  Decision of 2026-09-02: **all experiment code lives in this
repository** under `experiments/oracle/`; ORACLE keeps the proposal and the
packages (`noise_module`, `qp_simulator`, `reconstruction_model`,
`tidmad_transformer`, `oracle_paired`).  The `noise_module` copy here is
byte-identical to ORACLE's (verified 2026-09-02).  Companion checklist:
`docs/IMPLEMENTATION_CHECKLIST.md`._

## 0. Scope of this plan

Tier 1 (mechanism, ORACLE-Cov) is fully implementable on CPU here and is where
every theoretical prediction is decided (proposal §3, §5).  Tier 2
(ORACLE-Paired: Prometheus + NuBench DynEdge) and Tier 3 (TIDMAD transformer)
need the cluster and are specified as runbooks only (§4 below).

## 1. Tier 1 testbeds

**Tier-1a — planted-structure waveforms (primary).**  The
`experiments/synthetic/nfpa_confirmatory` generator: C=8 × T=256 events,
planted rank-(2,3) channel–time factorization, physical latents (amplitude,
rise, decay, and the nonseparable deformation weight α), Gaussian noise with
*exact* separable covariance Σ = Σ_t ⊗ Σ_c (1/f² temporal PSD, AR(1)
channels).  Everything the mechanism section needs is known in closed form:
the forward map g(z), its Jacobian by finite differences of the generator,
Σ^{-1/2}, and hence the excited tangent span T_S and the Fisher matrix I(z).
Regimes `colored` and `white` exist; `oracle_cov.families` adds the shifted
regimes.

**Tier-1b — realistic acquisition noise (N-family realism).**  `noise_module`
composite PSD (Johnson/SQUID/TD/ER components from `Al2O3_Al_athermal`),
`TemporalNoiseWrapper` nonstationarity, `ArtifactInjector` glitches and
lines, `NonGaussianNoiseGenerator` tails.  The generator reports implied and
realized covariance, so Σ̂ ≠ Σ is a measured lever.  Tier-1b is used only for
the N families that Tier-1a cannot produce (glitches, lines, tails,
nonstationarity); the S families and the designed dissociation stay on 1a.

## 2. Subjects (frozen models)

| Subject | Class F | Assumed covariance Σ̂ in training | Jacobian | Stages hooked |
|---|---|---|---|---|
| `lin_cw` | CW-PCA / tied linear AE, rank 6 | oracle Σ, or estimated Σ̂ from N_cal noise-only events, or deliberately wrong Σ̂ (κ sweep) | analytic: PPᵀΣ̂⁻¹ | code z (6), reconstruction |
| `lin_nfpa` | NFPA (2,3) | same | analytic | code C (2×3), reconstruction |
| `mlp_ae` | 2-hidden-layer MLP autoencoder on whitened input (torch if present, else numpy) | Σ̂-weighted loss | autograd / finite differences | h1, h2, code, reconstruction |

A subject is trained once under Σ̂ on the clean training support and then
frozen.  Output for consequence: amplitude estimate Â (linear readout on the
code fitted on clean training data), and the reconstruction.

## 3. Families (declared before any confirmatory run)

| Family | Contract | How it is produced | Severity knob |
|---|---|---|---|
| `clean` | — | held-out events from the training distribution | — |
| `N1_cov_slope` | N | temporal PSD exponent 2 → 2±δ at deployment (Σ changes, Σ̂ fixed) | δ, reported as κ(Σ̂⁻¹Σ) |
| `N2_cov_channel` | N | AR(1) ρ 0.4 → ρ' | ρ' |
| `N3_jitter` | N | per-event arrival-time jitter σ_t | σ_t |
| `N4_gain_drift` | N | per-channel gain drift ε | ε |
| `N5_glitch` | N | `ArtifactInjector` glitches at rate r (Tier-1b) | r |
| `N6_channel_loss` | N | zero one or more channels | count |
| `S1_amplitude` | S | clean events with amplitude outside the training range | quantile |
| `S2_decay` | S | clean events with decay time outside the training range | quantile |
| `S3_t0` | S | training fixes t0; deployment varies it (the unexcited direction of Prop. coverage) | σ_t0 |
| `S4_alpha` | S | training α=0; deployment α>0 (nonseparable deformation never excited) | α |
| `U1_tails` | undeclared | `NonGaussianNoiseGenerator` heavy tails | ν |
| `U2_line` | undeclared | narrow spectral line | amplitude |
| `null` | designed | representation displacement in the output Jacobian's null space, norm-matched | ‖δ‖ |
| `aligned` | designed | displacement along leading right-singular directions, same norm | ‖δ‖ |

S families are constructed by training on a restricted latent support and
evaluating on the complement; the "unexcited tangent complement" T_S^⊥ is
computed from the generator Jacobian on the training support.

## 4. Monitors and endpoints

**Monitors (all fitted on clean reference only):** standardized Euclidean
displacement of the stage representation; pullback-whitened displacement
(δᵀ J_hᵀ Σ⁻¹ J_h δ, needs Σ); Jacobian-projected displacement (Σ-free);
corrected univariate input tests; RBF-MMD; classifier two-sample test;
embedding mean/covariance distance; output χ̄² residual under Σ̂; layerwise
dominant-subspace alignment a_{ℓ,k}; contrastive-direction alignment for each
declared factor.  Two derived S-signature statistics from the mechanism
section: the fraction of whitened displacement energy in T_S^⊥, and the
Fisher-rank of I(z) on the evaluated stratum.

**Endpoints (proposal §4):**
- C1 detection: power at 1 % FAR on clean windows, delay in windows.
- C2 attribution: macro-F1 over {N,S} with conformal abstention; risk–coverage AUC; N/S F1 at 80 % coverage; ΔF1 layerwise vs all-generic with 95 % interval and the ±0.05 equivalence margin.
- C4 consequence: within severity strata, AUROC for K ≥ κ_m (κ_m = 10 % amplitude degradation), conditional high-consequence risk, off-diagonal rates; the `null`/`aligned` pair with predicted sign (pullback ≫ null; Euclidean cannot separate).
- Mechanism predictions (pre-registered before Tier 2): N cells put their displacement inside T_S; S cells put ≥ f* of theirs in T_S^⊥; layer alignment a_{ℓ,k} drops at early stages for N and stays near one for S.

## 5. Experiments (Tier 1)

| ID | Name | Families | Subjects | Primary endpoint | Accept |
|---|---|---|---|---|---|
| P3-E0 | Positive controls & analytic unit tests | `null`, `aligned`, white regime | all | pullback = Mahalanobis of Wδ to 1e-9; Jacobian-projected of null = 0; white regime: Euclidean = pullback up to scale | tests green |
| P3-E1 | κ sweep (N1, N2) | N1, N2 | lin_cw, mlp_ae | monotone alarm and K in κ; C1 power | monotone; power ≥ 0.8 at κ ≥ κ* |
| P3-E2 | Designed dissociation (cov_B) | `null`, `aligned` | all | predicted sign for pullback and Jacobian-projected; failure of Euclidean by construction | sign holds all seeds |
| P3-E3 | S-signature: tangent confinement | S1–S4 vs N1–N4 | lin_cw, mlp_ae | fraction of displacement energy in T_S^⊥, N vs S | separation with interval excluding zero |
| P3-E4 | Attribution with abstention | N1–N6, S1–S4, U1–U2, mixtures | all | C2 endpoints | reported either way (equivalence margin) |
| P3-E5 | Alarm–consequence within strata | all declared | all | C4 endpoints | reported |
| P3-E6 | Layerwise profile | N, S | mlp_ae | a_{ℓ,k} and v_{ℓ,h} predictions | reported |
| P3-E7 | Cost | — | mlp_ae | hooks benchmark p50/p95, bytes/event, 10 % sampling loss | measured |

All runs: 10 seeds for development, 20 held-out seeds for confirmatory;
resampling units are event groups and perturbation seeds (nested); every
result table carries config hash, git commit, seed list and output digests.
Dev and confirmatory outputs are written to different directories and only
the latter may be cited.

## 6. Tier 2 and Tier 3 (cluster; runbooks only)

- **Tier 2 ORACLE-Paired:** `ORACLE/src/oracle_paired` (Prometheus injection reuse across ORCA/ARCA, NuBench response reimplementation, N1–N5/S1–S5/U1–U4 cells, content-matched clean twins). Runbook to write: `docs/plans/oracle/TIER2_RUNBOOK.md` (production, DynEdge hooks, the pre-registered prediction table from Tier 1, one confirmatory run).
- **Tier 3 TIDMAD:** `ORACLE/src/tidmad_transformer` trained twice (MSE vs inverse-PSD); monitors from `experiments/oracle` applied to its hooks; consequence K_rel. Runbook: `docs/plans/oracle/TIER3_RUNBOOK.md`.

## 7. Repository layout for the implementation

```
experiments/oracle/
  oracle_cov/            # the Tier-1 package (importable as experiments.oracle.oracle_cov)
    protocol.py          # Family, Window, RepresentationBatch, Metric/Perturbation/Consequence protocols
    subjects.py          # lin_cw, lin_nfpa, mlp_ae — train under Σ̂, freeze, hooks, Jacobians
    families.py          # every family above; each records its parameters into meta
    tangent.py           # generator Jacobian, T_S, T_S^⊥ projector, Fisher rank
    monitors.py          # all monitors incl. pullback / Jacobian-projected / layerwise
    baselines.py         # MMD, C2ST, input tests, embedding distance
    consequence.py       # amplitude K, relative degradation, κ_m rule
    abstain.py           # split-conformal nonconformity, risk–coverage
    report.py            # nested bootstrap, BH-FDR, manifests
    tests/               # analytic unit tests and positive controls (P3-E0)
  run_e1_kappa.py … run_e7_cost.py
results/oracle/<ID>/{dev,confirmatory}/
docs/plans/oracle/{TIER2_RUNBOOK.md, TIER3_RUNBOOK.md, PREREGISTRATION.md}
```
