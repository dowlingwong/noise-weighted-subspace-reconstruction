# Implementation checklist — Papers 1 and 3 (long horizon)

_Created 2026-09-02 from `EXPERIMENT_PLAN_PAPER1.md`, `EXPERIMENT_PLAN_PAPER3.md`
and an inventory of this repository and ORACLE.  One line per item; tick in
place.  "CPU" items run here in `$HOME/venv`; "cluster" items need the GPU
nodes and the external datasets.  Status legend: `[x]` done, `[~]` partial,
`[ ]` open, `[cluster]` blocked on the cluster._

## A. What already exists (inventory, 2026-09-02)

| Area | Location | State |
|---|---|---|
| Paper-1 core library (OF, weighted PCA/EMPCA, tied AE, PSD/whitening, angles, CRESST/GWOSC/TIDMAD loaders and analyses) | `src/noise_geometry/`, `src/canonical/` | complete; tests in `tests/` |
| S1–S9 synthetic runner and sweeps | `scripts/run_experiment.py`, `sweep.py`, `run_s5_confirmatory_sweep.py`; `results/sweeps`, `results/metrics` | complete; S5 provenance open |
| NFPA confirmatory harness (generator, methods, metrics, semantics, estimated Σ, angle correction, calibration sweep) | `experiments/synthetic/nfpa_confirmatory/` | complete; E12 sweep never completed |
| Rung-4 factored attention on TIDMAD | `src/tidmad/` | complete for the 2-seed pilot |
| Noise simulator (PSD models, temporal nonstationarity, artifacts, tails, multichannel, calibration) | `noise_module/` (identical copy in ORACLE) | complete; unit tests |
| QP trace simulator | `QP_simulator/` | complete |
| Paper-2 reconstruction scaffold (AE/transformer trainers) | `paper2/` | scaffold; not used by any paper now |
| ORACLE packages: `oracle_paired` (toy + geometry + interventions), `reconstruction_model`, `tidmad_transformer` | `ORACLE/src/` | Tier 2/3 subjects; no diagnostics package (`oracle_diag`) exists anywhere |
| ORACLE NuBench feasibility (hooks, dropout positive control) | `ORACLE/scripts/nubench`, `ORACLE/results/` | pilot only, enriched sample |

## B. Paper 1

- [x] **P1-0** Create `experiments/paper1/` package with a shared `common.py` (manifest writer: config hash, git commit, seeds, SHA-256 of outputs) reusing `nfpa_confirmatory.r9_common`. (CPU)
- [x] **P1-E5** S5 provenance audit → `results/paper1/P1-E5/PROVENANCE_REPORT.md`; closes `OPEN_ISSUES.md` #0 in the manuscript repo. (CPU)
- [x] **P1-E1** (4 seeds; regression vs archived E8 to 3e-14) Figure-1 data product (`fig1_factors.py`) + renderer (`scripts/make_paper1_fig1.py`); regression against E8 to 1e-6. (CPU)
- [~] **P1-E2** _(dev run: 5 seeds × 4 α at the frozen 10 restarts; 10 seeds × 6 α still to run)_ Nonseparability α sweep (`alpha_sweep.py`), 10 seeds × 6 α × 4 methods; summary + figure. (CPU, ~1 h)
- [x] **P1-E6** Rank-one corollary check at C=8 (`rank_one_check.py`). (CPU, minutes)
- [~] **P1-E3** _(wrapper written (`run_e12_local.sh`); sweep not yet run)_ E12 calibration-size sweep rerun locally (`run_e12_local.sh` → `run_calibration_sweep.py`); tests `test_round12_sweep.py` green first. (CPU, hours; background)
- [x] **P1-E4** Instrument table compiler (`scripts/make_instrument_table.py`) with per-cell provenance trace. (CPU)
- [ ] **P1-W1** Manuscript wiring after 29 Sep: input `07b_separable_evidence.tex`, add the `\ref`, register runs in `MANUSCRIPT_EVIDENCE_MAP.md`, replace the figure placeholder, add the α-sweep sentence to §6.4, the instrument table to §7. (desk)
- [ ] **P1-W2** Rewrite the abstract to the problem → principle → experiment → striking result → boundary shape, ≤ 200 words; one Limitations paragraph. (desk)
- [cluster] **P1-E7** Rung-4 strengthening (≥ 5 seeds/arm) — only if the R/PS decision asks for it.

## C. Paper 3 — Tier 1 (ORACLE-Cov), all CPU

- [x] **P3-0** `experiments/oracle/oracle_cov/protocol.py` — Family, Window, RepresentationBatch, Metric / Perturbation / Consequence protocols; dev vs confirmatory mode flag; manifest writer.
- [x] **P3-1** `tangent.py` — generator Jacobian ∂g/∂z by central differences on the planted latents; T_S basis (whitened, orthonormalised), T_S^⊥ projector, Fisher matrix I(z) and its numerical rank; unit test: rank equals the number of varied latents on the training support.
- [x] **P3-2** `subjects.py` — `lin_cw`, `lin_nfpa` (train under Σ̂; analytic Jacobians; hooks), `mlp_ae` (torch if importable, numpy fallback with manual backprop; hooks h1, h2, code, recon; Jacobian by autograd or finite differences); amplitude readout fitted on clean training codes; `freeze()`.
- [x] **P3-3** `families.py` — N1–N6, S1–S4, U1–U2, `null`, `aligned`; every family writes its parameters into `meta`; N families change Σ at deployment only; S families are built from restricted-support training + complement evaluation.
- [x] **P3-4** `monitors.py` — Euclidean, pullback-whitened, Jacobian-projected, output χ̄², layerwise a_{ℓ,k}, contrastive v_{ℓ,h}; `baselines.py` — input tests (BH-corrected KS), RBF-MMD (median heuristic), C2ST (logistic, held-out AUC), embedding mean/cov distance. All fit on clean only.
- [x] **P3-5** `consequence.py` — amplitude K (relative RMSE degradation), κ_m = 0.10 rule; `abstain.py` — split-conformal nonconformity, risk–coverage curve, AUC; `report.py` — nested bootstrap (event groups × perturbation seeds), BH-FDR within claim, manifest.
- [x] **P3-E0** Analytic tests and positive controls (`tests/`): pullback = Mahalanobis of Wδ (1e-9); Jacobian-projected(null) = 0; white regime Euclidean ≡ pullback up to scale; monotone alarm vs severity; T_S rank test.
- [~] **P3-E1** _(rows come from the P3-E4 superset run; no separate κ-sweep run yet)_ κ sweep runner (`run_e1_kappa.py`) → `results/oracle/P3-E1/{dev,confirmatory}`.
- [x] **P3-E2** Designed dissociation (`run_e2_dissociation.py`).
- [x] **P3-E3** Tangent-confinement S-signature (`run_e3_tangent.py`).
- [x] **P3-E4** Attribution with abstention (`run_e4_attribution.py`).
- [~] **P3-E5** _(analysed from the P3-E4 rows)_ Alarm–consequence within strata (`run_e5_consequence.py`).
- [~] **P3-E6** Layerwise profile on `mlp_ae` (`run_e6_layerwise.py`).
- [x] **P3-E7** Cost benchmark (`run_e7_cost.py`).
- [~] **P3-PR** _(draft written; not frozen)_ `docs/plans/oracle/PREREGISTRATION.md` — endpoints, thresholds, families, margins, and the Tier-1 → Tier-2 prediction table, hash-frozen before any confirmatory run.

## D. Paper 3 — Tier 2 / Tier 3 (cluster)

- [cluster] **P3-T2a** _(runbook: `docs/plans/oracle/TIER2_RUNBOOK.md`)_ ORACLE-Paired production: Prometheus seeded injection reused across ORCA/ARCA; NuBench response reimplementation; N/S/U cells; clean twins (`ORACLE/src/oracle_paired`). Runbook `docs/plans/oracle/TIER2_RUNBOOK.md`.
- [cluster] **P3-T2b** DynEdge hooks at four blocks + post-processing + readout; parity gate (CPU = GPU ≠ release test); one confirmatory run against the frozen prediction table.
- [cluster] **P3-T3** _(runbook: `docs/plans/oracle/TIER3_RUNBOOK.md`)_ TIDMAD transformer trained twice (MSE vs inverse-PSD), monitors from `experiments/oracle` on its hooks, K_rel on the frozen science-file subset. Runbook `docs/plans/oracle/TIER3_RUNBOOK.md`.
- [cluster] **P3-T4** Panda V2 frozen backbone as the cross-modality subject — blocked on weight release.

## E. Review and hand-off

- [x] **R1** Internal review of C (code + science), findings fixed.
- [x] **R2** `docs/REVIEW_PROMPT_2026-09.md` — the prompt for an independent reviewing agent (Paper 1, Paper 3, implementation).
- [~] **R3** Commit; update `docs/README.md` and `CHANGELOG.md`. _(committed 2026-09-02; Tier-1 dev results synced from the container run)_
