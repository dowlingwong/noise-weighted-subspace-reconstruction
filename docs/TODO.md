# noise-weighted-subspace-reconstruction — TODO (as of 2026-09-03)

All experiments for Papers 1 and 3 live here. Status detail: `docs/IMPLEMENTATION_CHECKLIST.md`;
findings and open items: `docs/reviews/2026-09-02_experiments_review.md`.

## Housekeeping
- [ ] Delete `export/*.tgz` (transfer bundles from the 2026-09-02/03 container run; gitignored,
      11 MB) and the empty `archive/2026-09-02_md_sweep/paper_rps2026_tmp/tmp*.tmp`.
- [ ] Delete `.git/_stale_locks/` (moved lock files; the worktree metadata is already gone — verified 2026-09-03: `git worktree list` shows main only, `.git/worktrees/` empty).
- [ ] Decide the three deleted-but-uncommitted files (`OPEN_ISSUES.md`,
      `paper_rps2026/RPS_MINIPAPER_HANDOFF.md`; `docs/REMOTE_EXECUTION.md` is already committed as
      deleted but still listed in `docs/README.md` — restore or drop the link).
- [ ] Delete stale local branches after the R/PS decision (`codex/*`, `round9/*`, `round10/*`).

## Paper 1 runs (dispatch prompt: manuscript repo `plan/AGENT_PROMPT_REMOTE_2026-09.md`)
- [ ] P1-E2 at the frozen protocol (10 seeds × 6 α).
- [ ] P1-E3 calibration-size sweep (`experiments/paper1/run_e12_local.sh`).
- [ ] P1-E6 at n ≥ 20 000.
- [ ] P1-E1 at 10 seeds; re-render Figure 1; fix the panel-(c) caption overlap in
      `scripts/make_paper1_fig1.py`.
- [ ] Register all P1 runs in `transfer_paper/MANUSCRIPT_EVIDENCE_MAP.md` + bundle manifest.
- [ ] **P1-E8 noise-Laplacian PE uniqueness** (author decision 2026-09-03; spec in
      `docs/EXPERIMENT_PLAN_PAPER1.md` §P1-E8):
  - [ ] `experiments/paper1/pe_uniqueness.py --part a` kernel identification: eigenbasis-vs-DFT
        angles vs d; G_noise / shift invariance defects for sinusoidal, chain-Laplacian,
        noise-Laplacian PEs; stationary (TIDMAD PSD) and nonstationary Σ.
  - [ ] `--part b` weight sweep γ ∈ {0 … 1.5}, Var/CRB of a linear readout; argmin at γ = 1.
  - [ ] `experiments/paper1/pe_models.py` + `--part c` learning ladder (2-layer 1-head attention,
        ≈50k params matched across arms, 5 PEs × raw/whitened × stationary/nonstationary,
        train SNR 10 / test 3, 10, 30; 5 seeds); write predictions (1)–(4) into the manifest
        *before* running; report parameter count and CPU inference time per arm.
  - [ ] `scripts/make_paper1_fig_pe.py`; register P1-E8 in the evidence map.
  - [ ] Run a and b now (no learning, CPU, ~1 h); run c only after App. E.5 statement is fixed.

## Methodology adopted from Kieseler (GravNet/GarNet 1902.07987; object condensation 2002.03605)
- [ ] Every learned-model comparison (P1-E8c, Tier-1 `mlp_ae`, Tier-2 DynEdge) reports a
      resource row: parameter count, training time, CPU/GPU inference time per event, memory —
      arms matched in parameter count, not in "effort".
- [ ] Report inclusive *and* subset metrics side by side (e.g. all events vs the low-SNR / overlap
      / boundary subset) instead of one headline number.
- [ ] Extrapolation arm in every learned ladder: train in one regime, test outside it (SNR,
      multiplicity, noise family), and say so in the caption.
- [ ] One hyperparameter table per learned method with the values actually used (q_min / s_B style).
- [ ] Paper 3: proof-of-concept on a toy (Tier 1) *before* the physics benchmark (Tier 2), with the
      same metric names in both — already the tier structure; make the bridge table explicit.

## Paper 3 Tier 1 (ORACLE-Cov)
- [ ] Freeze `docs/plans/oracle/PREREGISTRATION.md`; set `n_clean_windows ≥ 300` (100 per third).
- [ ] Fix before confirmatory (review C1, C3, C6, C10): stronger `mlp_ae` (CW-PCA-initialised
      encoder or a denoising target); Tier-1b N families through `noise_module.ArtifactInjector`
      and `NonGaussianNoiseGenerator`; single-pass hook in `pipeline.make_batch`; restate C3.
- [ ] Add a per-event / second-moment statistic for S4 (invisible in the mean).
- [ ] Confirmatory runs: `run_tier1 --mode confirmatory` (P3-E1, E3, E4, E5, E6), `run_e2_dissociation`,
      `run_e7_cost`, then `augment_k_recon` and `analyze_tier1` — on the cluster (jobs must outlive
      a shell).
- [ ] Bridge table: pre-register Tier-2 failure predictions from the confirmatory Tier-1 output.

## Paper 3 Tier 2/3 (cluster; runbooks in `docs/plans/oracle/`)
- [ ] `TIER2_RUNBOOK.md`: parity gate → ORACLE-Paired production → DynEdge hooks → one confirmatory run.
- [ ] `TIER3_RUNBOOK.md`: TIDMAD arms as frozen subjects, five seeds per arm.

## Review
- [ ] Run `docs/REVIEW_PROMPT_2026-09.md` with an independent agent; file the report under
      `docs/reviews/`; act on its "three things first".
