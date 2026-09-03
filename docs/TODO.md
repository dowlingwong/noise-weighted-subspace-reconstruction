# noise-weighted-subspace-reconstruction — TODO (as of 2026-09-03)

All experiments for Papers 1 and 3 live here. Status detail: `docs/IMPLEMENTATION_CHECKLIST.md`;
findings and open items: `docs/reviews/2026-09-02_experiments_review.md`.

## Housekeeping
- [ ] Delete `export/*.tgz` (transfer bundles from the 2026-09-02/03 container run; gitignored,
      11 MB) and the empty `archive/2026-09-02_md_sweep/paper_rps2026_tmp/tmp*.tmp`.
- [ ] Delete `.git/_stale_locks/` (moved lock files and the pruned codex worktree metadata).
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
