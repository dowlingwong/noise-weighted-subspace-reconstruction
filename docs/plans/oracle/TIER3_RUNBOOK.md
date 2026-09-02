# Tier 3 runbook — TIDMAD compact transformer (cluster)

_Needs the TIDMAD release (`PAPER1_DATA_ROOT`) and a GPU.  The two trained arms
(MSE and inverse-PSD-weighted) already exist from the R/PS rung-4 work
(`results/rps2026/E3_t_mse`, `E4_t_chi2`); this tier re-uses them as frozen subjects._

## Steps
1. Load both arms from `ORACLE/src/tidmad_transformer` (or `src/tidmad`) with the E5/E7
   configs; hook: input embedding, each encoder block, pooled, readout.
2. Subject adapter: `represent`, `outputs` (denoised band-frame reconstruction), `jac_recon`
   by autograd; Σ̂ is the training PSD (inverse-PSD arm) or identity (MSE arm), so the
   pullback-whitened monitor uses the archived Welch PSD (`transfer_paper/data/tidmad/current`).
3. Families on real noise: N1 PSD change by file/epoch (use the release's file-to-file PSD
   drift as natural severity), N5 glitch injection, N6 band loss; S: injected-signal
   frequency strata outside the training sweep; U: line injection.
4. Consequence: K_rel = relative degradation of the injected-amplitude recovery on the frozen
   science-file subset (the release's own score is development-only).
5. Run with 5 seeds per arm (the rung-4 two-seed pilot is under-powered); report the
   MSE-vs-inverse-PSD contrast of every monitor's C4 AUROC as the "two Σ̂ on one real noise"
   paragraph of the proposal.

## Compute
One GPU-day for inference and monitoring over the validation files; no retraining.
