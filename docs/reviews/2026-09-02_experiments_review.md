# Review of the 2026-09-02 experiment implementation (Papers 1 and 3)

_Self-review of `experiments/paper1/`, `experiments/oracle/`, `scripts/make_instrument_table.py`,
`scripts/make_paper1_fig1.py`, run before the results were read.  Each finding says what was
wrong, what was changed, and what remains open.  Findings marked **OPEN** are for the
external reviewer (`docs/REVIEW_PROMPT_2026-09.md`)._

## A. Defects found and fixed during implementation

| # | Where | Finding | Fix |
|---|---|---|---|
| A1 | `paper1/alpha_sweep.py` | Principal angles were converted to degrees twice (`principal_angles(..., degrees=True)` then `np.degrees`), giving 3900°. | Removed the second conversion; angles now computed through `r9_semantics.fitted_basis_whitened`, the E11-corrected unstacking, not `methods.basis_whitened` (known Iso-MPCA vectorisation defect). |
| A2 | `oracle_cov/families.designed_perturbation` | The null arm drew a fresh random null direction per event; averaged over a window these cancel, so the Euclidean monitor separated `null` from `aligned` for a reason unrelated to the metric. | One coherent direction per window, Jacobian at the window mean (where the monitors linearise). Euclidean now cannot separate (40 vs 43 at smoke scale); pullback 32 vs −0.6. |
| A3 | `run_e2_dissociation.py` | Consequence was amplitude-only while the arms were built from the reconstruction Jacobian, so `aligned` carried no amplitude consequence. | Added `K_recon` (whitened reconstruction error against the clean signal) for the recon arms and a second pair `null_amp`/`aligned_amp` built from the amplitude gradient, scored by the new `pullback_output` monitor. |
| A4 | `pipeline.tangent_statistics` | Energy fractions of the mean displacement were noise-dominated (pure noise puts ~d/n of energy everywhere) and ignored (i) the reference mean's own noise and (ii) signal sampling variability along T_S; clean windows scored z ≈ 13 in T_S. | Standardised every energy against the empirical distribution of clean-window energies (`clean_energy_stats`); clean windows now score z ≈ 0 on all three. |
| A5 | `monitors.OutputResidual` | One-sided alarm; a flatter PSD *lowers* χ̄² and would never alarm. | `two_sided=True` (alarm on the standardised absolute score). |
| A6 | `subjects.MLPSubject.jac_code` | Raised at the `code` stage, breaking the `pullback_output` monitor there. | Identity Jacobian at `code`. |
| A7 | `tests/test_p3_e0.py` | The severity-monotonicity positive control used `output_residual` on S3 up to 8 samples, which saturates (the pulse leaves the subspace) and is machine-dependent at smoke scale. | Control now uses the unexcited-direction energy `mean_disp_excess_z_T_U` at 0.5–4 samples, which is first-order in the shift. |
| A8 | execution | Background jobs on the local VM are killed when a tool call returns; the 20 000-event rank-one check was also memory-bound. | Long runs execute in the cloud container from a tagged snapshot (`export/nwr_code_2026-09-02.tgz`, commit b8abf3c); results are committed back. Rank-one check at n = 6000. |
| A9 | `scripts/make_instrument_table.py` | Provenance printout keyed the wrong dictionary entry. | Fixed; every cell prints its file:key trace. |

## B. Scientific findings from the smoke runs (carry into the proposal)

| # | Finding | Consequence for Paper 3 |
|---|---|---|
| B1 | **N3 jitter is N by contract and S by geometry.** A zero-mean per-event time jitter is a signal deformation; it leaves the residual variance along T_S at 1 (no covariance change) and puts mean-displacement energy in the unexcited span T_U (z ≈ 19 at 2 samples), exactly like S3. | The proposal's "N in the excited span, S in the unexcited complement" is true for covariance-type N but not for acquisition-side signal deformations. `PREREGISTRATION.md` §5 now declares three signatures and names N3 as the ambiguous case resolved by the variance ratio. §3 of the proposal should say this. |
| B2 | **S1/S2 (amplitude, decay quantile strata) move the mean along excited coordinates.** The frozen model has first-order information there (z_S > 0, z_U ≈ 0); the coverage proposition does not make them invisible. Only S3 (t0) and S4 (nonseparable) are unexcited-direction shifts. | The S contract should distinguish "sparse but excited" from "unexcited" strata; the coverage theorem applies to the second only. |
| B3 | **S4 (nonseparable deformation) is invisible in the mean displacement** (zero-mean random R per event) and outside both spans per event. | Needs a per-event or second-moment statistic; the current design detects it only through χ̄² and MMD. |
| B4 | **The frozen linear subject's code is blind to S3**: mean displacement orthogonal to T_S does not enter `code = P^T x`, so Euclidean/pullback/Jacobian-projected alarms stay at zero while K = 0.14 > κ_m (a *missed consequence* cell by construction); the χ̄² residual does see it. | This is the coverage proposition made operational and is the cleanest Tier-1 result to lead with. |
| B5 | Designed dissociation is constructible only where the reconstruction Jacobian has a null space: `mlp_ae` `h1` (64 → rank ≤ 32); not at `lin_cw` `code` (P has full column rank) nor at any 32-dimensional stage. | State this in the proposal's dissociation paragraph. |

## C. OPEN — for the external reviewer

| # | Item | Why it matters |
|---|---|---|
| C1 | `mlp_ae` barely beats the whitened-noise floor (validation loss ≈ 1.04–1.06 per coordinate at smoke scale); at full scale it may still be a weak subject, so its consequence K is noisy and its layerwise profile is less interpretable. Options: initialise the encoder from the CW-PCA solution, add a denoising target, or lower the noise for the nonlinear subject only (declared as a separate regime). | The layerwise claims (C2, P3-E6) rest on this subject. |
| C2 | The 1 % false-alert threshold is set on 20 clean windows and checked on 20; the realised FAR resolves 0.05 at best. Confirmatory runs need ≥ 100 clean windows per seed (cost: ~5× the clean-window budget). | C1 power numbers at dev scale are indicative only. |
| C3 | `N5_glitch` is a home-made Gaussian bump, not `noise_module.ArtifactInjector`; `U2_line` likewise. Tier-1b (noise_module composite PSD + artifacts + non-Gaussian tails) is specified but not wired. | The N-family realism claim in the plan is not yet backed. |
| C4 | `P1-E2` in this session runs at 5 seeds × 4 α with `n_restarts = 4` (config hash records it); the frozen protocol is 10 seeds, 10 restarts, 6 α. | Dev evidence only; rerun before the manuscript cites it. |
| C5 | The pullback-whitened monitor uses the *assumed* covariance (identity in whitened coordinates). Under N1/N2 the realised Σ' differs; a monitor given Σ' would be a different, oracle object. This is the intended deployable definition but should be stated. | Definition clarity in the proposal. |
| C6 | Attribution features include the tangent statistics, which require the forward-model Jacobian (known on Tier 1 only). The Tier-2-deployable arm must drop them; `analyze_tier1` should report a third arm "layerwise without tangent features". | Otherwise the Tier-1 ΔF1 over-states what Tier 2 can inherit. |
| C7 | `P1-E5` rerun compares summary means; a digest-level identity is only possible if the archived JSON carries per-seed rows (it does — compare rows too). | Closure strength of OPEN_ISSUES #0. |
