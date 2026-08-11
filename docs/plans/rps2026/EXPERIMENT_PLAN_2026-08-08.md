# Experiment consolidation plan — R/PS 2026 short paper

**Date:** 2026-08-08
**Deadline:** 29 Aug 2026 AoE (= 30 Aug 2026 11:59 UTC). Working backwards, results must be frozen by **25 Aug** to leave time for the manuscript rewrite.
**Scope decision:** Blocks **A + B + H** only. Blocks C–G are deferred to the full paper.
**Codebase:** reuse and extend the Paper 2 implementation at `~/Documents/paper2-nonlinear-reconstruction`.
**Authoritative protocol:** `notes/nfpa_submission_experiment_plan.md`. **That document governs.** This plan scopes it, sequences it, and adds an audit phase. Where the two disagree on a definition, the protocol wins; where they disagree on scope, this plan wins.

---

> ## ⚠ SCOPE AMENDMENT — 2026-08-08, read before starting
>
> The paper is being restructured around an **architecture ladder** (see
> `PLAN_00_MASTER_RESTRUCTURE_2026-08-08.md`, which governs). A fourth rung — a
> factored-attention transformer applied to real detector reconstruction — now
> supplies the paper's headline empirical evidence. **NFPA no longer has to carry
> the paper alone.**
>
> **SECOND AMENDMENT, same day — this supersedes the first.** Rung 4 now runs on
> **TIDMAD**, public and CC-BY-4.0, with no clearance dependency. Track C is the
> critical path and NFPA's role has shrunk further.
>
> **Three changes to this plan, and nothing else:**
>
> 1. **Block A runs at 10 seeds, not 30.** Primary configuration `C = 8 > k_c = 2`
>    unchanged. The referee-facing defect is the `C = k_c = 2` degeneracy, not the
>    seed count; ten paired seeds with an interval retires it.
> 2. **Block H is dropped entirely.** Truth-referenced recovery is now covered by
>    rung 4's amplitude error against TIDMAD's real injected reference channel.
> 3. **Block B is unchanged and remains non-negotiable.**
>
> Phase 0 (audit and correctness tests) is **unchanged and still mandatory**. All
> definitions, acceptance gates, leakage protocol, and the output contract below
> stand exactly as written. Everywhere this document says "30 seeds" for Block A,
> read "10 seeds" and adjust the figure caption accordingly.
>
> **No clearance trigger remains.** The earlier DELight escalation condition is void.
>
> **Stretch goal.** Once Track C's STFT data path exists, running NFPA and CW-PCA on
> the same whitened TIDMAD spectrograms is cheap — ALS plus an eigendecomposition, no
> neural training — and would put rungs 2, 3 and 4 on one dataset in one
> representation. Attempt only after Blocks A and B are complete, and only if it
> cannot delay Track C. The synthetic Block A remains the primary NFPA evidence
> either way, because it is the controlled setting where planted factors are known.
>
> **Repository path correction** (from `plan/CODE_AUDIT_2026-08-08.md`): the Paper 2
> implementation is at `~/Documents/noise-weighted-subspace-reconstruction`, not
> `~/Documents/paper2-nonlinear-reconstruction`, which is LaTeX-only. Everywhere
> below that names the latter, read the former.
>
> **Ownership change.** `sections/03_nfpa.tex` is being replaced by
> `sections/03_architecture_ladder.tex`, created by Track C. You own **rung 3**
> within it. Do not create or restructure that file yourself; port the NFPA
> content into the rung Track C has laid out.

---

---

## 0. The problem being solved

The current NFPA evidence is one seed, one train/test split, and `C = k_c = 2`. That last condition makes the channel projector the identity matrix, so the headline experiment does not test channel compression at all — it tests a shared temporal restriction. The paper discloses this honestly (`sections/04_experiments_discussion.tex` line 7), which means a referee gets to quote the authors' own admission back at them.

The reported effect is also thin: `χ̄²` of 0.9916 for NFPA against 0.9929 for CW-PCA is a 0.13% difference with no interval. The paper correctly declines to interpret it, but a paper whose novelty is analytical needs its one empirical section to be unimpeachable.

**Consolidating Blocks A, B and H removes every cheap objection a referee can make.** This is the highest-leverage work available before the deadline.

---

## 1. Phase 0 — Audit before you run anything (do not skip)

You are reusing an existing codebase. Before generating a single confirmatory result, establish that the estimators are correct. If a bug is found after the 30-seed run, there is no time to redo it.

### A0.1 Inventory

Locate and read in `~/Documents/paper2-nonlinear-reconstruction`:

- the NFPA / ALS implementation
- the CW-PCA implementation
- the whitening code
- the data generator
- the metric functions
- `plan/status_and_open_work.md`

Write `plan/CODE_AUDIT_2026-08-08.md` recording file paths, entry points, and the git commit hash you are working from. Everything downstream references that hash.

### A0.2 Correctness tests — all must pass before Phase 1

Write these as an actual test file, not as ad-hoc checks.

1. **Whitening identity.** For the generated `Σ`, verify `‖W Σ Wᵀ − I‖_F / d` is below a fixed tolerance (protocol §5, check 3).
2. **CW-PCA reduces to PCA under white noise.** With `Σ = σ²I`, the CW-PCA subspace must equal the ordinary PCA subspace to numerical tolerance. Principal angles below 1e-6 degrees. *(Paper 1 already reports 1.48e-6 degrees for this; you should reproduce that order of magnitude.)*
3. **Rank-one NFPA equals OF.** For a separable template under isotropic whitened noise, the rank-one NFPA direction must match the optimal-filter template up to sign, absolute cosine ≥ 0.9999. *(This is Corollary 1; the archived run reports 1.0000.)*
4. **Kronecker projector identity.** For random `U_c`, `U_t`, `Z`, verify `vec(U_c U_cᵀ Z U_t U_tᵀ) = (U_t U_tᵀ ⊗ U_c U_cᵀ) vec(Z)` numerically. This is Eq. (8) and Appendix A.2.
5. **Idempotence and symmetry** of the Kronecker projector.
6. **NFPA with `k_c = C` and `k_t = T` is the identity map.** Reconstruction error must be zero.
7. **Degenerate-case reproduction.** Re-run the archived `C = k_c = 2` configuration and confirm you reproduce `χ̄²` = 0.9916 (NFPA) and 0.9929 (CW-PCA) and the principal-angle vector `[0.0, 59.3, 64.6, 73.8, 84.8, 87.8]`. **If you cannot reproduce these, stop and escalate.** A discrepancy means either the code has drifted or the archived numbers are wrong, and both are blocking.

### A0.3 Leakage audit

Verify, by reading the code rather than by assuming:

- the training mean is estimated on training data only and the *same* mean is subtracted from validation and test
- covariance is never fitted on data containing signal events
- no test metric influences any hyperparameter, rank, shrinkage or restart selection
- the frozen model is evaluated on test exactly once

Write a test that fails loudly if a test partition is touched before final evaluation.

### A0.4 Recover the sub-threshold values (task D7 from the desk plan)

While in this codebase, re-export the IsoAE and IsoPCA held-out raw MSE at full precision from the archived run. Hand these to the desk agent. If Block A supersedes the table, note that and move on.

**Phase 0 gate:** all seven correctness tests pass, leakage tests pass, `plan/CODE_AUDIT_2026-08-08.md` written. Do not proceed otherwise.

---

## 2. Phase 1 — Extend the harness

### A1.1 Non-degenerate data generation (the critical change)

The whole point of this consolidation is `C > k_c`. Implement the generator from protocol §4 with the primary configuration from §3:

| Quantity | Value |
|---|---:|
| Channels `C` | **8** |
| Samples `T` | 256 |
| Channel rank `k_c` | **2** |
| Temporal rank `k_t` | 3 |
| Total matched rank `k` | 6 |
| Training events | 800 |
| Validation events | 400 |
| Test events | 2,000 |
| Seeds | 30 |
| ALS restarts per seed | 10 |
| SNR | 10 |
| Noise | Brownian `1/f²` temporal PSD with channel correlation |
| Covariance | oracle |
| Non-separability `α` | 0.1 |

Requirements that are easy to get wrong:

- **`v_a` and `t_b` must be orthonormal** in their declared coordinates.
- **The non-separable component `R_i` must be projected out of `span(V) ⊗ span(T)`** and total clean-signal power renormalized at every `α`. Without this, the `α` knob silently becomes an SNR knob. Even though you are not running the `α` sweep (Block E), the primary run uses `α = 0.1` and must be constructed correctly.
- **SNR is defined as `SNR² = E[sᵀ Σ⁻¹ s]`** over the complete vectorized clean event. Rescale the *signal*, not the noise. Record realized mean and standard deviation of SNR in every output file.
- Latent variables must include an event amplitude, a rise-time or time-shift parameter, and a decay-time or width parameter. Block H needs these as ground truth.

Assert `C > k_c` and `T > k_t` at generation time and fail hard otherwise. This is the exact defect being fixed; make it impossible to reintroduce.

### A1.2 Covariance construction

Use `Σ = Σ_t ⊗ Σ_c` (protocol §5), axis-preserving so NFPA factor interpretation is meaningful. `Σ_c` should carry moderate off-diagonal correlation, e.g. AR(1) with `ρ = 0.4`. Run all five numerical checks from protocol §5 before generating data, and record their results.

### A1.3 Method interface

All five methods behind one interface, fitted at matched total rank `k = k_c k_t = 6`:

| Method | Metric | Class |
|---|---|---|
| CW-PCA | `Σ⁻¹` | unrestricted rank 6 |
| NFPA | `Σ⁻¹` | factored (2, 3) |
| Iso-MPCA / IsoAE | identity | factored (2, 3) |
| IsoPCA | identity | unrestricted rank 6 |
| OF | `Σ⁻¹` | fixed rank-one template |

**Resolve the Iso-MPCA / IsoAE duplication first.** Protocol §6 flags this: check whether their fitted projectors are mathematically and numerically identical under the tied linear architecture. If they are, report **one** method and mention the alternative parameterization in a footnote. Presenting two names for one estimator is the kind of thing a careful referee catches.

### A1.4 Semantic decoders (Block H)

Fit a linear map from latent codes to physical parameters, **on the training split only**, and evaluate on held-out test events against known truth. Report amplitude RMSE / bias / correlation, rise-time RMSE, decay-time RMSE, and channel- and temporal-subspace projector error against the planted `span(V)` and `span(T)`.

Compare **projectors**, not individual basis vectors — rotations and signs are not identifiable.

### A1.5 Output contract

One tidy row per `(seed, method, condition, split)`, with every field listed in protocol §13. Write raw seed-level results to disk **before** any aggregation. Generate all tables and figures from those raw files with a deterministic script. Never transcribe a number by hand.

---

## 3. Phase 2 — Freeze, then run

### A2.1 Freeze (do this as an explicit, timestamped commit)

Before looking at any aggregate result, freeze: the generator, the metric definitions, the seed list, the primary contrasts, and the acceptance gates. Commit with a message that says the configuration is frozen. **Smoke tests must not be used to tune confirmatory thresholds.**

### A2.2 Block A — primary paired test (30 seeds)

The main paper result. Primary contrasts, predeclared:

```
NFPA − Iso-MPCA   under normalized chi-square     → NFPA expected lower
NFPA − Iso-MPCA   under raw MSE                   → NFPA expected higher
NFPA − CW-PCA     under normalized chi-square     → restriction cost, direction not predicted
```

The first two test metric reversal **at fixed architecture** — this is the load-bearing contrast, because both models share the same factored class, so the only difference is the training metric. The third measures the cost of the Kronecker restriction and **must not be described as superiority** unless the paired interval supports it.

Every method sees the same data within a seed. **Pair by seed.** The seed, not the event, is the resampling unit.

Report mean and median paired difference, 95% bootstrap CI at seed level, standardized paired effect size, and archive all 30 individual seed values.

### A2.3 Block B — white-noise control (20 seeds)

Same clean signals, isotropic noise. Confirm covariance-aware and Euclidean projectors agree to numerical tolerance and that metric-reversal intervals include zero or fall inside a **predeclared** equivalence margin. Declare that margin before running.

This block is what converts "the covariance-aware method won" into "covariance geometry, and not rank or optimization, caused the separation." Do not skip it to save time — it is cheaper than Block A and carries disproportionate evidential weight.

### A2.4 Block H — semantic recovery

Run alongside Block A on the same seeds. Hypothesis H4: NFPA codes should recover amplitude and at least one pulse-shape parameter more accurately than the isotropic factored baseline under colored noise, evaluated against known latents rather than inferred from reconstruction loss.

This is the block that answers the workshop's own framing — "a useful scientific representation should expose structures that scientists can act upon." If only one optional thing survives a time crunch, keep amplitude recovery.

---

## 4. Acceptance gates

Do **not** promote results into the main paper unless all of these hold (protocol §11):

1. All 30 primary seeds complete with no post-result configuration changes.
2. `C > k_c`, and both planted factor projectors are genuinely non-trivial.
3. Fixed-architecture metric-reversal contrasts have intervals excluding zero in the predicted directions.
4. The white-noise control passes its predeclared criterion.
5. NFPA optimization diagnostics show stable convergence, or failure rates are clearly reported.
6. Downstream amplitude and pulse-shape metrics computed from held-out truth.
7. Covariance fitting and centering pass the leakage audit.
8. Table and figure generated directly from archived seed-level results.

**If a gate fails, report the failure — do not adjust the configuration to pass it.** Specifically:

- If the NFPA − CW-PCA interval includes zero: state that restricted and unrestricted models are indistinguishable at available precision. That is a perfectly publishable sentence.
- If CW-PCA wins: report the restriction cost. Also fine — it is what Proposition 2 predicts when the minimizer is not separable.
- If NFPA wins: describe it as a held-out finite-sample result, support it with train/test gaps, and only then offer regularization as an explanation.
- **If metric reversal itself fails to appear:** stop and escalate immediately. That contradicts a theorem-backed claim and means either a bug or a misspecified generator. Do not write it up before diagnosis.

---

## 5. Deliverables

### A5.1 The figure (protocol §12)

Two panels, replacing the slot the desk agent reserved in `sections/04_experiments_discussion.tex`:

- **Panel A:** paired relative difference between NFPA and Iso-MPCA for raw MSE and normalized likelihood residual, with 95% CIs and a vertical zero line. Label the preferred direction explicitly. **Do not put incomparable raw units on one axis.**
- **Panel B:** seed-level subspace alignment to CW-PCA for NFPA and Iso-MPCA — violin-plus-box or interval-and-points of mean or maximum principal angle.

Colorblind-safe palette, vector output, readable at final column width. Caption states matched total rank, shared factored architecture, and `n = 30 seeds`. **No curved-manifold language anywhere.**

### A5.2 The table

Replace `tables/results_summary.tex` with the 30-seed summary carrying paired intervals. Generated by script from archived results.

### A5.3 Manuscript edits (yours to make, in coordination with the desk agent)

Per protocol §14:

1. Replace the single-split table with the 30-seed summary and paired intervals.
2. Replace the `Factored proof-of-concept` paragraph (line 7) with the non-degenerate `C = 8 > k_c = 2` configuration.
3. Insert the two-panel figure.
4. Move the `C = k_c = 2` result to `appendix/appendix_experiment_details.tex` as an implementation check, or remove it.
5. Add one sentence interpreting amplitude and pulse-shape recovery as the representation's actionable scientific content.
6. Update `sections/00_abstract.tex` — **only** with effects supported by the final archived analysis. The abstract currently says "A controlled, single-split proof-of-concept illustrates metric reversal"; that sentence must change, and must not overclaim.
7. Update `sections/04_experiments_discussion.tex` line 16 — the limitation about single-seed evidence is no longer accurate and must be rewritten to describe the *remaining* limitations (no SNR sweep, no noise-color sweep, no covariance-misspecification boundary, oracle covariance only, synthetic data).
8. Update `notes/claim_audit.md` — move the affected rows from single-split status to 30-seed status, and record the new Block H claims.

**Boundary with the desk agent:** you own `sections/04_experiments_discussion.tex` lines 6–13 and 16, `tables/`, `figures/`, and the abstract's evidence sentence. The desk agent owns lines 4, 15, 18–19, the introduction, and the new Paper 3 paragraph inserted between lines 4 and 6. Coordinate before touching anything outside your set.

**Page budget:** your rewrite should *free* roughly 0.3 pages, because the hedging in lines 11 and 13 collapses once intervals exist. The desk agent is counting on this. If your section grows instead, say so early.

---

## 6. Schedule

| Dates | Work | Gate |
|---|---|---|
| 8–10 Aug | Phase 0: audit, correctness tests, leakage tests, degenerate-case reproduction | All 7 tests pass |
| 11–14 Aug | Phase 1: `C > k_c` generator, covariance construction, method interface, semantic decoders, output contract | Smoke test on 2 seeds runs end to end |
| 15 Aug | Freeze configuration, timestamped commit | Frozen before any aggregate is viewed |
| 16–20 Aug | Block A (30 seeds), Block B (20 seeds), Block H | All seeds complete |
| 21–22 Aug | Audit raw outputs, then generate table and figure by script | Acceptance gates evaluated |
| 23–25 Aug | Manuscript edits, coordinate with desk agent | Build clean, ≤ 4 pages |
| 25 Aug | **Results frozen** | Hand back for review |
| 26–29 Aug | Review, corrections, submission | — |

**Slack:** four days. If Phase 1 slips past 16 Aug, drop Block H's pulse-shape parameters and keep amplitude only. If it slips past 18 Aug, run Block A at 20 seeds rather than 30 and state the reduced count honestly. **Do not** drop Block B — it is small and evidentially disproportionate. **Do not** run fewer than 15 seeds; below that the bootstrap interval is not credible and you are better off keeping the current honest single-split framing.

---

## 7. Handoff back for review

Produce `plan/EXPERIMENT_LOG_2026-08-08.md` containing:

1. Code audit results and the git commit hash used.
2. Pass/fail for all seven correctness tests and the leakage tests.
3. The frozen configuration and its commit, with timestamp, demonstrating the freeze preceded aggregate inspection.
4. Acceptance-gate evaluation, one line per gate, including any that failed and what you did.
5. Paths to raw seed-level result files and the deterministic scripts that produced every table and figure.
6. Realized SNR statistics and covariance check results.
7. A plain statement of what the experiment does and does **not** now support, written so it can be pasted into the limitations paragraph.

Do not mark complete if any acceptance gate failed without an accompanying decision recorded, or if any number in the manuscript cannot be traced to an archived file.
