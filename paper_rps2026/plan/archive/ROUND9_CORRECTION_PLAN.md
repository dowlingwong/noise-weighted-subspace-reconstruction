# Round 9 correction plan — external audit of 2026-08-19

Source: the second external audit (verdict: major revision; mathematical errors;
rung-4 pseudoreplication; predeclared-but-untested hypotheses). **Every criticism
was verified against the files before planning; all five mathematical points and
the H3/H4 claim-to-evidence mismatch are confirmed real.** The previous review's
"minor revision" verdict was too lenient and is corrected in an addendum.

## Verification record (what was checked, where)

| Criticism | Verified | Evidence |
|---|---|---|
| Full-rank-factor claim false | **YES** | §3:20; with `k₁=n₁, k₂<n₂` the family is `ℝ^{n₁}⊗U₂`, strictly separable; the paper's own `C=k_c=2` run shows NFPA≠CW-PCA (mean angle 61.7°, appendix B.2) |
| Rank-one CW-PCA ≡ OF conflation | **YES** | §2:18 "at rank one this is the optimal filter" — learned leading direction vs. fixed template; equal only under Corollary 1's assumptions |
| NFPA formula mixes coordinates | **YES** | §3 introduces `Z` as the measurement; the projector formulas hold for the per-axis whitened `Z̃`; the implementation whitens/unwhitens correctly (`covariance.py`), the notation does not |
| "Same factorization, different nonlinearity" too strong | **YES** | no containment proof exists; §3 rung 4 and §1's novelty list both overstate |
| "Closed-form" at rungs 1–3 misleading | **YES** | NFPA's joint product-manifold problem is nonconvex; only the alternating subproblems are exact (appendix B.2 already says so — main text contradicts it) |
| H3 predeclared, not executed | **YES** | appendix B.2 declares H3 (planted-subspace recovery); `semantic.py` implements `planted_subspace_errors`/`amplitude_metrics`; `run_blocks.py` never imports `semantic` |
| H4 claim ("covered by rung-4 amplitude error") unsupported | **YES** | no rung-4 amplitude-error result exists in the manuscript or `results/rps2026/` |
| Stale 30-seed plan doc | **YES** | `notes/nfpa_submission_experiment_plan.md` §8 says 30 seeds; the executed, frozen protocol is 10 (2026-08-08 amendment, run-log freeze record) |
| Rung-4 CIs are within-trajectory | **YES** | already disclosed, but the "excluding zero" emphasis retains inferential framing the disclosure contradicts |
| White-noise control is an identity, not evidence | **YES** | with Σ=σ²I the objectives coincide up to scale; exact 0 is a pipeline check |
| Local H3 execution | **NOT FEASIBLE HERE** | one seed's NFPA fit (10 restarts × 150 ALS iters) exceeds the 2-min sandbox call cap → server Round 9 |

## Phase M — mathematical corrections (main text)

1. §3 proof sketch: delete "When either factor is full rank the restriction is
   inactive and rung 3 reduces to rung 2". Replace with the true statement: the
   restriction is inactive exactly under Proposition 2's condition; the
   `C=k_c=2` appendix run is the counterexample to the deleted claim and is
   now cited as such.
2. §2 proof sketch: separate the learned rank-one direction from the fixed
   template; coincidence only under Corollary 1's assumptions.
3. §3 rung 3: state the projection on `Z̃ = Σ₁^{-1/2} Z Σ₂^{-1/2}` with map-back;
   keep the already-added `Σ = Σ₂⊗Σ₁` hypothesis.
4. §3 rung 4 and §1: "same factorization at different levels of nonlinearity" →
   "an analogous axis-factored architecture"; §1's novelty item weakened to
   match (observation of parallel structure, not relaxation).
5. Abstract + §1: "closed-form answer/analysis" → "exact geometric
   characterization" where it covers rung 3; rung 1's genuinely closed-form
   status is kept.

## Phase S — statistical reframing of rung 4

1. §4: the paragraph opens "As an exploratory case study (one seed per
   objective)"; interval language becomes within-trajectory stability, with the
   explicit sentence that no across-run inference is claimed; "excluding zero"
   framing removed.
2. Figure panel C: whiskers dropped in single-seed mode; title gains
   "(exploratory)"; caption says within-run stability only. (The `--with-seed2`
   mode keeps whiskers per-seed since sign agreement across seeds is then the
   displayed quantity.)
3. Abstract: rung-4 sentence reframed as exploratory; the J-measurement
   refutation stays (it is seed-independent).
4. §5: alternatives named — capacity, optimization trajectory, implicit
   regularization — not only "absorbs the weighting".

## Phase C — control reframing

§4 and caption: the white-noise result is an implementation identity check
(objectives coincide up to scale at Σ=σ²I), not independent empirical
confirmation.

## Phase R — reconciliation

1. Appendix B.2 rewritten to state what ran: H1, H2 executed (10 seeds per the
   2026-08-08 amendment; the earlier 30-seed design superseded before
   execution); **H3 predeclared but not executed** (code exists, queued as
   Round 9 E8); the "covered by rung-4 amplitude error" sentence deleted.
2. `notes/nfpa_submission_experiment_plan.md` moved to `plan/archive/` with a
   header pointing to the executed protocol.
3. Claim audit: new rows for the H3 gap and the corrected statements.

## Phase L — strengthening additions

1. §2 population lemma: for independent zero-mean signal and noise,
   `Cov(Σ^{-1/2}x) = Cov(Σ^{-1/2}s) + I` — whitened PCA orders directions by
   whitened signal power, which is why the leading whitened eigenspace is the
   signal's without clean targets. Short proof in Appendix A. This is the
   self-supervision argument made precise.
2. §1: cite Matrix Normal PCA (Zhang et al., arXiv:1911.10796) in the
   latent-variable line.
3. Workshop fit: self-supervision primary (abstract already leads with it);
   tokenization/transfer kept as implications, not co-equal claims.

## Phase E — Round 9 server runbook (submission-level evidence)

`docs/plans/rps2026/ROUND9_RUNBOOK.md`: E8 semantic/H3 evaluation (planted
projector errors + amplitude decoding, all 10 archived seeds + methods); E9
frozen 20-seed replication extension (seeds 10–29, explicitly labelled
post-hoc replication of the frozen 10-seed protocol); E10 estimated-covariance
condition (Σ̂ from N_cal noise draws, same contrast); E11 additional
independent attention seeds if GPU time before the deadline permits.

## Order

Verify → M1–M5 → S1–S4 → C → R1–R3 → L1–L3 → figure regeneration → build/budget
→ E runbook → review addendum (verdict corrected to major revision, criticisms
credited) → commit. Page budget is enforced at every build; donors identified
per edit.
