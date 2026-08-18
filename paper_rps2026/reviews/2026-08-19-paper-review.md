# Paper Review — 2026-08-19

**Paper:** *Noise Covariance as Representation Geometry: One Metric, Four
Architecture Classes* (R/PS 2026 workshop submission, `paper_rps2026/`)
**Scope reviewed:** full main text §1–§5, both appendices, all quantitative
claims, build. · **Build:** PASS — 4 pages main content, references from p5,
0 overfull, 0 undefined, 0 `\PENDING`, anonymity sweep clean.
**Verdict (workshop calibration): minor revision — the minor revisions were
applied during this review.** Ready for the advisor now; ready for submission
after the three external dependencies land (below).

**Reviewer-independence caveat:** this audit was performed by the same agent
that drafted the recent revisions. Numeric verification was done against
archived files, not memory, which is where that bias matters least; the
novelty and framing judgments are where it matters most, and the advisor's
read is the real test of those.

## Summary

The paper is honest, internally consistent, and its headline numbers all trace
to archived records — I recomputed every main-text interval from the raw CSVs
and they reproduce, most to the printed digit. The audit found **two genuine
numeric errors** (a transposed χ²/dof and a misrounded interval), **two
missing mathematical hypotheses** (Σ⁻¹-orthonormality of the basis in rung 2's
reconstruction formula; Kronecker-separable Σ for Proposition 2), all now
fixed. The experiments are sound for the claims as worded; the two known
evidence weaknesses (rung-4 single seed, autocorrelated evaluation bootstrap)
are disclosed in the text rather than discoverable. The remaining blockers are
external: the E6 JSON, the E7 seed, and the `priorThesis` source.

## Blocking findings (all fixed during review)

1. **Transposed number: χ²/dof = 1.0002 → 1.0031.** §3 rung 4 and §5 quoted
   `1.0002` as the TIDMAD readout's stationary-PSD adequacy. The archived
   record (`transfer_paper/data/tidmad/current/TIDMAD_P4_real_seed20260727.json`)
   gives `global_model.chi2_per_dof_mean = 1.0031`; `1.0002` is the
   sigma-inflation / synthetic-control value. The claim survives — 1.0031 is
   still excellent adequacy — but the printed number was the wrong quantity.
   Propagated from `notes/claim_audit.md`, which carried the same error;
   both corrected. *Lesson: the claim audit is treated as a source; it must
   itself be traced to files.*
2. **Misrounded interval.** §4 rung 4 printed the raw-MSE interval as
   `[−71.6, −42.6]`; the archived value (round-6 log, reproduced here from
   `E5_*_eval_series.csv`) is `[−71.52, −42.52]`, which rounds to
   `[−71.5, −42.5]`. Fixed.
3. **Missing hypothesis, rung 2.** `ŝ = PPᵀΣ⁻¹x` requires `PᵀΣ⁻¹P = I`
   (the appendix proof states it; the main text said only "a learned-subspace
   basis P" — false for a general basis, where the projector is
   `P(PᵀΣ⁻¹P)⁻¹PᵀΣ⁻¹`). Now "a Σ⁻¹-orthonormal basis P". Fixed.
4. **Missing hypothesis, Proposition 2.** The Kronecker-restriction argument
   works in whitened coordinates and needs whitening to preserve the two-axis
   structure, i.e. `Σ = Σ₂⊗Σ₁`. Both experiments satisfy it (oracle
   `Σ_t⊗Σ_c`; TIDMAD's frequency-diagonal `J⊗I`), but the §3 statement
   omitted it — as stated it was false for general Σ. Now "For axis-aligned
   noise, Σ = Σ₂⊗Σ₁, ...". Fixed. (The claim audit's row 9 knew this —
   "factor semantics require axis-preserving whitening" — but the qualifier
   had not reached the paper.)

## Open blockers for submission (external; not fixable at the desk)

1. **`E6_whitening_range.json` is not in the local tree.** The paper's 0.69
   and 441 trace only to the run-log prose. On sync, verify specifically that
   0.69 is the **excluding-DC** CV — the remote report's table was headed
   "includes DC", while the predeclared gate was on the excluding-DC value.
   If they differ, the paper quotes the wrong variant.
2. **E7 seed replication** — one of the two pre-compiled contingency texts
   must be applied; under Outcome 2 the abstract and Figure 1C caption change
   too (`notes/round8_contingency.md`).
3. **`priorThesis` verifiability** — "Unpublished; available on request" is
   weak under double-blind; deposit the acknowledgements-stripped report at a
   DOI or be ready to defend the citation as-is.

## Major findings (disclosed in-text; a referee will still raise them)

- **Rung 4 is n=1 per arm** and its bootstrap resamples autocorrelated points
  on one trajectory. Both stated in §4/§5; E7 addresses the first. This is
  the paper's weakest evidence attached to its most interesting claim.
- **"The architecture absorbs the weighting" is an interpretation**, not a
  measurement — correctly hedged ("we read the absence as"), and the
  degenerate-metric alternative was measured and excluded, but a referee may
  ask what would falsify the absorption reading. Defensible answer exists
  (seed variance — being tested; capacity sweep — future work).
- **Novelty is arrangement, not invention** — and the paper says so plainly
  ("we introduce no new estimator"). At a workshop this honesty plus the
  measured refutation of its own explanation is a strength; at the main
  conference it would be attacked. Prop 1 is classical (Rayleigh–Ritz after
  whitening); Prop 2 is a clean but modest observation in the MPCA family;
  the empirical reversal-with-controls and the cross-rung dispersion
  comparison are the distinctive contributions.
- **Both rung-4 arms sit ~6 orders below published baselines** (band-limited
  569k-param model, 25k steps). Disclosed in §4 and quantified in the
  appendix table. A referee may still ask whether the metric question is
  meaningful so far from competitive performance; §5's capacity caveat is
  the honest answer.

## Minor findings

| Location | Issue | Status |
|---|---|---|
| §5 "Four caveats" | first sentence packs three clauses, so the list reads as six | left — count is defensible (4 caveat *topics*) |
| Fig 1 panel A | χ² bar (−1.86%) nearly invisible next to +49.6% | known tradeoff; sign is read off the zero line by design |
| §4 GWOSC | "leave-one-out 1.77, 7.29" — H1's 1.77 is close to the synthetic 1.15, so "does not rescue" is strictly true only for L1 | wording already says "does not rescue *it*" (the pair); acceptable, advisor may prefer "only partially rescues H1" |
| Appendix B.1 caption | unrounded scores 0.0241/0.0231 quoted without the interval | interval is in §4; acceptable |

## Verified clean (machine-checked against archived files)

- Block A: −0.0192 [−0.0194,−0.0190]; +0.0034 [+0.0034,+0.0034]; NFPA−CW-PCA
  −0.0026 [−0.0026,−0.0025] — recomputed from `block_A_colored_raw.csv`,
  reproduce exactly (seed 20260808, 10⁴ resamples).
- Block B: 20 seeds, both paired differences exactly 0.0.
- Rung 4: −1.148e-4 [−1.269,−1.034]e-4; −57.18 [−71.52,−42.50]; last-ten sign
  agreement — recomputed from `E5_*_eval_series.csv`.
- Benchmark contrast: −1.018e-3, window [−3.33,+1.38]e-3, file-block
  [−4.61,+2.37]e-3, both spanning zero — from
  `E5_benchmark_contrast_interval.json`.
- GWOSC: 10⁻¹² equivalence (H1 1.6e-12, L1 1.4e-12), synthetic dispersion
  1.1476→1.15, H1/L1 9.424/8.827, leave-one-out 1.771/7.288 — from
  `transfer_paper/tables/gwosc_*.csv`.
- Rung-3 dispersion 1.0964→1.10, white 0.0000 — recomputed via
  `scripts/rung3_whitening_range.py`.
- Rung-2 appendix numbers (48.419/47.869, 86.25°, 0.1209→0.0880) — match
  `07_experiments.tex` of the archived Paper 1.
- Baselines 6.55/4.18/3.84/1.10: traced to the experiment checklist's A5 row
  (published TIDMAD values); `none = 1.1026` independently reproduced in
  round 1. −2.7708/−1.8025 archived.
- vec/Kronecker identity in §3 and the appendix proof's `W = Σ^{-1/2}P`
  reduction checked by hand; Corollary 1's hypotheses (isotropic whitened
  noise, eigengap, separable template) are stated and used.
- Build, template byte-identity, anonymity, metadata: all pass from clean.

## Already-known (not re-raised as new)

Single-seed rung 4 + E7 in progress; autocorrelation caveat; E6 JSON sync;
`priorThesis` decision; code-link gates 3/5/6 — all tracked in
`plan/ROUND8_DESK_REPORT.md` and `notes/claim_audit.md`.

## Answers to the author's questions

- **Ready for the advisor? Yes, now.** Send the current build; do not wait for
  E7 — advisor feedback and the GPU run can overlap, and §5 already frames the
  single-seed status honestly.
- **Ready for submission? Not yet** — three external items: E6 JSON (verify
  the 0.69 variant), E7 (apply the matching contingency), `priorThesis`
  source. All have prepared, pre-tested paths.
- **Are the experiments sound? Yes, for the claims as worded.** Predeclared
  protocol, paired contrasts at fixed architecture, white-noise control at
  exactly zero, matched budgets and step grids, archived records that
  reproduce. The two evidence weaknesses are disclosed, not hidden.
- **Match for R/PS? Strong.** Three of four themes addressed by name with
  evidence behind each; format and anonymity compliant.
- **Novel and solid?** Solid: yes, within its stated scope, and now
  numerically clean. Novel: honestly scoped — the arrangement, the lossless
  condition, the measured metric-reversal boundary, and a benchmark-anchored
  negative result with its obvious explanation tested and refuted. That is
  workshop-appropriate novelty, and the paper never claims more.

---

# Addendum — same day, after the second external audit

**The verdict above was too lenient and is corrected here.**

**Corrected verdict:** *Numerically traceable and promising, but a **major
revision** at the time of the original review: several mathematical statements
were incorrect, a predeclared semantic test was silently untested, and the
nonlinear evidence was framed with pseudo-inferential intervals. Appropriate
for advisor feedback only after the immediate mathematical corrections; not
submittable unchanged.* The revision was carried out the same day
(`ROUND9_CORRECTION_PLAN.md`); the post-correction state is below.

## What the first review missed, and why

1. **"When either factor is full rank the restriction is inactive" — false.**
   With `k₁=n₁` the admissible family is `ℝ^{n₁}⊗U₂`, a strict separable
   subset; the paper's own `C=k_c=2` appendix run (mean angle 61.7° to CW-PCA)
   is the counterexample. The first review verified the numbers around this
   sentence and the vec identity, but not this limiting claim. *Corrected in
   §3, with the appendix run now cited as the counterexample.*
2. **Rank-one CW-PCA conflated with the optimal filter** in §2's proof sketch —
   learned leading direction vs. fixed template; equal only under Corollary
   1's assumptions. *Corrected.*
3. **NFPA's displayed formulas acted on the raw measurement `Z`** while being
   valid for the per-axis whitened `Z̃` — the implementation was right, the
   notation wrong. *Corrected (`Z̃ = Σ₁^{-1/2} Z Σ₂^{-1/2}`, mapped back).*
4. **"Same factorization at different levels of nonlinearity" overclaimed** —
   no containment result exists. *Now "analogous axis-factored architecture
   (no containment ... claimed)", in §1 and §3.*
5. **"Closed-form" for rungs 1–3** — NFPA's joint problem is nonconvex. *Now
   "exact geometric characterization", with the nonconvexity stated in §1.*
6. **H3 (planted-subspace recovery) was predeclared and never run**, and the
   appendix claimed truth-referenced recovery was "covered by the rung-4
   amplitude error" — no such result exists. *Appendix B.2 now separates
   executed (H1, H2) from not-executed (H3); the false coverage sentence is
   deleted; E8 in `ROUND9_RUNBOOK.md` runs the existing `semantic.py`
   evaluation with a reproduce-the-archived-fit gate.*
7. **Rung-4 checkpoint bootstraps were pseudo-replication** despite the
   disclosed autocorrelation — the "intervals excluding zero" framing carried
   inferential weight one seed cannot support. *Rung 4 is now titled and
   framed exploratory throughout (abstract, §4, §5); Figure 1C shows bars
   without whiskers and the caption says why; "not distinguished by one seed
   per arm" replaces the absorption reading as sole account.*
8. **The white-noise control was presented as evidence** rather than the
   algebraic identity check it is. *Reframed in §4, the caption, and appendix
   H2.*

Also actioned from the audit: the stale 30-seed plan archived with a
supersession header; the 10-vs-30-seed history stated in the appendix;
Matrix Normal PCA cited; the population lemma
`Cov(Σ^{-1/2}x)=Cov(Σ^{-1/2}s)+I` added to §2 with proof in Appendix A —
the self-supervision claim now has its precise statement; workshop fit no
longer claimed as uniformly "strong" (self-supervision primary, tokenization
and transfer secondary); full-band baselines flagged as not directly
comparable to the band-limited arms in §5.

## Post-correction status

Build: 4 pages main content, refs p5, 0 overfull/undefined/PENDING, anonymity
clean, figure regenerated (panel A/B numbers unchanged; C bars unchanged, no
whiskers). **Ready for the advisor as an advanced draft with experiments
pending.** Submission additionally requires, beyond the Round 8 items
(E6 sync + variant check, E7 contingency, priorThesis): **E8** (the
now-declared untested H3), and ideally E9 (20-seed frozen replication) and
E10 (estimated covariance) — all specified with predeclared signatures in
`ROUND9_RUNBOOK.md`.

## Lesson for this review process

The first review verified every number against files and still passed four
false mathematical sentences, because it checked *quantities* harder than
*quantified statements*. Limiting-case claims ("when X is full rank...",
"at rank one this is...") need the same trace-to-source discipline as
numbers: each should be checked against a constructed example or the
paper's own experiments.
