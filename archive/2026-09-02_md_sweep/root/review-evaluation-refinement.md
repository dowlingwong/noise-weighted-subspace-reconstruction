# Review, Evaluation, and Refinement — Papers 1, 2, 3

**Date:** 2026-07-27 · *Revised 2026-07-28 — Paper 3 section rewritten against
the two-page `paper3_proposal.pdf`, confirmed by the author as the sole
Paper 3 proposal.*
**Reviewer stance:** dual — (a) experimental/statistical physicist assessing a
low-threshold-detector reconstruction paper for a physics-adjacent venue;
(b) principal ML scientist assessing the same material for methodological
soundness and contribution clarity.
**Materials read:** Paper 1 LaTeX sources (`00_abstract`–`08_discussion`,
theorem statements in `04_empca`, `05_linear_ae`), `OPEN_ISSUES.md`,
`ARXIV_CHECKLIST.md`; `paper2_nonlinear_reconstruction.pdf` (60 pp) and
`paper2/PAPER2_NOISE_INTEGRATION_AND_PAPER3_OUTLINE.md`; **`paper3_proposal.pdf`
(2 pp, 27 July 2026 — the operative Paper 3 document)**, with `PAPER3_PLAN.md`,
`COLLABORATOR_REVIEW_BRIEF.md`, `CLOSEST_WORK_COMPARISON.md`, and
`NUBENCH_FEASIBILITY_AUDIT.md` read as supporting material.

> **Revision note.** The first version of this review treated `PROPOSAL.md`
> ("ShiftScope"), `PAPER3_PLAN.md`, and the Junjie notes as three competing
> Paper 3s. That was wrong: `paper3_proposal.pdf` supersedes them and is the
> only proposal. The finding is withdrawn. The two-page PDF is a **materially
> better document** — de-branded, tighter, with an explicit continuity claim and
> a real technical link to Paper 1 that `PROPOSAL.md` lacked. Section 3 below is
> rewritten against it, and the residual criticism is narrower and, I think,
> more useful.

**Machine-checked:** Paper 1's Bridge Theorem (`05_linear_ae.tex:105`) was
verified numerically on random complex Gaussian data (d=8, k=3, N=4000): the
constraint `P†Σ⁻¹P = I`, the generalized-eigenproblem characterization
`R̂ₓΣ⁻¹P = PΛ`, the minimum value `Tr(Σ⁻¹R̂ₓ) − Σλⱼ`, and optimality against
300 random competitor subspaces all hold to numerical precision. **The central
theorem of Paper 1 is correct.**

**Items deliberately not re-raised** as new findings, because `OPEN_ISSUES.md`
and `ARXIV_CHECKLIST.md` already track them with a stated action: the TIDMAD
Phase 2 file-`0000` sync (#1), legacy supplementary provenance (#2), the
main-text length overrun (#3), partial corrective evidence (#4), and
CRESST/TIDMAD citation permissions (#5). Where this review touches them, it is
to propose a *decision*, not to report a discovery.

---

## Executive summary

| | Paper 1 | Paper 2 | Paper 3 |
|---|---|---|---|
| **State** | Substantively complete, one evidence blocker | Working draft, one validated case study | Well-formed 2-page proposal; sound design |
| **Theory** | Correct and verified | One near-definitional proposition + one open conjecture | None, and none required — it is an empirical design |
| **Evidence** | Unusually disciplined; four public datasets | Single seed, single setting, SNR≈914 | Feasibility only; V0 gate open |
| **Biggest risk** | Headline number measures rank, not covariance | The paper's one theorem is the one not proven | Papers 1–2 are cited, not used; C2's easy class inflates its headline |
| **Verdict** | Minor-to-moderate revision → arXiv; moderate → MLST | Major revision (restructure) | Approve to circulate, with three targeted amendments |

The three-paper arc has a real intellectual spine. Paper 3's proposal is
**good and should go to the postdoc and advisor essentially as written** — the
amendments in §3 are refinements, not a redirection. The one structural gap is
that Papers 1–2 currently supply Paper 3 with a testbed rather than a
prediction; §3.4 shows how one added axis fixes that cheaply and simultaneously
strengthens Paper 3's own headline claim.

---

# 1. Paper 1 — arXiv preprint / MLST submission

## 1.1 What is good

These are strengths worth protecting through revision; several are rare enough
that they are the paper's competitive advantage.

**The theory is correct, and it is the right theory.** The organizing claim —
that under a linear Gaussian model Σ⁻¹ *is* the reconstruction metric, and that
OF, EMPCA, and the tied noise-aware linear autoencoder are the rank-1, rank-k,
and reparameterized cases of one projection — is exactly the right level of
abstraction. It is not a repackaging: the Bridge Theorem's constraint
`P†Σ⁻¹P = I` and its generalized-eigenproblem characterization are a genuine
statement, and I verified all three of its conclusions numerically. The
separation stated at `01_intro.tex:35` ("the likelihood determines residual
geometry, the signal model determines admissible reconstructions, the algorithm
parameterizes the projection") is the sentence the whole field needed and
should survive into every future abstract.

**The evidence discipline is exceptional.** Predeclared gates; frozen
configurations tied to commit hashes and seeds; five negative controls per
real-data stage; a post-freeze sensitivity analysis explicitly labeled
post-freeze and explicitly *not* substituted for the primary gate; checksummed
evidence bundles. I review a lot of ML-for-physics work and this is the top
decile. The `07_experiments.tex:32` paragraph — stating in advance that the
OF/EMPCA contrast and the same-rank PCA/EMPCA contrast answer different
questions and that success on one is not evidence for the other — is the kind
of thing referees notice and reward.

**The four-dataset arc has an actual logic, not just accumulation.** Synthetic
(identities under a known Σ) → CRESST (positive within-channel real-detector
result, no truth) → GWOSC (calibration boundary, no truth) → TIDMAD (injected
reference channel, i.e. truth). The paper states explicitly at
`07_experiments.tex:226` *why* GWOSC's lack of truth forced the TIDMAD stage.
That is a research narrative, and it is more persuasive than a fourth positive
result would have been.

**The negatives are reported as negatives.** The GWOSC calibration failure, the
non-confirmation of the rank-2 prediction on all three TIDMAD files, and the
denoising score of −1.8025 against a no-denoising baseline of 1.10 are all in
the abstract. The Limitations section (`08_discussion.tex:107`) is the best-written
part of the manuscript. The identifiability paragraph at `08_discussion.tex:120`
— that eigendecomposition yields decorrelated statistical modes, not labeled
physical sources — is publishable language that Papers 2 and 3 should quote
verbatim rather than restate.

**Scope discipline.** "Within-channel, not cross-channel"; "reconstruction, not
detection"; "one interval, not GW sensitivity"; "one file, not the release."
Each is stated in the abstract, the intro, the relevant results subsection, and
Limitations. A referee looking for overclaim will not find one.

## 1.2 What is missing, incorrect, or insufficient

### Blocking (must fix before posting)

**B1 — The manuscript contains a literal placeholder.** `07_experiments.tex:331`
reads `[PLACEHOLDER --- pending evidence: ...]` in the primary TIDMAD Phase 2
paragraph. This is tracked as OPEN_ISSUES #1, so the finding is not new — but
the *decision* is still open and it is the only thing standing between this
draft and arXiv. You have two options and should pick one this week:

- *(a) Re-sync.* Recover the file-`0000` Phase 2 JSON under a distinct filename,
  insert the result, add the abstract sentence. Highest value **if** the result
  is what the evidence map claims (+0.0190, CI [0.0045, 0.0321]) — that CI
  excludes zero and would be your only real-data metric-reversal confirmation.
- *(b) Restructure around what you have.* Drop file `0000` from the primary
  slot; promote files `0005` and `0010` to the reported Phase 2 result; state
  plainly that both are directionally consistent and neither excludes zero.
  The paper survives this intact — §7.4's calibration result and the six
  identity gates are unaffected, and the abstract already makes no
  metric-reversal claim.

Option (b) is available *today* and costs you one paragraph of framing.
Option (a) is strictly better science if the file exists. Do not let (a) block
the preprint indefinitely; set a date, and fall back to (b).

### Major (a referee will raise these)

**M1 — The headline number measures signal-model rank, not noise geometry.**
This is the single most important scientific criticism of the paper, and the
paper already knows it — `07_experiments.tex:144` and `08_discussion.tex:31`
both say the OF contrast "mainly tests the signal-model hierarchy" and "should
not be attributed wholly to covariance weighting."

The problem is presentational, and it is serious. The abstract leads with
`1012.27` (CI [927.90, 1097.61]) — a rank-1-vs-rank-6 comparison. The quantity
that actually supports the title *Noise Covariance Defines Reconstruction
Geometry* is the same-rank PCA-vs-EMPCA contrast: fractional advantage 1.6%–62%
across channels, equal-channel mean 21.6%, unnormalized channel-balanced
difference `3.78`, CI [1.22, 6.69]. That number appears in the abstract only as
a bare interval `[1.22,6.69]` with no indication that it is the paper's actual
covariance-weighting evidence.

A hostile referee writes: *"The abstract's headline effect is 267× larger than
the effect the title claims. The authors concede in the discussion that the
headline effect is dominated by rank, not weighting."* That reads badly even
though the paper is being honest.

→ **Rewrite the abstract to lead with the same-rank contrast.** Something like:
"Holding subspace rank fixed at six, covariance weighting reduced the
likelihood-aligned held-out residual in all seven CRESST channels
(equal-channel mean reduction 21.6%, range 1.6%–62%; channel-balanced
difference 3.78, hierarchical 95% CI [1.22, 6.69]). Enlarging the signal model
from a fixed template to a learned rank-6 subspace produced a much larger but
differently-attributable gain (1012.27, CI [927.90, 1097.61])." This costs
nothing in honesty, is a strictly more accurate summary, and inoculates against
the referee's best attack. **Highest-value single edit in the paper.**

**M2 — Unnormalized weighted-residual differences are reported as headline
quantities.** `1012.27`, `3.78`, `489.40`, `59812.72` have no interpretable
scale. They are sums of χ²-like quantities whose per-channel magnitudes differ
by orders of magnitude — the paper says so itself when it introduces the
logarithmic axis in Fig. 2A and when it notes "these scale-dependent values are
reported to verify direction... not to compare scientific severity"
(`07_experiments.tex:163`). But that caveat cannot rescue a number in the
abstract.

→ Report a **standardized effect** as primary throughout: fractional reduction,
or residual per degree of freedom, or a paired standardized mean difference.
Keep the raw differences in a table. Specifically: `1-R_EMPCA/R_OF = 96.2%` and
`1-R_EMPCA/R_PCA = 21.6%` are interpretable; `1012.27` and `3.78` are not.
The paper already computes both — it just leads with the wrong one.

**M3 — There is no Related Work section.** Confirmed: the main text has eleven
`\section` blocks and none of them is related work; the literature discussion
lives in three paragraphs of the introduction (`01_intro.tex:12–25`). For arXiv
this is survivable. For MLST it is a likely referee complaint, because the paper
makes a unification claim and unification claims are judged against prior
unifications.

→ Add a compact §Related Work (0.75–1 page) covering: weighted/probabilistic
PCA lineage (Tipping–Bishop, Roweis, Bailey); linear-autoencoder/PCA
equivalence (Baldi–Hornik, Bourlard–Kamp, Plaut); GLS/Mahalanobis estimation
and matched filtering; noise-aware deep denoisers in physics (the
Noise2Noise/SURE family, and the Xenon-TPC work Paper 2 already cites); and
explicitly, the *closest* prior claims to yours so the delta is visible. State
the delta in one sentence: prior work establishes each equivalence separately
under Euclidean geometry; this paper places them in one Σ⁻¹-metric hierarchy
with the finite-sample degeneracy and the tied-encoder constraint made exact.

**M4 — Negative-control saturation is a design flaw, not just a caveat.** On
TIDMAD, the random-weighted-basis control *improved* the metric
(−0.0062, CI entirely negative) because it collapses to the zero-recovery level
whose RMSE (0.0747) is below the nominal (0.0880) (`07_experiments.tex:397`).
The paper reports this honestly and calls the control "saturated rather than
informative." Correct — but the consequence is that two of five TIDMAD controls
carry no information, and the control set therefore cannot demonstrate what it
was designed to demonstrate.

→ Fix the control, don't just annotate it. A norm-matched random basis (random
subspace constrained to the same whitened energy as the fitted basis) does not
collapse to zero-recovery and would be informative. This is a cheap rerun and
converts a caveat into a result.

**M5 — Multiplicity is never discussed.** Nine synthetic gates, six TIDMAD
identity gates, five CRESST controls, five TIDMAD controls, two GWOSC check
families, multiple bootstrap intervals. Predeclaration largely defends this, but
the paper never says so.

→ One paragraph in Evaluation conventions: state that gates were predeclared
and are reported exhaustively (no selection), that intervals are nominal and
not family-wise corrected, and that the paper's conclusions rest on the two
primary predeclared contrasts rather than on the count of passed gates. This is
three sentences and removes a whole line of attack.

**M6 — Covariance estimation error is never propagated into the theory.** The
theorems assume Σ known or "independently estimated." S7 empirically varies
calibration size, and Limitations notes covariance drift can invalidate the
uncertainty interpretation. But there is no statement connecting `‖Σ̂ − Σ‖` to
subspace error — which is exactly what Davis–Kahan gives you, and which Paper 2
is already reaching for.

→ Add a short Remark after the Bridge Theorem: under a perturbation `Σ̂ = Σ + E`
with spectral gap Δλ, `sinΘ(Ŝ, S) ≲ ‖E‖/Δλ`. Half a page, standard, and it
(i) closes an obvious referee question, (ii) makes S7 a test of a stated bound
rather than a descriptive sweep, and (iii) is the exact lemma Paper 2 needs, so
you pay for it once and use it twice.

### Minor

| Location | Issue | Fix |
|---|---|---|
| `00_abstract.tex:15` | "reduced ... by $1012.27$" — unit-free magnitude in first-read position | see M2 |
| `00_abstract.tex:8` | "Nine controlled tests supported..." — past tense mixed with present elsewhere | unify tense |
| `07_experiments.tex:66` | "$48.418$ is a residual value, not a fold degradation" — defensive clarification implies earlier confusion | move to caption or delete; the units should make it unnecessary |
| `07_experiments.tex:249` | Table 1 mixes implementation checks and calibration checks in one block | horizontal rule + subheadings; the distinction is the point of the table |
| `07_experiments.tex:317` | "direction only" as a Status value | use "consistent, not significant" — "direction only" reads as a hedge |
| `08_discussion.tex:229` | `\enlargethispage` before Conclusion | symptom of the length problem; remove once compressed |
| Throughout | `≈`, "approximately −0.005" for a CI-bearing quantity | give the value at stated precision |
| `main.tex:1–11` | iopart migration notes as a comment block | move to a `SUBMISSION.md`; comments in `main.tex` rot |

## 1.3 Advice toward preprint-ready

**Do this, in order:**

1. **Resolve B1** (set a hard date; fall back to option (b)).
2. **Rewrite the abstract per M1 + M2.** Half a day. Largest return on effort in
   the entire project.
3. **Add §Related Work (M3)** and the covariance-perturbation remark (M6).
4. **Rerun the norm-matched control (M4)**; add the multiplicity paragraph (M5).
5. **Then** post to arXiv. Do not wait for the length compression.

**On the arXiv/MLST split.** These are different documents and trying to make
one file serve both is what produced the 52-page/12,300-word state that
OPEN_ISSUES #3 tracks.

- *arXiv v1:* post the extended version essentially as-is once B1–M6 are done.
  Length is not a defect on arXiv; the appendices are an asset there.
- *MLST submission:* build a separate compressed branch. The compression is
  mechanical and the plan already exists (the July 12 instruction's §2–§6 →
  supplement move). Concretely: §2–§6 collapse to one "Setup and results"
  section of ~4 pages carrying only the objective, the whitening normal form,
  the two theorem statements, and the MSE-misspecification consequence, with
  every derivation in the supplement. §7.3 (GWOSC) compresses to ~1 page — it
  currently spends 65 lines to conclude "implementation correctness ≠
  calibration," which is one paragraph plus a table. That alone recovers ~4 pp.

**One framing suggestion for MLST specifically.** The version of this paper that
an ML audience will find most valuable is not "OF = EMPCA = tied linear AE" —
they will read that as linear algebra. It is **"your loss function encodes a
noise model, and if that model is wrong your benchmark ranking inverts."** The
S5 result (raw MSE prefers PCA 2367.99 vs 5388.75, weighted residual prefers
EMPCA 47.878 vs 48.418, whitened principal angle 86.07°) is a clean, portable,
alarming demonstration of that. Lead the MLST version with it.

---

# 2. Paper 2 — enhancement, NFPA structure, storyline, verification

## 2.1 Assessment

Paper 2 asks the right next question — once Σ⁻¹ fixes the metric, which
architecture priors are justified and under what coverage are directions
identifiable — and the tuple `(Σ⁻¹, F, P_train→P_target)` is a good organizing
device. But the draft has a **contribution-type problem** and a **theory
problem**, and they compound.

**Contribution-type problem.** The paper currently contains five different
kinds of thing: (i) a framework/taxonomy; (ii) NFPA, a new estimator with two
propositions; (iii) a reproduced external case study (Xenon TPC, Parker et al.
2026) containing no new computation; (iv) two "framework-derived extensions"
(noise-aware VAE, noise-aware Transformer) explicitly not validated; (v) an
experimental program for other people. Reviewers read this as a survey with a
result stapled in the middle. The paper's own §Scope concedes "only the NFPA
contribution is experimentally validated, and only in one controlled synthetic
setting."

**Theory problem — and this is the crux.** The paper's one architectural
contribution rests on:

- **Proposition 4.1 (restricted bridge):** NFPA solves the EMPCA objective
  restricted to Kronecker-separable subspaces. As stated this is close to
  definitional — the constrained minimizer over a subset minimizes over that
  subset. Its content is the *equality condition* (equals EMPCA iff an EMPCA
  minimizer is separable), which is also nearly immediate.
- **Theorem 4.2 (rank-one NFPA = OF under separable whitened template):**
  genuine, but narrow.
- **The approximation bound** `sinΘ(S_NFPA, S_EMPCA) ≤ F(ε_kron, Δλ)` — stated
  at p.24 and again in §10.2 as an **open problem, not a result**.

So the theorem that would make NFPA a contribution is the one not proven. And
the experiment that would make it an empirical contribution reports a mean
principal angle of **61.7°** (Table 4) — which the draft correctly reads as
"the Kronecker constraint is active," i.e. NFPA does *not* recover the EMPCA
subspace. Meanwhile Table 3's whitened-χ² margins are ~0.1% on a single seed
with raw-MSE entries printed as `<10⁻⁵`.

As it stands, a reviewer's summary is: *"A near-definitional proposition, a
narrow corollary, a conjectured bound, and a single-seed experiment showing the
proposed constraint costs 61.7° of subspace fidelity."* That is fixable, and
the fix makes the paper considerably stronger than the current draft aims to be.

## 2.2 How NFPA should be structured — in detail

### (a) Present NFPA as a constrained estimator, not as an architecture

Paper 1 succeeded because it stated an *estimator* (a Σ⁻¹-orthogonal projection)
and then showed several parameterizations realize it. Paper 2 currently states
an *architecture* (a Kronecker-constrained tied linear AE) and derives its
properties. Invert this.

Define, in whitened coordinates:

> **NFPA estimator.** Given whitened data `x̃`, minimize the empirical whitened
> residual over projectors of the form `P = P_c ⊗ P_t`, with
> `P_c ∈ Gr(k_c, ℂ^C)`, `P_t ∈ Gr(k_t, ℂ^T)`.

Then: the ALS autoencoder is *one algorithm* for this estimator; the tied linear
AE of Paper 1 is the unconstrained case; Theorem 4.2 is the `(k_c,k_t)=(1,1)`
case. Every theorem becomes a statement about the estimator, and the
architecture stops carrying theoretical weight it cannot bear.

### (b) Separate the two independent separability assumptions — this is the gap

**This is the most important technical observation in this review of Paper 2.**

The Kronecker constraint is imposed in *whitened* coordinates. But whether that
constraint is natural depends on Σ, and the draft conflates two distinct
assumptions:

1. **Noise separability:** `Σ = Σ_c ⊗ Σ_t`. If this holds, then
   `Σ^{-1/2} = Σ_c^{-1/2} ⊗ Σ_t^{-1/2}`, and whitening *preserves*
   separability — a separable signal subspace stays separable after whitening.
2. **Signal separability:** the (whitened) signal subspace is itself Kronecker
   separable.

If Σ is *not* separable, whitening maps a separable signal subspace to a
**non**-separable one. So NFPA's prior is only self-consistent in the corner
where both hold. The current draft's §5.5.5 sweep perturbs the *template*
(`s = vtᵀ + αR`), i.e. it varies assumption 2 while silently holding
assumption 1 — and the Brownian single-channel-per-channel noise in §5.1 makes
assumption 1 true by construction, so the experiment cannot see the interaction
at all.

→ **Restructure the theory and the experiments around a 2×2:**

| | Σ separable | Σ non-separable |
|---|---|---|
| **Signal subspace separable** | Exact restricted bridge; NFPA = EMPCA when the EMPCA minimizer is separable | Approximation error from Σ alone |
| **Signal subspace non-separable** | Approximation error from the signal alone | Both; the realistic detector case |

Only the top-left cell admits the clean statement. The other three are where the
approximation bound is needed — and now the bound has *two* arguments, which is
both more honest and more interesting.

### (c) Prove the approximation bound — it is provable, and it is the paper

Make the conjecture a theorem. The route:

1. **Quantify noise non-separability with a computable number.** The nearest
   Kronecker product problem `min ‖Σ − A ⊗ B‖_F` is solved exactly by the
   Van Loan–Pitsianis rearrangement: reshape Σ into `R(Σ)` and take its top
   singular pair. Define `ε_Σ := ‖Σ − Σ_c ⊗ Σ_t‖_F / ‖Σ‖_F` = the normalized
   trailing singular mass of `R(Σ)`. **This is computable in closed form for
   any Σ**, which turns "how separable is my detector's noise?" from a vague
   question into a measurable one — a genuinely useful deliverable in its own
   right.
2. **Quantify signal non-separability** by `ε_kron` := the best Tucker-2
   approximation error to the whitened EMPCA projector (already in the draft).
3. **Compose.** `ε_Σ` perturbs the whitening map, hence the whitened covariance,
   hence its eigenvectors — Davis–Kahan gives `sinΘ ≲ c·ε_Σ‖Σ‖/Δλ`. `ε_kron`
   contributes the Tucker truncation term. Combine to
   `sinΘ(S_NFPA, S_EMPCA) ≤ f(ε_kron, ε_Σ, Δλ)`.

This is a real theorem, it is within reach with standard tools (Kolda–Bader for
the Tucker bound, Davis–Kahan for the perturbation, Van Loan–Pitsianis for the
rearrangement), and it converts §5.5.5 from "a vague cost-of-the-prior plot"
into *empirical verification of a stated bound*. It also gives Paper 2 the thing
Paper 1 has and Paper 2 lacks: a prediction that could fail.

### (d) State the identifiability/gauge structure as a proposition

`P_c ⊗ P_t` is invariant under `O(k_c) × O(k_t)` acting within each factor and
under the scaling `(αU_c, α⁻¹U_t)`. §4.6 discusses gauge freedom; promote it to
a numbered proposition, because it determines what may be *reported*: individual
factor loadings are not identified, but the factor *subspaces* and the product
projector are. This is the Paper 2 analogue of Paper 1's identifiability
paragraph, and it protects you from the "learned channel basis is physically
interpretable" trap that Question E currently walks toward.

### (e) State what ALS does and does not guarantee

ALS on `Gr(k_c,ℂ^C) × Gr(k_t,ℂ^T)`: monotone descent, convergence to a
stationary point, **no global optimality**. HOSVD initialization gives the
standard quasi-optimality factor. Two sentences, one citation; omitting it is a
gift to a referee.

### (f) Say when NFPA should be *used* — the missing headline claim

The draft never answers "why would I use NFPA instead of EMPCA?" The answer is
**sample complexity**, and it should be the paper's main claim.

- EMPCA at rank k over a `C×T` domain estimates a subspace with `O(k·CT)`
  free parameters.
- NFPA with `k = k_c·k_t` estimates `O(k_c·C + k_t·T)`.

For `C=100, T=1024, k=6`: EMPCA ≈ 6·102,400 ≈ 6.1×10⁵ parameters; NFPA
(`k_c=2, k_t=3`) = 3,272. A factor of 188 — over two orders of
magnitude. **The Kronecker prior is a bias–variance instrument: it should beat
unrestricted EMPCA in held-out weighted residual when N is small relative to
CT, and lose when N is large.** That is a crossover with a predictable location,
it is exactly the kind of claim Paper 1's methodology knows how to test, and it
converts the 61.7° principal angle from an embarrassment into the *mechanism*:
NFPA is supposed to give up subspace fidelity in exchange for estimation
variance.

Note that the current experiment (`N_train=800`, `C=2`, `T=256`, `k=6`, so
`k·CT = 3072 > 800`) is already in the regime where NFPA should win — and Table
3 shows NFPA ahead of EMPCA on whitened χ² (0.9916 vs 0.9929), by an amount the
single seed cannot resolve. **The result you need is already hiding in your
existing experiment; it is under-powered, not absent.**

### (g) Proposed section structure

1. Introduction — the architecture question, one paragraph inheriting Paper 1.
2. Setup — whitened objective, the Grassmannian, notation. (~2 pp, compressed
   from current §2–§3.)
3. **NFPA as a constrained estimator** — definition, the 2×2 separability
   table, gauge proposition, ALS + guarantees.
4. **Approximation theory** — `ε_Σ` via Van Loan–Pitsianis, `ε_kron` via
   Tucker-2, the composed bound, the exact-equality corollary, rank-one/OF
   corollary.
5. **Sample-complexity claim** — parameter counts, the predicted crossover.
6. **Experiments** — the grid in §2.4 below, ≥20 seeds throughout.
7. Framework positioning — the tuple, the taxonomy table, coverage condition.
   *Demoted to ~2 pp*; it is context, not contribution.
8. Related work, discussion, limitations.

**Cut or demote:** the Xenon-TPC case study (§7) to a discussion paragraph — it
reproduces Table 6 from Parker et al. and contains no computation of yours;
"illustrates the framework" is not a contribution a reviewer credits. The VAE
and Transformer appendices to a single "extensions" paragraph plus a pointer.
The §8 experimental program to a short forward-reference to Paper 3.

## 2.3 Making the storyline coherent

The current spine is *"here is a framework; here is one small validated example;
here are some things one might do."* The proposed spine is a claim with a
mechanism and a test:

> Fixing the likelihood metric does not fix the estimator. Restricting the
> admissible projector class trades subspace fidelity for estimation variance.
> We give the exact restricted-bridge condition, a computable bound on the
> fidelity cost in terms of two separately measurable non-separability
> quantities (`ε_Σ`, `ε_kron`), and show the resulting bias–variance crossover
> in N.

Three properties make this work as a story: it *inherits* from Paper 1 rather
than recapping it (one paragraph, not seven pages); it has a claim that could be
false (the crossover, the bound); and its negative result (61.7°) becomes
evidence for the mechanism rather than an apology.

Two narrative rules for the rewrite:

- **One paragraph of Paper 1 recap, not a section.** Current §2 spends ~8 pages
  re-deriving Paper 1. Cite it.
- **Every framework claim must be cashed out in §3–6 or cut.** The tuple earns
  its place only if `F` and `P_train→P_target` each do work in the NFPA
  analysis. `F` does. Coverage currently does not — so coverage becomes Paper 3's
  material, not a §2.6 promissory note.

## 2.4 Experimental verification — the grid

Every cell: exact Σ from the noise module, ≥20 seeds, held-out evaluation,
bootstrap CIs, and a predeclared primary metric (**held-out whitened χ²**).
Report `N_train, N_test, C, T, noise model, SNR definition, whitening method,
(k_c,k_t), k, seeds, CI type` per cell, as §5.5 already specifies.

| # | Sweep | Axis | Tests | Priority |
|---|---|---|---|---|
| **E1** | **Sample complexity** | `N` over ~2 decades at fixed `C,T,k` | The bias–variance crossover of §2.2(f). **Primary result.** | **Highest** |
| **E2** | **Noise non-separability** | `ε_Σ` via `Σ_α = (1−α)Σ_sep + αΣ_nonsep`, reporting *realized* `ε_Σ` from the Van Loan–Pitsianis rearrangement, not nominal α | The Σ-arm of the bound | **Highest** |
| **E3** | **Signal non-separability** | realized `ε_kron` (Tucker-2 error), not nominal α | The signal arm of the bound | High |
| E4 | Rank pair | `(k_c,k_t)` at matched `k=k_c k_t` vs unrestricted EMPCA | Separates rank effect from factorization effect | High |
| E5 | Channel count | `C = 2,4,8,16,(64)` | Where the parameter advantage grows; pairs with E1 | High |
| E6 | SNR | low/med/high (current 914 is degenerate) | Whether reversal and rank-1/OF correspondence survive nontrivial amplitude recovery | Medium |
| E7 | Noise color | white / pink / Brownian / measured PSD | Reversal appears under color, vanishes under white | Medium |

E2 and E3 are the two arms of the bound and should be plotted *against the
realized ε*, with the theoretical bound overlaid. That figure is the paper.

**Immediate fixes to existing results, independent of new runs:**

- Re-export Table 3 at full precision. `<10⁻⁵` is not publishable.
- Re-run Table 3 over ≥20 seeds and report mean ± CI. The 0.1% EMPCA–NFPA gap
  either becomes a result (supporting E1's mechanism) or is honestly reported as
  unresolved. Right now it is neither.
- Add the realized `ε_kron` **and** `ε_Σ` to the Table 4 caption, so the 61.7°
  is attributable rather than merely observed.
- Split Figure 1 into three publication figures (metric reversal; principal
  angles; rank-one/OF), panel grid to appendix — as the draft's own caption asks.
- Fix the `r = 1.0000` amplitude correlations: at SNR ≈ 914 these are
  uninformative and reporting four nines invites the reader to think something
  was verified that was not.

---

# 3. Paper 3 — reevaluation against `paper3_proposal.pdf`

*Controlled Testing and Failure Attribution for ML Sensor Pipelines*, two-page
proposal, 27 July 2026.

## 3.1 Assessment: this is a good proposal

Read on its own terms — a two-page ask for postdoc and advisor review — this
document does its job well, and it is a substantial improvement on the
`PROPOSAL.md` draft it supersedes. The improvements are worth naming, because
they should be protected in any further revision:

**It stopped being a product.** "A controlled, replay-based study of that
distinction — not a maintained software product"; "its artifact is a
reproducible experiment suite … not a branded platform or a promise of
production maintenance." Dropping the ShiftScope branding removes the single
biggest reviewer objection to the earlier draft, which read as a tool pitch
wearing a paper's clothes. The artifact framing that replaced it —
configurations, adapters, hooks, replay scripts, reports — is what a reviewer
will actually credit.

**The failure taxonomy is now clean and, importantly, non-overlapping by
construction.** N (acquisition/noise shift), S (training-support shift, "events
themselves are not corrupted"), E (evaluation-contract fault). The S definition
is a genuine improvement over the earlier "coverage/support shift": specifying
that the events are *clean* and only the window composition moves makes S a
sharp, injectable intervention rather than a vague domain-shift notion. And the
explicit demarcation — "E is not a distribution shift and is not the
likelihood-metric mismatch studied in Papers 1–2" — is the kind of sentence
that prevents a reviewer from misreading the contribution.

**The baseline design is the strongest part of the proposal.** Five arms fitted
with identical splits — input only; output/uncertainty; final embedding only;
*all generic-monitor outputs*; full layerwise signature — with identical windows
and false-alert budgets. The fourth arm is the honest strong baseline, and most
papers in this space omit it. Committing to it in advance is a credibility
asset and should be foregrounded, not buried in §3.

**The claims are falsifiable and numerically specific,** the feasibility gate is
reported as *open* rather than quietly passed, and the pilot is explicitly
labeled "without presenting the pilot as a scientific claim." The discipline
that makes Paper 1 credible is visibly carried forward.

**The waveform arm now uses your own estimator.** "A rank-2 noise-aware EMPCA
encoder fitted on the chronological training split, and sinusoid-amplitude
reconstruction" is the Paper 1 TIDMAD operating point. `PROPOSAL.md` had no such
link. This is real continuity at the level of materials.

**Verdict: circulate it.** The three amendments below are refinements to a sound
design, not a redirection. My earlier recommendation to reframe the whole paper
around noise-model misspecification is **withdrawn** — it was written against
`PROPOSAL.md`, and it would replace a well-scoped, executable study with a
harder one you have not scoped. §3.4 keeps the proposal intact and adds one
axis instead.

## 3.2 Is the storyline coherent with Papers 1 and 2?

**Better than before, and now the question has a precise answer: Papers 1–2
supply Paper 3 with a testbed, not with a prediction.**

The continuity sentence — "Paper 1 identifies likelihood geometry under
structured noise; Paper 2 separates residual geometry, architecture class, and
training support; Paper 3 asks whether those contracts can be tested after
training" — is well phrased, and the N/S/E taxonomy does mirror the
(Σ⁻¹, F, P_train→P_target) tuple loosely: N moves the noise, S moves the training
support, E breaks the evaluation contract.

But test the inheritance the way a reviewer will: **could Paper 3 be executed
unchanged if Papers 1 and 2 did not exist?** Substantially, yes. C1 is detection
power at a false-alert budget. C2 is attribution macro-F1 against feature-matched
baselines. C3 is a latency/storage/power frontier. None of the three references
Σ⁻¹, the Bridge Theorem, the Kronecker restriction, or any quantity Papers 1–2
compute. Swap the rank-2 EMPCA encoder for a small CNN and every claim, endpoint,
and threshold stands exactly as written.

So the relationship is *co-citation plus shared materials* rather than
dependence. That is a legitimate way to run a research program — but it means
the three papers are a portfolio rather than an arc, and the proposal's opening
paragraph currently promises an arc.

There is also a specific tension worth noticing. N is defined to include "PSD or
covariance change" — i.e. **N is exactly Σ-shift**. Papers 1–2 are the theory of
what Σ-shift *costs an estimator*. Paper 3 asks whether a monitor can *see* it.
Those are complementary questions about the same object, and the proposal
currently leaves the connection unstated. §3.4 is about closing that gap with
one added axis rather than a rewrite.

## 3.3 What a reviewer will push on

**R1 — C2's margin condition is likely to fail even if the science works, and
E inflates the headline.** Two coupled problems in one claim.

*The margin.* C2 predeclares macro-F1 ≥ 0.80 **and** a ≥ 0.20 F1 advantage of
the full layerwise signature over the "all generic-monitor outputs" arm. Those
can conflict. If the generic arm reaches 0.72 — plausible, since it aggregates
six monitors — you need 0.92 to clear the margin, well above your own absolute
bar. You have predeclared a condition under which a real, publishable effect
(0.72 → 0.86) is scored as a failure.

*The easy class.* E is caught by "schema/range invariants and clean-reference
tests, not drift statistics." Wrong units, swapped coordinate ordering, and
changed PSD conventions produce gross deterministic signatures. Including E in a
three-class macro-F1 means one class is near-perfect for both arms, which
simultaneously **inflates the absolute score** and **compresses the achievable
margin** — the generic arm also gets E right, so the 0.20 gap must be earned
entirely on N and S.

→ **Amendment 1.** Report macro-F1 over {N, S} as the primary attribution
endpoint, with E-gate sensitivity/specificity reported separately (it is a gate
result, not a classification result — E0 already treats it that way). Restate the
margin as a superiority test with a confidence interval on the paired difference
rather than a fixed 0.20 point threshold, or lower the threshold to ~0.10 on the
{N, S} endpoint. This makes the claim harder to game *and* easier to satisfy
honestly, which is the correct direction for both.

**R2 — The hard discrimination is N vs S, and the proposal does not say so.**
Module dropout and channel loss are listed under N; S is "window composition
toward physical strata that are sparse under the clean training distribution."
But losing modules *changes the effective support*, and a low-density-stratum
event and a module-dropped event can present nearly identically at the
representation and output layers. That is the genuinely difficult and
genuinely interesting case, and it is where the layerwise signature should beat
output-only monitoring — input-quality summaries see hit loss directly, whereas
S is invisible at the input by construction.

→ **Amendment 2.** Foreground N-vs-S as the discriminative core of C2, and state
the mechanism you expect to exploit (input-layer evidence separates N; S is
input-invisible and must be detected from representation position relative to
the training support). Add the N+S mixture to the primary endpoint rather than
leaving it as a secondary multi-label score. If the layerwise signature cannot
separate N from S, C2 has no interesting content regardless of what E does — so
this is where the paper's risk actually lives, and saying so is a strength.

**R3 — Modality asymmetry.** The point-cloud arm has three model families
(DynEdge, ParticleNeT, GRIT) and multiple geometries. The waveform arm has one
task and one linear estimator (rank-2 EMPCA, sinusoid amplitude). C3's
reproducibility claim spans "modalities, three NuBench model families, two
machines, and CPU/GPU," which reads as symmetric coverage but is not. A rank-2
linear encoder also has an unusually simple representation geometry — its
"layerwise signature" is essentially one 2-D projection — so a layerwise
advantage found on DynEdge may have nowhere to come from on the waveform side.

→ Either add a second waveform model (the NFPA/Kronecker encoder from Paper 2 is
sitting right there and costs nothing new to fit), or state plainly that the
waveform arm is a *transfer check on a deliberately minimal system under test*,
not a co-equal modality. The second option is free and honest; the first is
better and also gives Paper 2 an external citation.

**R4 — V0 remains open, and the discrepancy pattern is diagnosable.** Unchanged
from the previous review, and still the top execution risk. Both tasks are high
relative to published: 23.6548° vs 23.60° (+0.055°) and 56.1553° vs 56.12°
(+0.035°). Matched sign and comparable magnitude on two independent populations
is not what backend or floating-point variation produces; it is what an
**event-selection or metric-definition difference** produces (a quality cut, a
different denominator, or median-of-medians vs pooled median). Rescoring
*released predictions* should be deterministic, which makes a selection
explanation more likely than a numerical one. Test it directly: re-derive the
published medians under three or four plausible selections and see which
reproduces 23.60/56.12 exactly. Separately, note that 0.01° is 0.042% of a
23.65° median — a self-imposed and very strict tolerance. If the selection
hypothesis fails, negotiate a justified tolerance and document the negotiation;
leaving the gate open indefinitely blocks E1–E4.

**R5 — Smaller points.**

- The abstract's "distinguish a sensor/acquisition shift from unsupported
  physical states or a broken evaluation contract" is pitched wider than C2's
  own fine print ("intervention-family attribution, not unrestricted physical
  root-cause discovery"). Align the abstract to the caveat; the caveat is the
  defensible claim.
- C1's "80% within five windows" needs an N. The proposal defers this to
  "development profiling and power analysis" — acceptable for a proposal, but
  the advisor will ask, so have the power calculation ready.
- Figure 1(b)–(c) are labeled "project outputs" while 1(a) is reproduced from
  NuBench Fig. 3. Good practice; keep the attribution explicit in the final
  manuscript.
- "Delay is primary in events and simulated seconds" — define the conversion,
  since the two testbeds have very different event rates and a delay claim that
  is not comparable across modalities cannot support C3's cross-modal
  reproducibility statement.

## 3.4 The one amendment that makes Papers 1–2 load-bearing

This is the highest-value change and it is additive — it does not touch the
existing claim structure.

**The observation.** Paper 3 measures whether a shift is *detectable*. Papers
1–2 predict what a shift *costs*. These are different quantities, and the gap
between them is the most interesting thing this trio of papers can say.

For the waveform arm you can compute, for any injected N intervention, the
Paper 1 quantity that determines consequence: the mismatch between the assumed
Σ̂ and the realized Σ, and the resulting degradation in the Σ⁻¹ geometry
(subspace angle, amplitude bias, calibration ratio). The noise module already
constructs both covariances exactly, so this is bookkeeping you are almost
doing already — you know the true Σ because you generated it.

**Amendment 3 — add a consequence axis to E1/E2.** For each N intervention,
record both:

- *detectability* — does the monitor fire, at what delay (already planned);
- *consequence* — how much does the reconstruction actually degrade in the
  likelihood geometry (new, cheap, waveform arm only).

This yields a 2×2 that no generic drift-monitoring paper can produce:

| | Low consequence | High consequence |
|---|---|---|
| **Monitor fires** | **False alarm in the sense that matters** — the operational cost of monitoring | True positive |
| **Monitor silent** | Correctly ignored | **Silent failure** — the dangerous cell |

Why this is worth doing:

1. **It makes Papers 1–2 necessary rather than cited.** The consequence axis is
   computable *only* because Paper 1 established what the correct residual
   geometry is. Remove Paper 1 and this table cannot be built.
2. **It strengthens Paper 3's own headline.** "Generic monitors detect N/S
   shifts; layerwise evidence improves attribution" is a solid but incremental
   result. "Detection and consequence are only partially aligned; monitors fire
   on shifts that do not degrade the estimate and stay silent on some that do"
   is a finding an ML-for-science audience will actually cite, and it makes the
   power–cost frontier of C3 interpretable — you can price a false alert.
3. **It is the natural home for Junjie's systematics interest** (see §3.5).
4. **It costs one column in an existing results table**, not a new experiment.

Optionally promote it to a fourth claim: *C4 — detection power and reconstruction
consequence are not monotonically related; the rank correlation between monitor
response and Σ⁻¹-geometry degradation is below a predeclared threshold.* Include
it only if you are prepared to power it; as an axis on E1/E2 it already earns its
place.

Note the scoping constraint: this applies to the waveform arm, where you control
Σ and own the estimator. On NuBench there is no analogous computable
consequence quantity beyond downstream angular error — which is fine, and worth
saying explicitly, since it is a clean statement of what the two testbeds can and
cannot each support.

## 3.5 Junjie alignment

The two-page proposal aligns with Junjie's discussion list **partially and
asymmetrically**, and Amendment 3 is what closes most of the gap.

| Junjie's item | Status in `paper3_proposal.pdf` |
|---|---|
| Embedding separability; clustering robustness; perturbation stability | **Covered.** These are the layerwise signature and the N/S severity response. |
| Uncertainty / score calibration | **Covered.** Calibration is a recorded window signal; the Guo et al. citation and the "clean miscalibration is a model result, not a bug" stance are correct. |
| Artificial noise injection probe, calibration verification | **Covered** — this is precisely the declared-intervention design. |
| "论文一找找 noise decomposition, decorrelation; linear 能做到哪步" | **Absent.** The identifiability material (`PAPER3_PLAN.md` §8, Paper 1 `08_discussion.tex:120`) does not appear in the two-page proposal. It is the most defensible writing in the whole package and it directly answers a question Junjie asked. At minimum, one sentence in §1 bounding what the noise module can and cannot claim about sources; in the full manuscript, a subsection. |
| Fitted correlated-noise categorization (ask IMS) | **Absent.** A parameterized Σ̂ family (SQUID floor, Johnson, 1/f, lines) validated on held-out predictive likelihood is exactly the "added structure" that makes decomposition well-posed — and it would supply realistic N interventions instead of analytic ones. Worth one line as a collaborator ask. |
| Performance near interaction vertices | **Partially covered** — vertex/containment appears as an S stratum, which is the correct reframing (boundary- and containment-stratified, not "near vertices"). Keep it stratified; do not let it become a separate claim without geometry labels. |
| KATRIN, fast forward model, surrogate, systematic error estimation, computational efficiency, interpretability | **Absent, and this is the real gap.** |

**On the KATRIN thread.** Junjie's list is weighted toward *systematic
uncertainty and interpretability* — carry system uncertainty, systematic error
estimation, surrogate forward models, computational efficiency. The proposal as
written offers him detection power, attribution F1, and a storage/latency
frontier. Those are monitoring concerns, not systematics concerns, and the
mismatch is the reason I would not expect this document alone to land with him.

Amendment 3 is the bridge. A surrogate or fast forward model's approximation
error *is* a declared N intervention; its consequence for the physics parameter
*is* the quantity a systematics budget needs. Framing one paragraph that way —
"the consequence axis measures what a declared model error costs the estimate,
which is the form a systematic-uncertainty budget takes for a surrogate-based
analysis" — connects your proposal to his stated interest without adding
KATRIN-specific work you cannot yet scope.

Two cautions carried forward:

1. **KATRIN has no material in this repository.** Grep finds no KATRIN, tritium,
   or surrogate work outside the noise module's convention documents. If KATRIN
   is the intended destination, say so as a *future direction* in the meeting
   rather than committing to it in the proposal, and confirm data and
   collaboration access before any scoping.
2. **Confirm the modality of Junjie's own pipeline.** The two-page proposal has
   quietly resolved the earlier Track A/Track B question by dropping Track B —
   both testbeds are now yours (NuBench public + your waveform task). That is a
   *good* de-risking move and makes the proposal executable without him. But it
   also means his role is currently undefined. Decide before the meeting whether
   you are asking him for (a) review only, (b) a third testbed, or (c)
   co-authorship on the consequence/systematics axis. The proposal reads as (a);
   his discussion list suggests he is interested in (c).

## 3.6 Recommended actions

**Before circulating (half a day):**

1. Amendment 1 — split the E gate out of the C2 macro-F1; restate the margin as
   a CI-based superiority test on {N, S}.
2. Amendment 2 — foreground N-vs-S as the discriminative core; promote the N+S
   mixture to the primary endpoint.
3. Amendment 3 — add the consequence axis to E1/E2 on the waveform arm, with the
   2×2 above as the intended figure.
4. Add one sentence on identifiability limits (quote Paper 1 `08_discussion.tex:120`).
5. Align the abstract's attribution language to C2's own caveat.

**Before the meeting:**

6. Run the V0 selection-hypothesis test (R4). Walking in with "the discrepancy is
   an event-selection difference, here is the evidence" converts your one open
   gate from a liability into a demonstration of exactly the diagnostic
   discipline the paper is about.
7. Decide what you are asking Junjie for, and put it in one line at the end.
8. Have the C1 power calculation ready.

**Keep unchanged:** the de-branded framing, the N/S/E taxonomy, the five-arm
feature-matched baseline design, the honest-open feasibility gate, the
non-claims, and the explicit statement that cross-geometry results test
consistency rather than causal detector-design effects. Those are the reasons
this proposal reads as credible, and none of the amendments above touches them.

---

# 4. Cross-cutting notes

**The arc, restated.** *Paper 1: the metric is Σ⁻¹, and using the wrong one
inverts your rankings. Paper 2: restricting the admissible projector class
trades subspace fidelity for estimation variance, by a computable amount.
Paper 3: after training, can you tell that the metric, the support, or the
evaluation contract has moved — and does the alarm correspond to any real
loss?* With Amendment 3 (§3.4) the final clause holds and Σ⁻¹ does work in all
three papers. Without it, Paper 3 is a good study that happens to share
materials with the first two.

**Reusable assets you are under-exploiting.** The Davis–Kahan perturbation lemma
is needed in all three papers (Paper 1 M6, Paper 2's bound, Paper 3's
consequence axis). State it once in Paper 1, cite it twice. Likewise Paper 1's
identifiability paragraph (`08_discussion.tex:120`) should be quoted, not
restated, in Papers 2 and 3 — and it is currently missing from the Paper 3
proposal entirely, where it answers a question Junjie asked directly.

**One habit to carry forward, one to drop.** Carry: predeclared gates with
exhaustive reporting of passes *and* failures — it is the reason Paper 1 is
credible, and the Paper 3 proposal already carries it forward correctly (the
open V0 gate is reported, not buried). Drop: reporting unnormalized,
scale-dependent differences as headline quantities. This bites Paper 1 (M2), and
Paper 3 should hold its own line here — "latent displacement is normalized by
clean within-class dispersion" is exactly right and must survive into the
manuscript.

**A predeclaration caution that now spans two papers.** Paper 1's predeclared
gates worked because the thresholds were about *direction* and *calibration
bands*, not about beating a baseline by a fixed margin. Paper 3's C2 breaks that
pattern with its ≥0.20 F1 margin (§3.3, R1), and Paper 2's planned sweeps risk
the same if "NFPA beats EMPCA" is predeclared as a point threshold rather than a
crossover location. Predeclare *shapes and directions*, with CIs on the
difference; reserve fixed numeric thresholds for calibration bands where a
target value is physically meaningful.

**Sequencing recommendation.** Post Paper 1 to arXiv within two weeks (B1 + M1 +
M2 are the gating edits; M3–M6 are one more week). Circulate the Paper 3
proposal *now*, after the half-day of amendments in §3.6 — it is the item with
an external dependency and the longest feedback latency, so it should not wait
on either manuscript. Paper 2's restructure is ~6–8 weeks including the E1/E2
runs and depends on nothing else. Run the V0 selection test (R4) before the
Paper 3 meeting rather than after.

---

## Appendix — verification record

| Checked | Method | Result |
|---|---|---|
| Bridge Theorem, all three conclusions | Numerical, random complex Gaussian, d=8/k=3/N=4000 | Correct; constraint, eigenproblem, minimum value, and optimality vs 300 random subspaces all verified |
| Paper 1 section structure | `grep '\section'` on `0*.tex` | 11 sections; **no Related Work** (M3) |
| Arithmetic in this review | Recomputed in Python | 1012.27/3.78 = 267.8×; EMPCA 614,400 vs NFPA 3,272 params (188×); k·CT = 3,072 > N=800; NuBench offsets 0.055°/0.035°; 0.01° = 0.042% of 23.65° |
| Paper 1 main-text word count | detex/strip on `0*.tex` | ~10.0k by this method; repo records ~12.3k (OPEN_ISSUES #3). Both exceed the 7–8k target |
| Placeholder present | `07_experiments.tex:331` | Confirmed literal `[PLACEHOLDER --- pending evidence...]` |
| KATRIN/tritium/surrogate material | `grep -ril` across repo excluding `.venv` | None outside noise-module convention docs |
| Paper 3 source of record | Full text of uploaded `paper3_proposal.pdf` (2 pp) extracted and read | Supersedes `PROPOSAL.md`; earlier "three competing proposals" finding withdrawn |
| Paper 3 quoted figures | Traced to `paper3_proposal.pdf` §2–§4 | 1,358,099 params; 128-D hook; 2,989,339 predictions; 23.6548/23.60, 56.1553/56.12; 0.944° median, 7.708° p95; 15.28°→25.12°; 10-NN 1.000→0.393; 8.82 GiB |
| Numbers quoted in this review | Traced to `07_experiments.tex`, `08_discussion.tex`, paper 2 PDF pp. 24–28, `paper3_proposal.pdf` | No number in this document is computed or inferred except the arithmetic row above; all others are quoted from source |

**Not verified:** appendix proofs (A–H) were not read line by line; the CRESST,
GWOSC, and TIDMAD run JSONs were not opened, so every quoted result is as
reported in the manuscript, not independently recomputed; Paper 2's NFPA
notebooks were not executed; the NuBench release, checkpoint, and rescoring
pipeline were not run, so all Paper 3 feasibility figures are as reported in the
proposal.

**Superseded in this revision:** the previous §3 (dated 2026-07-27) assessed
`PROPOSAL.md` and recommended reframing Paper 3 around noise-model
misspecification. That recommendation is withdrawn. The surviving element of it
is Amendment 3 (§3.4), which adds the consequence axis to the existing design
rather than replacing the design.
