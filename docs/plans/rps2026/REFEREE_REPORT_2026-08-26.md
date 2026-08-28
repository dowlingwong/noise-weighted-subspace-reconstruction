# Referee report — R/PS 2026 mini-paper

**Date:** 2026-08-26 · **Reviewed:** `paper_rps2026/` at `cd37380a17f2ed4f9881069a0a26ecab2d497271`
**Build reviewed:** `paper_rps2026/paper.pdf`, 11 pages (main 1–4, references 5–6, appendix 7–11),
330,966 bytes, built 2026-08-24 02:08 and newer than every source file.
**Mode:** read-only. No file in the repository was edited and no experiment was run.
This report is the only file created.

## HEAD is not the SHA the dispatch prompt expects — and it does not matter

The prompt expects `6b07df4`. HEAD is `cd37380`. `6b07df4` is an ancestor, and
`git diff --name-status 6b07df4 HEAD` is a single addition:
`docs/plans/rps2026/RPS_PAPER_REFEREE_AGENT_PROMPT.md`, the dispatch prompt
itself. **No manuscript, figure, table, script or result file differs between
the two.** Every page and line reference in the prompt is current.

I did not use the `paper-review` skill's file table, section names or symbol
grading. `main.tex`, `02_unified_objective.tex`, Theorems 4.1/5.1 and
`appH_symbols.tex` do not exist here and were not opened.

---

## 1. Orientation check

### The thesis, in my words

A reconstruction method is not an architecture — it is the pair
`(Σ⁻¹, F)`. Two independent choices are usually made as one: the **residual
metric**, which the measurement likelihood fixes (for additive Gaussian noise
the negative log-likelihood is the Mahalanobis form, so `Σ⁻¹` is the
likelihood-consistent inner product and MSE is the special case `Σ = σ²I`), and
the **admissible representation class** `F`, which the architecture fixes.
Because the metric comes from the instrument rather than from taste, measured
observations can supply their own reconstruction supervision without clean
targets — the covariance says which residual directions are informative. And
whether the metric choice *bites* is not an aesthetic question either: it is a
measurable joint property of the instrument's noise geometry and the
representation class, which the paper measures by holding `Σ⁻¹` fixed and
varying `F`.

**What is deliberately not claimed.** No new estimator (§1 says so in italics).
NFPA is a covariance-aware member of the established MPCA/Tucker family, not a
new factorization. Nothing claims NFPA beats CW-PCA intrinsically — and I
checked that the one place §4 compares them says the right thing: NFPA's
**held-out** residual is lower by 0.0026, which is finite-sample generalization,
not a restricted optimum beating an unrestricted one.

### The four classes

Labels, not a capacity ordering. Rung 2 (unrestricted rank-`k` linear,
CW-PCA / tied noise-aware linear AE) is the general case. **Rung 3 restricts rung
2** to Kronecker-separable factors — the embedded product
`Gr(k₁,n₁) × Gr(k₂,n₂) ↪ Gr(k₁k₂, n₁n₂)`, a strictly smaller family. **Rung 1 is
rung 3's degenerate fixed-template case** (`k₁ = k₂ = 1`, and only under
Corollary 1's isotropic-whitened-noise + separable-template + eigengap
assumptions). **Rung 4 is not comparable** — an analogous axis-factored
attention architecture in the same band–frame coordinates, with no containment
claimed or proved. I read §1 and §3 for ladder language and found none: §3 is
titled "Four representation classes", says the labels are "*not* a capacity
ordering", and §1 says the first three are nested and the fourth is analogous.
The text does not invite the misreading.

### Weak claim versus substantive claim

The weak claim is that a model trained under a metric leads under that metric —
near-tautological, and the paper says so. The substantive claim is
**truth-referenced**: at the exactly separable condition the covariance-weighted
factors recover the *planted* channel subspace almost exactly (0.035 projector
error against the isotropic baseline's 1.021), decode held-out planted signal
parameters better (amplitude RMSE 0.143 vs 0.198; decoded-amplitude correlation
0.70 vs 0.13), while **temporal recovery stays weak** (0.768 vs 1.280) — an
asymmetry no objective-relative argument explains, and one a joint number would
hide.

The abstract keeps this ordering correctly. **§1's contribution sentence does
not** — see blocking finding B1.

### The two conditions

- Primary reversal experiment: **α = 0.1**, `results/rps2026/E8_semantics/primary_alpha_0p1/`
  (and E1/E9 for the reversal itself).
- Truth-referenced recovery: **α = 0**, `results/rps2026/E8_semantics/separable_alpha_0/`.

**The current text does not conflate them.** I checked all five places the prompt
names: §4's setup paragraph states both explicitly and says they must not be
conflated; §4's recovery paragraph re-states α = 0 at the point of use; the
Figure 1 caption carries both; **both panel titles carry their α** ("A rung 3
reversal, primary condition α = 0.1" / "B recovery, exactly separable condition
α = 0"); and the Conclusion says "in the controlled setting" without attaching a
separability property to the primary condition. Nowhere in the built PDF is the
primary condition described as separable.

### Evidence classes

E10b is one frozen sensitivity condition at one calibration size
(`N_cal = 2048`) — the paper says "one fixed condition, not a sweep" in §4 and
"This is a single condition… no such sweep is claimed" in B.4. E11 is a post-hoc
diagnostic correction, appendix-only; I confirmed no angle claim reaches §1–§5
or the abstract. Rung 4 is a two-seed pilot, and §4's heading, its closing
sentence and the abstract all say so.

**My understanding did not diverge from Step 3 anywhere.** Where I found a
problem it is that one sentence of the paper diverges from Step 3, not that
Step 3 is wrong.

---

## 2. Blocking findings

Two, both one-edit fixes. I am calling them blocking because each is an
overclaim or a mismatch at a place a referee or a submission form will actually
look, not because they are large.

### B1 — §1's contribution sentence attaches the reversal's replication evidence to the recovery claim

**Claim.** `sections/01_introduction.tex:7`, last sentence, typeset at PDF
p1 ll.42–45:

> "Our contribution is a precise separation of measurement geometry from
> representation capacity, and the finding it makes testable: at matched capacity
> the covariance-weighted model **recovers planted structure the isotropic one
> does not, reproduced in a disjoint twenty-seed cohort and surviving finite
> covariance estimation**."

**Why it fails.** On the natural reading, "reproduced in a disjoint twenty-seed
cohort" and "surviving finite covariance estimation" modify *recovers planted
structure*. No such evidence exists. Planted-structure recovery is E8 only: ten
seeds, α = 0, oracle Σ. I checked the column schemas of both files the sentence
implicitly invokes:

```
$ head -1 results/rps2026/E9_blockA_replication/replication_raw.csv     | tr ',' '\n' | grep -i 'project|recover|amplitude|rise|decay'   -> (empty)
$ head -1 results/rps2026/E10b_calibrated_estimated_covariance/raw.csv | tr ',' '\n' | grep -i 'project|recover|amplitude|rise|decay'   -> (empty)
```

Neither E9 (the twenty-seed cohort) nor E10b (the estimated-covariance
condition) measures any recovery quantity at all. Both measure the **reversal**
— `test_normalized_chi2` and `test_raw_mse` at α = 0.1. So the sentence borrows
the weak claim's replication evidence to support the strong claim. That is an
evidence-class upgrade of exactly the kind `claim_audit.md` exists to prevent,
sitting in the paper's single contribution sentence.

**This is a compression artifact, not a position.** `ROUND10_PAPER_REVISION_QA.md`
("Third revision", item 3) records the intent precisely, and correctly:

> "What is not tautological is that the reversal appears on **held-out** data,
> extends to **semantic recovery**, replicates on an independent cohort,
> survives finite covariance estimation, and then **fails to appear** at rung 4.
> §1's contribution sentence now names that chain."

The chain is *about the reversal*, with semantic recovery as one link. In the
sentence as typeset, recovery became the grammatical subject and the other links
became its modifiers. The abstract still gets it right — it separates the two
claims with a semicolon and attributes the cohorts and the estimated covariance
to the ranking reversal. §1 should match the abstract.

**Cost to fix.** One sentence, no net line change, no risk to the four-page
gate. E.g. restore the chain's subject: "…and the finding it makes testable: at
matched capacity the covariance-weighted model's held-out ranking reverses —
reproduced in a disjoint twenty-seed cohort and surviving finite covariance
estimation — and, at the separable condition, it recovers planted structure the
isotropic one does not."

### B2 — the OpenReview abstract field does not match the PDF

**Claim.** `notes/openreview_metadata.md`, "Abstract (plain text, LaTeX
stripped)", says of itself: "transcribed from `sections/00_abstract.tex` with
LaTeX stripped **and the pair notation spelled out**… the field must match the
PDF." Its own submission-day checklist item 3 requires the field to be synced.

**Why it fails.** I extracted the abstract from the built PDF, normalised
whitespace and pdftotext hyphenation, and diffed it against the fenced block.
There is exactly one substantive difference, and it is not a spelling-out:

- PDF: "…we organize classical methods as a pair **(Σ⁻¹, F)** that separates…"
- Metadata block: "…we organize classical methods as a pair that separates…"

The notation was **deleted**, not spelled out. Pasted as-is, the OpenReview
abstract loses the paper's central object in its own summary. Word counts are
194 (PDF) vs 191 (block); the 2026-08-21 review's "word-identical, 182 words
each, machine-diffed" no longer holds against this build.

**Cost to fix.** Edit one line of a notes file. No rebuild.

---

## 3. Non-blocking findings, in the order a referee would care

**N1 — the abstract's "exact geometric characterization" drops the qualifier §1
carries.** `claim_audit.md` (Round 9 corrections) records: "Rungs 1–3 are
'closed-form' → **weakened** — exact geometric characterization; NFPA's joint
problem is nonconvex, subproblems exact", with the scope listed as "abstract,
§1". §1 carries the parenthetical ("the separable joint optimization is
nonconvex; its subproblems are exact"). The abstract says only "The linear
classes admit an exact geometric characterization." A referee who reads only the
abstract will believe rung 3's global optimum is characterised in closed form.
It is not: Proposition 2 characterises *when the restricted optimum equals the
unrestricted one*, not what it is. Three words would close it ("…exact geometric
characterization, up to a nonconvex joint fit"), if three words fit.

**N2 — Figure 2 is the paper's best explanatory asset and it is three pages from
the text that needs it.** §3's rung-3 paragraph is the densest passage in the
manuscript: `Σ = Σ₂ ⊗ Σ₁`, per-axis whitening, two bases, a code, a
reconstruction and a map back, in five lines. Figure 2 — the block-grid drawing
of `U₂ ⊗ U₁` against a free `P` — is what makes that followable, and it is on
page 7. §3 says "Figure 2 draws that restriction" with no signal that this is a
three-page turn into the appendix. For a reader who has not read the source
papers (the pass-3 test), that paragraph is the place the paper is hardest, and
the fix is sitting in the appendix. Two options: cheap — write "Figure 2, in the
appendix, draws that restriction"; expensive — move the schematic into §3, which
needs a compression round and re-runs the `\pdfendlink` segfault risk the QA
record documents. See §6 for my recommendation.

**N3 — Table 1's caption does not label which arm scored which.** The caption
says "rescoring without the rounding gives $0.0241$ vs.\ $0.0231$ in band". I
traced both:

- `E5_t_mse_inband_unrounded.json: raw_score_unrounded  = 0.024149077851187167`  → 0.0241 = **T-mse**
- `E5_t_chi2_inband_unrounded.json: raw_score_unrounded = 0.023130609616883405` → 0.0231 = **T-chi2**

Higher is better on this score, so the unrounded in-band rescoring favours the
**MSE-trained** arm. The caption's order matches the table's row order, so the
mapping is inferable — but it is not stated, and it points the opposite way from
§4's "the Σ⁻¹-trained arm is better on *both* criteria". The paper is not hiding
this (the caption says the contrast's bootstrap interval spans zero, and §4 says
the coarse score cannot separate the arms), but an unlabelled pair of numbers
that quietly cuts against the neighbouring sentence is the kind of thing a
hostile referee finds and calls selective. Name the arms.

**N4 — the rung-3/rung-4 dispersion comparison quotes the range statistic for
one rung only.** §4 says the rung-4 weights "span a factor of 441, with
coefficient of variation 0.69 — the same order as the rung-3 geometry where the
reversal *is* observed (1.10; white control 0)". Both CVs check out
(`E6_whitening_range.json: cv_inverse_J = 0.6914`,
`E6b_rung3_whitening_range.json: rung3_colored.cv_inverse_weights = 1.0964`,
`rung3_white = 0.0`), and CV is the right comparable statistic — `E6b`'s own
comment says so. But the *range* is quoted only for rung 4 (441×), and rung 3's
archived range is **154,710×** (`rung3_colored.ratio_max_min = 154709.84`). A
referee who opens the code release sees two statistics that disagree about
whether the two rungs are comparable, and the paper printed the one where they
are. Not dishonest — but if the sentence keeps 441, it is safer to either drop
the range or say plainly that the CV, not the range, is the comparable statistic
across bases.

**N5 — Proposition 2's "such a separable subspace" is not pinned to the given
(k₁, k₂) split.** As stated: "Its optimum equals the unrestricted CW-PCA optimum
if and only if at least one rank-`k₁k₂` CW-PCA minimizer has such a separable
subspace." A rank-`k₁k₂` subspace can be a tensor product under several splits,
and only the declared split is feasible for NFPA. I built the counterexample:
for `n₁ = 4, n₂ = 5`, a rank-4 subspace separable as (1,4) has partial trace over
the second axis of rank 1, so it is not (2,2)-separable, yet it satisfies a loose
reading of "separable". The (⇐) direction is unaffected; (⇒) is what needs the
pin, plus the (harmless, compactness-given) attainment of the restricted
infimum. One word fixes it: "…has such a `(k₁,k₂)`-separable subspace." Purely a
precision item; the proof and the experiments are unaffected.

**N6 — Appendix B.5's angles are the only printed numbers with no archived
result file.** Every other number in the manuscript traces to a file under
`results/`, `evidence/` or `transfer_paper/` (see §4 below). B.5's
`0.0°, 59.3°, 64.6°, 73.8°, 84.8°, 87.8°` (mean 61.7°), the IsoAE list
`82.1–89.4` (mean 86.0°), and `4.58 × 10⁻⁹ / 4.33 × 10⁻⁹` appear in exactly one
file in the repository — `paper_rps2026/plan/archive/CODE_AUDIT_2026-08-08.md:28`
— which is a plan document, not evidence. It records that the numbers were
independently reproduced on 2026-08-08 by re-running the deterministic
`src/NFPA/nfpa_demo.py` (`np.random.seed(42)`), which is a real provenance
chain, just a weaker class than everything else. B.5 explicitly labels itself "an
implementation sanity check, not as evidence", which is why this is not
blocking. But in a paper whose principal asset is digest-pinned provenance, one
paragraph of numbers sourced to a markdown note is the loose thread a
provenance-minded referee pulls.

**N7 — the QA record and the submission metadata describe the previous build.**
Both say "one **10-page**, 293 kB file" and "Appendix excluded, unlimited, after
references ✅ pages **7–10**". The current build is **11 pages, 330,966 bytes,
appendix 7–11** — the schematic commit `f068053` added a page on 2026-08-24,
three days after the 2026-08-21 anonymity, style and font audits were run.
Nothing in the artifact is wrong (the appendix is unlimited and the four-page
main-text gate still passes), but the audit of record no longer covers the
submitted PDF. **I re-ran both sweeps on the current build and they pass** — see
§4 — so this is a documentation fix, not a rebuild.

**N8 — the last three content commits are not in `claim_audit.md` or the QA
record.** `f068053` (add the schematic), `a665f8e` (correct the Kronecker factor
order in §3) and `6b07df4` (state Appendix A.3 in §3's notation) all changed
typeset mathematics or the figure inventory, and none appears in the claim
ledger or the revision QA. `notes/source_map.md`'s figure-provenance paragraph
still describes `geometry_schematic.pdf` as "an unused alternative" and does not
mention `figures/classes.tex`, which is the figure the paper now prints. The
ledger's own discipline is that it is kept current; three commits are outside it.

**N9 — the strongest rejection argument available to a hostile referee, and how
far the paper answers it.** I would write it as: *"Every quantitative result in
this paper comes from one generator configuration. `C = 8`, `T = 256`,
`(k_c,k_t) = (2,3)`, rank 6, SNR 10, Brownian 1/f² with correlated channels,
α ∈ {0, 0.1}. Thirty seeds are thirty draws from that one configuration, not
thirty conditions. The authors report `|d_z| > 50`, sign agreement 1.0, and a
white-noise control that is **exactly** 0.0 — which is precisely the signature
of an algebraic identity being measured, not an empirical effect being
discovered. The one real-data instance at the linear rung is explicitly not
replicated; the one real-data instance at rung 4 is a two-seed null the authors
say the score cannot resolve. Strip the framing and this is generalized least
squares plus multilinear PCA, verified on a simulator that was built to contain
the structure the method then recovers."*

The paper answers about two-thirds of this, and does it well:

- The **tautology** half is answered, and this is the paper's real move. The
  truth-referenced recovery cannot be explained by "each model wins under its own
  objective", and the **channel/temporal asymmetry** (0.035 vs 0.768) is the
  single most persuasive thing in the manuscript: a metric artefact would not
  recover one planted axis almost exactly and the other barely at all. Keep
  leaning on that asymmetry — it is better evidence than the reversal.
- The **novelty boundary** is answered by being drawn honestly and in advance.
  §1's italicised "We introduce no new estimator", the MPCA attribution inside
  Proposition 2's proof sketch, and the cut rows in `claim_audit.md` leave a
  "this is known linear algebra" referee with nothing to expose. That is the
  right play for a workshop.
- The **missing SSL comparator** is answered by §1's "so are context here, not
  comparators", which is exactly the one clause the 2026-08-21 review asked for.
- The **single-configuration** half is *disclosed* (B.1: "A full-length empirical
  study should extend this pilot across additional seeds, splits, noise regimes,
  ranks and model capacities"; §5's limitations; B.4's "no such sweep is
  claimed") but not *answered*, and cannot be in three days. Disclosure is the
  correct and only available move. If a reviewer raises it, the author response
  is B.1 verbatim plus the planted-structure asymmetry.

**N10 — §4 remains four dense paragraphs of numbers with little connective
tissue, and §5 is one eight-line stack of limitations.** The 2026-08-21 review
said this and it is still true. I am recording it as confirmed-and-unfixed rather
than as a new finding: it is the honest price of four pages, every paragraph in
§4 carries load, and I would not spend a deadline day on prose rhythm.

### Settled items I checked rather than re-raised

- **The workshop footer.** The QA record's reasoning is that `\@trackname` is
  consumed only inside the `\if@neuripsfinal` branch at
  `vendor/template/neurips_2026.sty:397`, with the submission branch at line 401
  hard-coding the generic notice. I did not recompile the vendored template, but
  I did verify the premise the argument rests on: our `neurips_2026.sty` and
  `vendor/template/neurips_2026.sty` are both
  `5820fe764d9a0667e2b60c97f87c9032f2ee2b0c12ee8357bc785e3b7fdc15f8`, so the
  file being reasoned about is the file being compiled. **The reasoning holds and
  I have nothing to add.** Not a defect. Fourth time asked; let it rest.
- **The companion preprint as a deanonymization channel.** Still live and still
  an author decision. I confirmed all five carried numbers are present in the
  built PDF (`48.419`, `47.869`, `0.550`, `86.25`, `0.1209`/`0.0880` — one
  occurrence each), so the search string the QA record describes does exist. I
  could not check whether the preprint has been posted: the
  `arXiv-v1-polished-extended-preprint` folder is not reachable from this
  session (see §5). Decide before uploading, not after.
- **The `priorThesis` attribution.** Resolved as an `ack` environment. I verified
  the submission build hides it: `Sjostrom`, `Imperial` and `acknowledg` are
  0 occurrences in the PDF text and 0 in the qpdf-decompressed bytes, and there
  is no bibliography entry. Closed.

---

## 4. What I machine-checked

Scripts were written to `~/refcheck/` in the session home — **outside** the
repository — and are listed here in full so they can be re-run.

### 4a. Algebra (numpy, `~/refcheck/algebra.py`, `~/refcheck/props.py`) — 30/30 pass

I did **not** verify any of this by eye. Both scripts print PASS/FAIL per claim.

**The Kronecker factor order, which `a665f8e` and `6b07df4` claim to have fixed
— treated as unverified and now verified correct:**

| Checked | Result |
|---|---|
| A.3 as printed: `vec(U₁U₁ᵀ Z U₂U₂ᵀ) = (U₂U₂ᵀ ⊗ U₁U₁ᵀ) vec(Z)`, column-stacking vec, at the §4 sizes `n₁=8, n₂=256, k₁=2, k₂=3` | **PASS** |
| the reversed order `(U₁U₁ᵀ ⊗ U₂U₂ᵀ)` is *not* equal — i.e. the identity actually discriminates order | **PASS** |
| §3's rung-4 sentence "`range(U_m) ⊗ range(U_f)`, `U_f` over bands, `U_m` over frames" agrees with A.3 given §3's own "frequency band × time frame" axis assignment | **PASS** |
| Figure 2's basis is `U₂ ⊗ U₁`, orthonormal, and `BBᵀ` equals the A.3 projector | **PASS** |
| Figure 2's verbal claim "every block is a single `U₁` motif rescaled by one entry of `U₂`" — checked block-by-block for all `n₂ × k₂` blocks | **PASS** |
| Figure 2's parameter counts: `n₁k₁ + n₂k₂ = 784` free vs `n₁n₂k₁k₂ = 12288` | **PASS** |
| A.4 / Corollary 1's `vec(v tᵀ) = t ⊗ v` — same convention, consistent with A.3 | **PASS** |

**§3, §2 and Appendix A.1–A.2:**

| Checked | Result |
|---|---|
| the A.3 map is symmetric, idempotent, and of rank exactly `k₁k₂` — i.e. an orthogonal projector | **PASS** |
| §3 rung 3's code/reconstruction pair is self-consistent: `U₁U₁ᵀZU₂U₂ᵀ = U₁ C U₂ᵀ` with `C = U₁ᵀZU₂` | **PASS** |
| §3 rung 2: `PᵀΣ⁻¹P = I` holds for `P = Σ^{1/2}W`, `WᵀW = I` | **PASS** |
| A.1: `Σ^{-1/2} ŝ = WWᵀ x̃` for `ŝ = PPᵀΣ⁻¹x` | **PASS** |
| `PPᵀΣ⁻¹` is idempotent and `Σ⁻¹`-self-adjoint | **PASS** |
| `ŝ` minimises `χ²` within its subspace (200 random perturbations, all worse) | **PASS** |
| the Σ⁻¹-orthonormality hypothesis is **load-bearing**: for a general basis the formula differs from `P(PᵀΣ⁻¹P)⁻¹PᵀΣ⁻¹x` — the 2026-08-19 review's fix was necessary, not cosmetic | **PASS** |
| A.2: `Cov(Σ^{-1/2}x) = Cov(Σ^{-1/2}s) + I`, and `+I` leaves all eigenvectors fixed | **PASS** |

**Proposition 2 and Corollary 1 (higher-order orthogonal iteration vs full-vec
eigendecomposition, 200 samples, multi-restart):**

| Checked | Result |
|---|---|
| (⇐) data generated inside a separable subspace ⇒ restricted and unrestricted optima coincide (`1.67e-28` vs `5.06e-28`) | **PASS** |
| the restricted optimum is never better than the unrestricted one | **PASS** |
| (⇒) generic data ⇒ strict gap (3055.27 vs 2809.58), and the unrestricted minimizing projector is verified not to be of Kronecker form | **PASS** |
| §3's "even a full-rank factor leaves the restriction active": at `k₁ = n₁` the fit gives `U₁U₁ᵀ = I` and the gap to unrestricted rank-`n₁k₂` is still strictly positive (465.02) — the Round 9 retraction is correct | **PASS** |
| Corollary 1: rank-one NFPA recovers `v/‖v‖` and `t/‖t‖` up to sign (\|cos\| = 0.999987, 0.999991) and its scalar code correlates \|r\| = 0.99999982 with the whitened matched-filter/OF amplitude | **PASS** |
| the (k₁,k₂) pinning gap of N5: a (1,4)-separable rank-4 subspace has partial-trace rank 1, hence is not (2,2)-separable | **PASS (finding N5)** |

### 4b. Every number in the main text, traced to its archived file

Recomputed from raw per-seed rows where possible — never re-derived from other
numbers in the paper. Seed-unit percentile bootstrap, 10,000 resamples,
`bootstrap_seed = 20260819` as recorded in the manifests.

| Paper location | Printed | Recomputed / archived | Source |
|---|---|---|---|
| §4 frozen, χ̄² | −0.0192 [−0.0194, −0.0190] | −0.019201 [−0.019411, −0.019003] | `E1_nfpa_blockA/block_A_colored_raw.csv` |
| §4 frozen, raw MSE | +0.0034 [+0.0034, +0.0034] | +0.003414 [+0.003394, +0.003434] | same |
| §4 replication, χ̄² | −0.0190 [−0.0192, −0.0189] | −0.019046 [−0.019167, −0.018924] | `E9_blockA_replication/replication_raw.csv` |
| §4 replication, raw MSE | +0.0034 | +0.003413 | same |
| §4 white control | "returns exactly zero" | max\|Δχ̄²\| = 0.000e+00, max\|ΔMSE\| = 0.000e+00, 20 seeds | `E2_nfpa_blockB/block_B_white_raw.csv` |
| §4 E10b degradation | 0.0078 % / 0.0345 % | +7.905e−05 (+0.0078 %) / +3.548e−06 (+0.0345 %) against NFPA's own archived oracle held-out values 1.013904285 / 0.010297200 | `E10b/raw.csv` |
| §4 E10b reversal | "all ten seeds" | Δχ̄² < 0 and ΔMSE > 0 in 10/10 | same |
| §4 rung 4, χ̄² | −1.15, −1.17 ×10⁻⁴ | −1.14794e−04, −1.16896e−04 | `E7_rung4_seed_replication.json` |
| §4 rung 4, raw MSE | −57.2, −57.4 | −57.184, −57.408 | same |
| §4 weight dispersion | 441×, cv 0.69 | `ratio_max_min = 441.275`, `cv_inverse_J = 0.69145` | `E6_whitening_range.json` |
| §4 rung-3 dispersion | 1.10; white 0 | 1.09642; 0.0 | `E6b_rung3_whitening_range.json` |
| §3 token axis | 328 tokens, 656 rows | `n_bands = 328`, 2 × 328 = 656 | same E6 file |
| §3 stationary-PSD fit | χ²/dof = 1.0031 | `global_model.chi2_per_dof_mean = 1.0031210` (and 1.000154 is the control — correctly **not** the number printed) | `transfer_paper/data/tidmad/current/TIDMAD_P4_real_seed20260727.json` |
| §4 recovery, channel | 0.035 vs 1.021 | 0.0349 vs 1.0207 | `E8_semantics/separable_alpha_0/raw.csv` |
| §4 recovery, temporal | 0.768 vs 1.280 | 0.7677 vs 1.2796 | same |
| §4 "all seven paired contrasts" | 7, two predeclared / five post-hoc | exactly 7 contrasts in the summary; all mean < 0, all `excludes_zero`, all `fraction_negative = 1.0` | `E8_semantics/separable_alpha_0/summary.json` |
| §4 config | `C=8, T=256, (2,3)`, rank 6, SNR 10, 800/400/2000, α = 0.1 | all match the frozen config row | `E1` CSV + `E9/manifest.json` |
| Figure 1A | −1.86 % / −1.84 %, +49.60 % / +49.58 % | −1.8586 % / −1.8440 %, +49.6070 % / +49.5793 % | recomputed per-seed from E1 and E9 |
| Figure 1B | 0.03 / 0.60 / 0.72 | 0.0349/1.0207 = 0.034, 0.7677/1.2796 = 0.600, 0.1434/0.1984 = 0.723 | E8 separable |
| B.2 effect sizes | −55.1/+99.5, −66.2/+71.2, −52.6/+93.6 | −55.1/+99.5, −66.2/+71.2, −52.6/+93.6 | recomputed from E1, E9, E10b raw rows |
| B.2 rung-2 contrast | −0.0026 [0.0025, 0.0026] | −0.002558 [−0.002592, −0.002521], negative in 10/10 | `E1` CSV |
| B.2 archived-fit reproduction | "≤ 10⁻⁹ relative" | max = 6.36e−15 over 50 rows | `E8_semantics/*/raw.csv` |
| B.2 restarts | "10 ALS restarts per seed" | `provenance.config.n_restarts = 10` | `E9/manifest.json` |
| B.2 α = 0.1 post-hoc | "agrees in direction on all seven contrasts" | 7/7, all `excludes_zero`, `analysis_status = post_hoc_exploratory` | `E8_semantics/primary_alpha_0p1/summary.json` |
| B.3 transfer | 1.15; 9.42, 8.83; 1.77, 7.29 | `local_radius_64s` (the predeclared primary) `score_std` = 1.1476 synthetic, 9.4242 H1, 8.8268 L1; `global_leave_one_out` 1.7713 H1, 7.2880 L1 | `transfer_paper/tables/gwosc_time_local_psd_summary.csv` |
| B.4 rung-2 benchmark | 48.419, 47.869, 0.550 [0.530, 0.571] | 48.41922, 47.86896, 0.55026 [0.53010, 0.57091] | `evidence/paper_bundle/20260726T050816Z_eb4592d72ad9/sweeps_v2/S5/S5_confirmatory_30seeds.{json,csv}` |
| B.4 subspace angle | 86.25° | mean `subspace_angle_deg_max` over 30 seeds = 86.2469° | same CSV |
| B.4 white-noise control angle | 1.48 × 10⁻⁶ ° | 1.4849e−06 | `evidence/paper_bundle/.../synthetic/S4_sweep_10seeds.json` |
| B.4 TIDMAD amplitude RMSE | 0.1209 → 0.0880 | 0.12091876 → 0.08798843 | `evidence/tidmad/20260727T063119Z/TIDMAD_P2_real_seed20260727.json` |
| Table 2 (all cells) | as printed | see 4c — table regenerates byte-identically | E10b + E11 |
| Table 1 `none` | 1.10 (1.1026) | 1.1026161164402752 | `results/rps2026/P2_none_anchor/benchmark_stdout.txt` |
| Table 1 linear Wiener | −1.8025 | archived upstream-benchmark stdout for `model_tag = wiener` | `transfer_paper/data/tidmad/current/TIDMAD_P5_file_seed20260727.json` |
| Table 1 `T-mse` / `T-chi2` | −2.7708 / −2.7708 | −2.7708098959837675 for both — the bit-identity the caption explains is real | `E5_t_mse_all20_stdout.txt`, `E5_t_chi2_all20_stdout.txt` |
| Table 1 unrounded in-band | 0.0241 / 0.0231 | 0.024149078 (T-mse) / 0.023130610 (T-chi2) | `E5_t_*_inband_unrounded.json` — see finding N3 |
| Table 1 published baselines | 6.55, 4.18, 3.84 | no archived file — see §5 | — |
| B.5 single-split angles | 0.0/59.3/…/87.8, 4.58e−09, 4.33e−09 | no archived result file — see finding N6 | — |

**No unsourced number in the main text.** The two gaps are both in the appendix
(Table 1's three published baselines and B.5), and both are discussed above.

### 4c. Table generation and evidence authentication

```
$ python3 scripts/make_rps_round10_table.py --out ~/refcheck/round10_regen.tex
$ diff ~/refcheck/round10_regen.tex paper_rps2026/tables/round10_sensitivity.tex
  -> BYTE-IDENTICAL
$ sha256sum results/rps2026/E10b_.../summary.json  -> 1846242438133ce52c1dd4adf2d78f34cfc34570c4748ac3e71f50f26b0d496b  [matches the pin]
$ sha256sum results/rps2026/E10b_.../raw.csv       -> f76c212f60e075a43196cd535f124262ccc6eb7f49bd5ce6734ff509851409ac  [matches]
$ sha256sum results/rps2026/E11_.../summary.json   -> d4f9d004430f5bc9cf55a347c33bced1448190d1e1c9cc8678b1841c22bd6170  [matches]
```

The generator also re-printed its own derivations, which match Table 2: E10b χ̄²
delta `7.904835e-05` / 0.0078 %, MSE delta `3.547906e-06` / 0.0345 %; E11 frozen
NFPA v2 62.8898, iso v2 86.7054 (difference −23.82 as printed), replication
63.0267 / 86.7345 (−23.71). I independently confirmed the E11 values against
`E11_corrected_angles/summary.json`, and that the legacy→v2 shift is 0.8059°
frozen — the "0.81°" the claim audit and Table 2's note record.

### 4d. Submission-readiness gates, re-run on the current 11-page build

| Gate | Result |
|---|---|
| four pages of main content, references from p5 | **pass** — p4 ends with Limitations, p5 begins the reference list, appendix begins p7 |
| overfull boxes | **0** (`paper.log`) |
| undefined references / citations | **0** |
| `\PENDING` in sources or PDF | **0** — the unused macro is also gone from `paper.tex` |
| build currency | `paper.pdf` (2026-08-24 02:08:29) is newer than every `.tex` source |
| style file unmodified | `neurips_2026.sty` = `vendor/template/neurips_2026.sty` = `5820fe76…c15f8` |
| anonymity, PDF text | 0 occurrences of dowling, wong, wongdowling, dwong, karlsruhe, kit.edu, delight, imperial, sjostrom, acknowledg, `/Users`, `/home`, `/ceph`, `.tex`, `.bib` |
| anonymity, qpdf-decompressed bytes (streams + font programs) | 0 occurrences of Dowling, Wong, dwong, Karlsruhe, Imperial, Sjostrom, DELight |
| Document Info dictionary | `/Title`, `/Author`, `/Subject`, `/Keywords` all empty; `/Creator (LaTeX with hyperref)`, `/Producer (pdfTeX-1.40.22)` |
| internal codes in the typeset paper | one occurrence each of `E10b` and `E11` — the deliberate traceability line in Table 2's caption; `Round 10` is 0 |
| repository / code link | none present, consistent with the code link being no-go. The only `github.com` in the PDF is inside the TIDMAD reference (the public dataset), which is not a deanonymization channel |
| fonts | DejaVuSans and DejaVuSans-Oblique are **CID TrueType, embedded and subset** — the Type 3 fix from the 2026-08-21 round survives the schematic commit. The two unnamed `[none]` Type 3 fonts remain and are the template's own, as the QA record established |
| **abstract field vs PDF** | **FAIL — see B2** |

The 2026-08-21 anonymity and font audits were run against the 10-page
pre-schematic build. Everything above is my own re-run against the 11-page build
that would actually be uploaded.

---

## 5. What I could not verify, and what I would have needed

1. **The long preprint is not reachable from this session.** Only
   `~/Documents/noise-weighted-subspace-reconstruction` is connected; there is no
   sibling `arXiv-v1-polished-extended-preprint` under it. I therefore could
   **not** perform the check `source_map.md` exists to enable — whether each
   compressed claim survived compression intact against its origin file
   (`02_unified_objective.tex`, `04_empca.tex`, `05_linear_ae.tex`,
   `06_loss_geometry.tex`, `03_nfpa.tex`, `07_experiments.tex`). What I did
   instead is stronger for numbers and weaker for framing: I traced numbers to
   result files rather than to the source paper's prose. *Needed:* the preprint
   folder connected as a second directory.
2. **Whether the companion preprint has been posted.** `ARXIV_CHECKLIST.md` is
   in that unreachable folder. The QA record's last reading was "not posted, so
   the channel is latent". *Needed:* same access, or a one-line answer from you.
3. **Table 1's three published baselines** (FCNet 6.55, Transformer 4.18, PUNet
   3.84). No archived file in this repository holds them; the source of record is
   `docs/plans/rps2026/PLAN_03_RUNG4_TIDMAD_2026-08-08.md`, which marks them
   "published". They are presumably the TIDMAD paper's own numbers and are
   verifiable by any referee — but **Table 1's caption says "from published
   baselines" and carries no citation**, so the referee has to go find which
   publication. `fry2024tidmad` is already in the bibliography and cited in §3;
   adding `\citep{fry2024tidmad}` to the caption costs nothing and closes it.
   *Needed:* the TIDMAD paper, to confirm the three values.
4. **Appendix B.5's angles** (finding N6). Confirming them means executing
   `src/NFPA/nfpa_demo.py`, which this task forbids. *Needed:* permission to run
   that one deterministic script, or an archived JSON for it.
5. **The full test suite.** The authoritative result is the server's (417 passed,
   9 skipped, recorded at `results/rps2026/Round10_remote/`). I ran no tests. The
   QA record states nothing since has touched experiment code, and
   `git diff 6b07df4 HEAD` confirms nothing at all has changed since.
6. **The vendored template's compile-time behaviour.** I did not run `latexmk` on
   `vendor/template/neurips_2026.tex` to re-confirm the footer. I verified the
   premise instead (byte-identical style file); the QA record's own reproduction
   recipe stands.

**Labelled as questions, not findings:** N3's reading of the Table 1 caption
assumes the caption's number order follows the table's row order — the archived
values are certain, the caption's intent is inferred. N5 is a wording gap I
believe is real but which no experiment in the paper depends on.

---

## 6. Deadline call — 29 Aug AoE, three days out

Page 4 has roughly four blank lines at the foot, so a fix of one to three lines
is affordable without a compression round. That budget shapes everything below.

### Fix — total time under an hour, one rebuild

1. **B1, §1's contribution sentence.** ~15 minutes, no net line change. This is
   the only thing in the paper I would call an overclaim, it sits in the sentence
   a referee reads first, and it contradicts your own abstract two inches above
   it. Fix it even if you fix nothing else.
2. **B2, the OpenReview abstract field.** ~5 minutes, no rebuild. Restore the
   pair notation, then re-diff against the rebuilt PDF as your own checklist
   item 3 requires.
3. **N3, name the arms in Table 1's caption.** ~5 minutes, appendix page, no
   four-page risk. "…gives 0.0241 (`T-mse`) vs. 0.0231 (`T-chi2`) in band."
4. **N2 cheap half: "Figure 2, in the appendix, draws that restriction."** Two
   words. Tells the reader the picture exists and where.
5. **N5, one word in Proposition 2**: "`(k₁,k₂)`-separable". Free.
6. **N1 if three words fit in the abstract**; skip if they do not — §1 already
   carries the qualifier, so the exposure is small.
7. **Add `\citep{fry2024tidmad}` to Table 1's caption** (§5 item 3). Free, and it
   closes the last unattributed numbers in the manuscript.
8. **Decide the companion-preprint hold before you upload.** Not a manuscript
   edit and not a referee's business, but it is the one open item with an
   irreversible failure mode, and the review window opens the day you submit. The
   cheapest option remains holding the preprint until 29 Sep notification; the
   third option (dropping the five carried numbers from B.4) also costs almost
   nothing, since they are superseded rung-2 evidence supporting no main-text
   claim. Either is fine. Doing neither, and posting during the window, is the
   one choice I would argue against.

Rebuild once after 1, 3, 4, 5, 6, 7 and re-run the four-page and anonymity gates.
Keep the `\usepackage[draft]{hyperref}` fallback in mind — item 4 adds words near
the §3 text that straddled the page-2/3 break when the `\pdfendlink` segfault
last fired.

### Leave

- **N7 and N8, the stale QA record and the unlogged commits.** Documentation
  debt, not artifact defects, and I re-ran the gates they cover. Reconcile them
  after submission, while it is still fresh — the ledger's value is that it is
  trusted, and three unlogged commits is how that erodes.
- **N2's expensive half, moving Figure 2 into §3.** It would make the paper
  meaningfully easier to read, and I still say don't. It needs a compression round
  to free a figure's worth of space three days before a deadline, on a document
  with a documented pdflatex crash at exactly that page boundary. The cheap
  pointer captures most of the benefit at none of the risk.
- **N4's dispersion asymmetry.** A referee has to open the code release to see
  it, and the code link is no-go for this submission. Revisit for camera-ready.
- **N6, B.5's provenance.** The paragraph already calls itself a sanity check and
  not evidence. Archiving a JSON for it is a post-submission task.
- **N9's single-configuration exposure and N10's §4 density.** Both real, neither
  fixable in three days, both already disclosed in the text. The sweep is the
  camera-ready or the full paper. Do not start it now.

### Things that are simply fine, and I want to say so plainly

Every number in the main text traces to an archived file and reproduces — most
to the printed digit, several exactly. The Kronecker factor order that survived
ten rounds of human reading is now **correct in all three places**, and I
verified that by running it rather than reading it. The Round 10 table
regenerates byte-identically from digest-pinned inputs. The α = 0.1 / α = 0
conflation is fixed everywhere the prompt asked me to look, including both panel
titles. The white-noise control really is exactly 0.000e+00. The anonymity sweep
is clean on the build that would be uploaded, not just on the one that was
audited. The evidence-class discipline — frozen versus replication never pooled,
E10b never called robustness, E11 confined to the appendix, rung 4 called a
two-seed pilot in its own heading — holds throughout the typeset paper.

The exposure on this submission is ambition, not correctness, and the paper has
already made the right trade by saying so first.


---

## Post-review fixes applied 2026-08-26 (author-directed)

All six items from §6 "Fix" were applied and the paper rebuilt once; every gate
re-run and passing (4 main pages, references from p5, 0 overfull, 0 undefined,
0 pdfendlink, anonymity sweep clean, empty PDF metadata, only the template's two
[none] Type 3 fonts).

1. **B1** — §1's contribution sentence now attributes the twenty-seed cohort and
   the estimated covariance to the held-out ranking reversal, with
   planted-structure recovery stated separately at the separable condition.
2. **B2** — the OpenReview abstract block restores the pair notation
   "(Σ⁻¹, F)"; re-diffed against the rebuilt PDF: 193 words both, matching.
3. **N3** — Table 1 caption names the arms: "$0.0241$ (\texttt{T-mse}) vs.\
   $0.0231$ (\texttt{T-chi2})".
4. **N2 (cheap half)** — §3 now says "Figure 2, in the appendix, draws that
   restriction."
5. **N5** — Proposition 2 pins the split: "$(k_1,k_2)$-separable subspace".
6. **Table 1 citation** — caption now carries \citep{fry2024tidmad}.

**Companion-preprint decision (author, 2026-08-26): drop the carried B.4
numbers.** Confirmed against the now-connected preprint folder that the
preprint is not posted (author metadata still gates it). The "Separate rung-2
benchmark" paragraph was de-numbered rather than deleted, so §5's
non-replication limitation keeps its referent: 48.419, 47.869, 0.550, 86.25°,
1.48e-6 and 0.1209→0.0880 no longer appear anywhere in the PDF, and "27.2%"
(also verbatim in the preprint) was avoided in favour of "roughly a quarter".

**Residual overlap, stated plainly:** two numbers shared verbatim with the
preprint's `07_experiments.tex` remain in the PDF because they are load-bearing
mini-paper evidence — χ²/dof = 1.0031 (§3, stationary-PSD adequacy) and 9.42
(B.3, H1 dispersion). Dropping B.4's numbers narrows the search channel; it does
not close it. If the preprint is posted during the 29 Aug → 29 Sep window, a
full-text search on either number still finds it. The hold-until-notification
question therefore remains live even after this edit.
